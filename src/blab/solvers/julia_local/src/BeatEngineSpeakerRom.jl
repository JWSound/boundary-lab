module BeatEngineSpeakerRom

using LinearAlgebra, SparseArrays, StaticArrays, Statistics

using ..BeatEngineCore
using ..BeatEngineCoupled

export build_parity_petrov_galerkin_rom

function _parity_sectors(symmetry::Symbol)
    symmetry == :off && return ((name="general", sign_x=1, sign_y=1, image_signs=(1,)),)
    symmetry == :x && return (
        (name="even_x", sign_x=1, sign_y=1, image_signs=(1, 1)),
        (name="odd_x", sign_x=-1, sign_y=1, image_signs=(1, -1)),
    )
    symmetry == :xy && return (
        (name="even_even", sign_x=1, sign_y=1, image_signs=(1, 1, 1, 1)),
        (name="odd_even", sign_x=-1, sign_y=1, image_signs=(1, -1, 1, -1)),
        (name="even_odd", sign_x=1, sign_y=-1, image_signs=(1, 1, -1, -1)),
        (name="odd_odd", sign_x=-1, sign_y=-1, image_signs=(1, -1, -1, 1)),
    )
    error("Speaker ROM parity symmetry must be off, x, or xy.")
end

function _reflection_map(points, axis::Int)
    mins = [minimum(point[index] for point in points) for index in 1:3]
    maxs = [maximum(point[index] for point in points) for index in 1:3]
    extent = maximum(maxs .- mins)
    tolerance = max(extent * 2.0e-5, 1.0e-7)
    center = (mins[axis] + maxs[axis]) / 2
    key(point) = ntuple(
        index -> round(Int, (Float64(point[index]) - mins[index]) / tolerance),
        3,
    )
    index_by_key = Dict(key(point) => index for (index, point) in enumerate(points))
    mapping = Vector{Int}(undef, length(points))
    for (index, point) in enumerate(points)
        reflected = collect(Float64.(point))
        reflected[axis] = 2 * center - reflected[axis]
        mapped = get(index_by_key, key(reflected), 0)
        mapped > 0 || error("Could not reflect speaker ROM node $index on axis $axis.")
        maximum(abs.(Float64.(points[mapped]) .- reflected)) <= 2 * tolerance || error(
            "Speaker ROM reflection exceeded tolerance at node $index on axis $axis.",
        )
        mapping[index] = mapped
    end
    all(mapping[mapping[index]] == index for index in eachindex(mapping)) || error(
        "Speaker ROM reflection on axis $axis is not involutory.",
    )
    return mapping
end

function _face_reflection_map(mesh, node_map)
    index_by_vertices = Dict{NTuple{3,Int},Int}(
        Tuple(sort(collect(Int.(face)))) => index
        for (index, face) in enumerate(mesh.faces)
    )
    mapping = Vector{Int}(undef, length(mesh.faces))
    for (index, face) in enumerate(mesh.faces)
        key = Tuple(sort([node_map[Int(vertex)] for vertex in face]))
        mapped = get(index_by_vertices, key, 0)
        mapped > 0 || error("Could not reflect speaker ROM BEM face $index.")
        mapping[index] = mapped
    end
    return mapping
end

function _image_maps(points, symmetry::Symbol)
    identity_map = collect(eachindex(points))
    symmetry == :off && return [identity_map]
    map_x = _reflection_map(points, 1)
    symmetry == :x && return [identity_map, map_x]
    map_y = _reflection_map(points, 2)
    return [identity_map, map_x, map_y, map_x[map_y]]
end

function _face_image_maps(mesh, node_image_maps)
    return [
        image == 1 ? collect(eachindex(mesh.faces)) :
        _face_reflection_map(mesh, node_image_maps[image])
        for image in eachindex(node_image_maps)
    ]
end

function _orbits(image_maps)
    visited = falses(length(first(image_maps)))
    rows = Vector{Int}[]
    sizes = Int[]
    for index in eachindex(first(image_maps))
        visited[index] && continue
        images = [mapping[index] for mapping in image_maps]
        unique_images = unique(images)
        representative = minimum(unique_images)
        representative == index || continue
        push!(rows, images)
        push!(sizes, length(unique_images))
        visited[unique_images] .= true
    end
    all(visited) || error("Speaker ROM reflection orbits do not cover the boundary.")
    return rows, sizes
end

function _parity_project(values, image_maps, image_signs)
    result = zeros(eltype(values), size(values))
    for (mapping, sign) in zip(image_maps, image_signs)
        result .+= sign .* values[mapping, :]
    end
    return result ./ length(image_maps)
end

function _compact_parity_values(values, orbits, image_signs)
    result = similar(values, length(orbits), size(values, 2))
    for (row, orbit) in enumerate(orbits)
        for column in axes(values, 2)
            result[row, column] = sum(
                image_signs[image] * values[orbit[image], column]
                for image in eachindex(orbit)
            ) / length(orbit)
        end
    end
    return result
end

function _reconstruct_parity_values(compact, orbits, image_signs, full_count)
    result = zeros(eltype(compact), full_count, size(compact, 2))
    for (row, orbit) in enumerate(orbits), image in eachindex(orbit)
        target = orbit[image]
        result[target, :] .= image_signs[image] .* view(compact, row, :)
    end
    return result
end

function _normalize_columns!(values)
    for column in axes(values, 2)
        scale = norm(view(values, :, column))
        scale > eps(real(one(eltype(values)))) || error(
            "Speaker ROM training produced a numerically zero parity sample.",
        )
        view(values, :, column) ./= scale
    end
    return values
end

function _sample_patterns(points, wavenumber, count::Int, offset::Int, ::Type{T}) where {T}
    mins = T[minimum(point[index] for point in points) for index in 1:3]
    maxs = T[maximum(point[index] for point in points) for index in 1:3]
    center = (mins .+ maxs) ./ T(2)
    half_extent = (maxs .- mins) ./ T(2)
    scale = maximum(maxs .- mins)
    golden_angle = T(pi * (3 - sqrt(5)))
    patterns = zeros(Complex{T}, length(points), count)
    for column in 1:count
        sample_index = column + offset
        z = T(1) - T(2) * T(mod(sample_index * 0.6180339887498949, 1.0))
        radial = sqrt(max(zero(T), one(T) - z^2))
        phi = golden_angle * T(sample_index)
        direction = T[radial * cos(phi), radial * sin(phi), z]
        if isodd(sample_index)
            for (row, point) in enumerate(points)
                patterns[row, column] = cis(wavenumber * dot(T.(point) .- center, direction))
            end
        else
            exit_distance = minimum(
                half_extent[index] / max(abs(direction[index]), T(1.0e-4))
                for index in 1:3
            )
            levels = T[0.02, 0.05, 0.10, 0.20, 0.40, 0.80]
            source = center .+ (exit_distance + levels[mod1(sample_index, 6)] * scale) .* direction
            for (row, point) in enumerate(points)
                distance = norm(T.(point) .- source)
                patterns[row, column] = cis(wavenumber * distance) / distance
            end
        end
    end
    return patterns
end

function _factor(system, matrix)
    if system.linear_backend == :cuda
        cuda = BeatEngineCore.cuda_module()
        storage = cuda.CuArray(matrix)
        factorization = lu!(storage)
        cuda.synchronize()
        return (backend=:cuda, factorization=factorization, storage=storage)
    end
    return (backend=:cpu, factorization=lu!(matrix), storage=nothing)
end

function _solve(factor, rhs)
    if factor.backend == :cuda
        cuda = BeatEngineCore.cuda_module()
        device_rhs = cuda.CuArray(rhs)
        device_solution = nothing
        try
            device_solution = factor.factorization \ device_rhs
            cuda.synchronize()
            return Array(device_solution)
        finally
            cuda.unsafe_free!(device_rhs)
            isnothing(device_solution) || cuda.unsafe_free!(device_solution)
        end
    end
    return factor.factorization \ rhs
end

function _release_factor!(factor)
    factor.backend == :cuda || return nothing
    BeatEngineCore.cuda_module().unsafe_free!(factor.storage)
    return nothing
end

function _right_hand_side(system, layout, pressure)
    T = system.scalar_type
    rhs = zeros(Complex{T}, layout.state_count, size(pressure, 2))
    rhs[layout.flux_range, :] .= system.interface_operators.bem_trace * pressure
    if !isempty(layout.mechanical_range)
        rhs[layout.mechanical_range, :] .=
            -transpose(system.transducer_operators.bem_force) * pressure
    end
    return rhs
end

function _boundary_output(system, layout, state)
    T = system.scalar_type
    result = system.interface_operators.bem_flux * view(state, layout.flux_range, :)
    if !isempty(layout.mechanical_range)
        scale = Complex{T}(0, system.density * system.omega)
        result .+= scale .* (
            system.transducer_operators.bem_normal_velocity *
            view(state, layout.mechanical_range, :)
        )
    end
    return Matrix(result)
end

function _left_hand_side(system, layout, face_test)
    T = system.scalar_type
    rhs = zeros(Complex{T}, layout.state_count, size(face_test, 2))
    rhs[layout.flux_range, :] .= adjoint(system.interface_operators.bem_flux) * face_test
    if !isempty(layout.mechanical_range)
        scale = Complex{T}(0, system.density * system.omega)
        rhs[layout.mechanical_range, :] .= conj(scale) .* (
            adjoint(system.transducer_operators.bem_normal_velocity) * face_test
        )
    end
    return rhs
end

function _input_sensitivity(system, layout, state)
    result = -adjoint(system.interface_operators.bem_trace) *
             view(state, layout.flux_range, :)
    if !isempty(layout.mechanical_range)
        result .+= system.transducer_operators.bem_force *
                   view(state, layout.mechanical_range, :)
    end
    return Matrix(result)
end

function _snapshot_coefficients(observations, rank::Int, ::Type{T}) where {T}
    analysis = ComplexF64.(observations)
    decomposition = svd(analysis; full=false)
    singular_values = decomposition.S
    cutoff = max(first(singular_values), eps(Float64)) * 1.0e-6
    supported_rank = count(value -> value > cutoff, singular_values)
    effective_rank = min(rank, supported_rank)
    effective_rank > 0 || error(
        "Speaker ROM training snapshots do not contain a numerically supported mode.",
    )
    return (
        Complex{T}.(decomposition.V[:, 1:effective_rank]),
        singular_values .^ 2,
        effective_rank,
    )
end

function _biorthogonalize(right_basis, left_basis, rank::Int, ::Type{T}) where {T}
    overlap = ComplexF64.(adjoint(left_basis) * right_basis)
    decomposition = svd(overlap)
    minimum_value = decomposition.S[rank]
    maximum_value = first(decomposition.S)
    minimum_value > max(maximum_value, eps(Float64)) * 1.0e-10 || error(
        "Speaker ROM Petrov overlap is rank deficient at rank $rank " *
        "(condition estimate $(maximum_value / max(minimum_value, eps(Float64)))).",
    )
    inverse_root = Diagonal(Complex{T}.(inv.(sqrt.(decomposition.S[1:rank]))))
    right = right_basis * Complex{T}.(decomposition.V[:, 1:rank]) * inverse_root
    left = left_basis * Complex{T}.(decomposition.U[:, 1:rank]) * inverse_root
    return right, left, maximum_value / minimum_value
end

function _input_matrices(system, layout, excitations)
    T = system.scalar_type
    input_count = length(excitations)
    b_matrix = zeros(Complex{T}, layout.state_count, input_count)
    e_matrix = zeros(Complex{T}, length(system.bem_mesh.faces), input_count)
    for (column, excitation) in enumerate(excitations)
        kind = Symbol(excitation.kind)
        if kind == :voltage
            index = Int(excitation.transducer_index)
            b_matrix[first(layout.electrical_range) + index - 1, column] = one(Complex{T})
        elseif kind == :normal_velocity && Int(get(excitation, :bem_source_index, 0)) > 0
            source_index = Int(excitation.bem_source_index)
            e_matrix[:, column] .= view(system.prescribed_bem_neumann, :, source_index)
            isempty(get(excitation, :fem_boundary_tags, Int[])) || error(
                "Parity ROM export does not yet support FEM prescribed-velocity inputs.",
            )
        else
            error("Parity ROM export currently supports voltage and exterior prescribed-velocity inputs.")
        end
    end
    return b_matrix, e_matrix
end

function _curve_errors(exact, candidate)
    errors = Float64[]
    for column in axes(exact, 2)
        scale = max(norm(view(exact, :, column)), eps(real(one(eltype(exact)))))
        push!(errors, norm(view(candidate, :, column) - view(exact, :, column)) / scale)
    end
    return Dict(
        "median" => median(errors),
        "p95" => quantile(errors, 0.95),
        "maximum" => maximum(errors),
    )
end

"""
    build_parity_petrov_galerkin_rom(system, k_matrix, layout, excitations; ...)

Build one, two, or four rank-at-most-`rank` projection models for full-domain,
X-symmetric, or XY-symmetric cabinets. The right space is selected from exact
state responses using boundary-flux POD and the operator-induced left space
forms the Petrov projection. Frequencies or parity sectors with fewer
numerically supported modes are padded with decoupled states so the serialized
package retains a fixed `rank` dimension.
"""
function build_parity_petrov_galerkin_rom(
    system,
    k_matrix,
    layout,
    excitations;
    rank::Int=32,
    training_count::Int=max(96, 3 * rank),
    validation_count::Int=24,
    symmetry::Symbol=:xy,
)
    rank > 0 || error("Speaker ROM rank must be positive.")
    training_count >= rank || error("Speaker ROM training count must be at least its rank.")
    validation_count > 0 || error("Speaker ROM validation count must be positive.")
    symmetry in (:off, :x, :xy) || error("Speaker ROM parity symmetry must be off, x, or xy.")
    T = system.scalar_type
    sectors = _parity_sectors(symmetry)
    node_image_maps = _image_maps(system.bem_mesh.vertices, symmetry)
    face_image_maps = _face_image_maps(system.bem_mesh, node_image_maps)
    node_orbits, node_orbit_sizes = _orbits(node_image_maps)
    face_orbits, _face_orbit_sizes = _orbits(face_image_maps)
    pressure_training = _sample_patterns(
        system.bem_mesh.vertices,
        system.wavenumber,
        training_count,
        0,
        T,
    )
    pressure_validation = _sample_patterns(
        system.bem_mesh.vertices,
        system.wavenumber,
        validation_count,
        100003,
        T,
    )
    b_matrix, e_matrix = _input_matrices(system, layout, excitations)

    right_factor = _factor(system, copy(k_matrix))
    left_factor = _factor(system, Matrix(adjoint(k_matrix)))
    sector_models = NamedTuple[]
    validation = Dict{String,Any}[]
    effective_ranks = Int[]
    try
        driven_state = _solve(right_factor, b_matrix)
        driven_boundary_output = _boundary_output(system, layout, driven_state) .+ e_matrix
        driven_velocity_output = isempty(layout.mechanical_range) ?
                                 zeros(Complex{T}, 0, size(b_matrix, 2)) :
                                 Matrix(view(driven_state, layout.mechanical_range, :))
        driven_current_output = isempty(layout.electrical_range) ?
                                zeros(Complex{T}, 0, size(b_matrix, 2)) :
                                Matrix(view(driven_state, layout.electrical_range, :))
        for sector in sectors
            training_pressure = _normalize_columns!(_parity_project(
                pressure_training,
                node_image_maps,
                sector.image_signs,
            ))
            validation_pressure = _normalize_columns!(_parity_project(
                pressure_validation,
                node_image_maps,
                sector.image_signs,
            ))
            right_snapshots = _solve(
                right_factor,
                _right_hand_side(system, layout, training_pressure),
            )
            right_observations = _boundary_output(system, layout, right_snapshots)
            right_coefficients, right_spectrum, effective_rank = _snapshot_coefficients(
                right_observations,
                rank,
                T,
            )
            effective_rank < rank && @warn(
                "Speaker ROM sector uses fewer modes than requested",
                sector=sector.name,
                requested_rank=rank,
                effective_rank=effective_rank,
            )
            push!(effective_ranks, effective_rank)
            right_seed = right_snapshots * right_coefficients
            right_basis = Complex{T}.(Matrix(qr(right_seed).Q[:, 1:effective_rank]))
            # Kᴴ W = V gives Wᴴ K V = Vᴴ V. This operator-induced Petrov
            # space avoids the poorly conditioned overlap produced by two
            # independently truncated snapshot spaces while retaining the
            # response-informed right space.
            left_basis = _solve(left_factor, right_basis)
            petrov_identity = adjoint(left_basis) * k_matrix * right_basis
            overlap_condition = cond(ComplexF64.(petrov_identity))

            reduced_k = adjoint(left_basis) * k_matrix * right_basis
            c_full = -adjoint(view(left_basis, layout.flux_range, :)) *
                     system.interface_operators.bem_trace
            if !isempty(layout.mechanical_range)
                c_full .+= adjoint(view(left_basis, layout.mechanical_range, :)) *
                           transpose(system.transducer_operators.bem_force)
            end
            d_full = _boundary_output(system, layout, right_basis)
            # Projection bases inherit parity only up to the Float32 condensed
            # solve tolerance. Enforce the declared sector before storing one
            # representative per orbit; otherwise tiny forbidden components in
            # tail modes are amplified by quarter-boundary reconstruction.
            c_full = Matrix(transpose(_parity_project(
                Matrix(transpose(c_full)),
                node_image_maps,
                sector.image_signs,
            )))
            d_full = _parity_project(
                d_full,
                face_image_maps,
                sector.image_signs,
            )
            # The isolated electrical/source response is stored exactly as a
            # direct affine term. The reduced state is reserved for pressure-
            # induced loading feedback, which is the response family used to
            # train and validate this basis.
            reduced_b = zeros(Complex{T}, effective_rank, size(b_matrix, 2))
            projected_e = _parity_project(
                driven_boundary_output,
                face_image_maps,
                sector.image_signs,
            )
            compact_c = zeros(Complex{T}, effective_rank, length(node_orbits))
            for (column, (orbit, orbit_size)) in enumerate(zip(node_orbits, node_orbit_sizes))
                compact_c[:, column] .= orbit_size .* view(c_full, :, orbit[1])
            end
            compact_d = Matrix(d_full[[orbit[1] for orbit in face_orbits], :])
            compact_e = Matrix(projected_e[[orbit[1] for orbit in face_orbits], :])
            velocity_output = isempty(layout.mechanical_range) ?
                              zeros(Complex{T}, 0, effective_rank) :
                              Matrix(view(right_basis, layout.mechanical_range, :))
            current_output = isempty(layout.electrical_range) ?
                             zeros(Complex{T}, 0, effective_rank) :
                             Matrix(view(right_basis, layout.electrical_range, :))
            velocity_drive = sector.sign_x == 1 && sector.sign_y == 1 ?
                             driven_velocity_output :
                             zeros(Complex{T}, size(driven_velocity_output))
            current_drive = sector.sign_x == 1 && sector.sign_y == 1 ?
                            driven_current_output :
                            zeros(Complex{T}, size(driven_current_output))

            exact_validation_state = _solve(
                right_factor,
                _right_hand_side(system, layout, validation_pressure),
            )
            exact_validation_output = _boundary_output(
                system,
                layout,
                exact_validation_state,
            )
            compact_pressure = _compact_parity_values(
                validation_pressure,
                node_orbits,
                sector.image_signs,
            )
            reduced_state = reduced_k \ (-compact_c * compact_pressure)
            compact_output = compact_d * reduced_state
            candidate_output = _reconstruct_parity_values(
                compact_output,
                face_orbits,
                sector.image_signs,
                length(system.bem_mesh.faces),
            )
            push!(
                validation,
                Dict(
                    "sector" => sector.name,
                    "requested_rank" => rank,
                    "effective_rank" => effective_rank,
                    "boundary_output_error" => _curve_errors(
                        exact_validation_output,
                        candidate_output,
                    ),
                    "petrov_overlap_condition" => overlap_condition,
                    "right_snapshot_tail_ratio" => sqrt(
                        right_spectrum[min(rank, length(right_spectrum))] /
                        first(right_spectrum),
                    ),
                    "petrov_identity_error" => norm(
                        petrov_identity -
                        Matrix{Complex{T}}(I, effective_rank, effective_rank),
                    ) / sqrt(T(effective_rank)),
                ),
            )

            # Deploy packages use a fixed state dimension across every
            # frequency and sector. Keep the supported model in the leading
            # block and make any remaining slots inert but nonsingular so the
            # Deploy Schur factorization can use the existing package layout.
            padded_k = Matrix{Complex{T}}(I, rank, rank)
            padded_k[1:effective_rank, 1:effective_rank] .= reduced_k
            padded_c = zeros(Complex{T}, rank, size(compact_c, 2))
            padded_c[1:effective_rank, :] .= compact_c
            padded_d = zeros(Complex{T}, size(compact_d, 1), rank)
            padded_d[:, 1:effective_rank] .= compact_d
            padded_b = zeros(Complex{T}, rank, size(reduced_b, 2))
            padded_b[1:effective_rank, :] .= reduced_b
            padded_velocity = zeros(Complex{T}, size(velocity_output, 1), rank)
            padded_velocity[:, 1:effective_rank] .= velocity_output
            padded_current = zeros(Complex{T}, size(current_output, 1), rank)
            padded_current[:, 1:effective_rank] .= current_output
            push!(
                sector_models,
                (
                    k=padded_k,
                    c=padded_c,
                    d=padded_d,
                    b=padded_b,
                    e=compact_e,
                    velocity=padded_velocity,
                    current=padded_current,
                    velocity_drive=velocity_drive,
                    current_drive=current_drive,
                ),
            )
        end
    finally
        _release_factor!(right_factor)
        _release_factor!(left_factor)
    end

    stack(field) = cat((getproperty(model, field) for model in sector_models)...; dims=ndims(getproperty(first(sector_models), field)) + 1)
    # Move the appended sector dimension to the front for stable package axes.
    sector_first(field) = permutedims(
        stack(field),
        (ndims(getproperty(first(sector_models), field)) + 1, 1:ndims(getproperty(first(sector_models), field))...),
    )
    metadata = Dict{String,Any}(
        "format_version" => 1,
        "method" => "response_pod_with_operator_induced_petrov_test_space",
        "symmetry_mode" => String(symmetry),
        "image_count" => length(node_image_maps),
        "rank_per_sector" => rank,
        "effective_rank_per_sector" => effective_ranks,
        "training_count_per_sector" => training_count,
        "validation_count_per_sector" => validation_count,
        "sector_names" => [sector.name for sector in sectors],
        "sector_signs" => [[sector.sign_x, sector.sign_y] for sector in sectors],
        "node_orbits" => [[index - 1 for index in orbit] for orbit in node_orbits],
        "face_orbits" => [[index - 1 for index in orbit] for orbit in face_orbits],
        "input_port_count" => length(excitations),
        "transducer_count" => length(system.transducers),
        "validation" => validation,
        "equations" => [
            "K_r a + C_r P_parity p = B_r u",
            "q = sum_parity R_parity (D_r a + E_exact,r u)",
        ],
    )
    return Dict(
        "speaker_rom_k" => sector_first(:k),
        "speaker_rom_c" => sector_first(:c),
        "speaker_rom_d" => sector_first(:d),
        "speaker_rom_b" => sector_first(:b),
        "speaker_rom_e" => sector_first(:e),
        "speaker_rom_velocity" => sector_first(:velocity),
        "speaker_rom_current" => sector_first(:current),
        "speaker_rom_velocity_drive" => sector_first(:velocity_drive),
        "speaker_rom_current_drive" => sector_first(:current_drive),
        "metadata" => metadata,
    )
end

end
