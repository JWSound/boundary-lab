module BeatEngineSpeakerRom

using LinearAlgebra, SparseArrays, StaticArrays, Statistics

using ..BeatEngineCore
using ..BeatEngineCoupled

export build_parity_petrov_galerkin_rom

const PARITY_SECTORS = (
    (name="even_even", sign_x=1, sign_y=1),
    (name="odd_even", sign_x=-1, sign_y=1),
    (name="even_odd", sign_x=1, sign_y=-1),
    (name="odd_odd", sign_x=-1, sign_y=-1),
)

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

function _orbits(map_x, map_y)
    visited = falses(length(map_x))
    rows = NTuple{4,Int}[]
    sizes = Int[]
    for index in eachindex(map_x)
        visited[index] && continue
        images = (index, map_x[index], map_y[index], map_x[map_y[index]])
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

function _parity_project(values, map_x, map_y, sign_x, sign_y)
    map_xy = map_x[map_y]
    return (
        values .+
        sign_x .* values[map_x, :] .+
        sign_y .* values[map_y, :] .+
        (sign_x * sign_y) .* values[map_xy, :]
    ) ./ 4
end

function _compact_parity_values(values, orbits, sign_x, sign_y)
    result = similar(values, length(orbits), size(values, 2))
    signs = (1, sign_x, sign_y, sign_x * sign_y)
    for (row, orbit) in enumerate(orbits)
        for column in axes(values, 2)
            result[row, column] = sum(
                signs[image] * values[orbit[image], column] for image in 1:4
            ) / 4
        end
    end
    return result
end

function _reconstruct_parity_values(compact, orbits, sign_x, sign_y, full_count)
    result = zeros(eltype(compact), full_count, size(compact, 2))
    signs = (1, sign_x, sign_y, sign_x * sign_y)
    for (row, orbit) in enumerate(orbits), image in 1:4
        target = orbit[image]
        result[target, :] .= signs[image] .* view(compact, row, :)
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
    decomposition = eigen(Hermitian(analysis' * analysis))
    order = sortperm(real.(decomposition.values); rev=true)
    values = max.(real.(decomposition.values[order]), 0.0)
    values[rank] > max(first(values), eps(Float64)) * 1.0e-12 || error(
        "Speaker ROM training snapshots do not support requested rank $rank.",
    )
    return Complex{T}.(decomposition.vectors[:, order[1:rank]]), values
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

Build four rank-`rank` two-sided projection models. The right space is selected
from exact state responses using boundary-flux POD; the left space is selected
from adjoint responses using boundary-pressure sensitivity POD. The bases are
then biorthogonalized before projecting K/C/D/B.
"""
function build_parity_petrov_galerkin_rom(
    system,
    k_matrix,
    layout,
    excitations;
    rank::Int=32,
    training_count::Int=max(96, 3 * rank),
    validation_count::Int=24,
)
    rank > 0 || error("Speaker ROM rank must be positive.")
    training_count >= rank || error("Speaker ROM training count must be at least its rank.")
    validation_count > 0 || error("Speaker ROM validation count must be positive.")
    T = system.scalar_type
    node_map_x = _reflection_map(system.bem_mesh.vertices, 1)
    node_map_y = _reflection_map(system.bem_mesh.vertices, 2)
    face_map_x = _face_reflection_map(system.bem_mesh, node_map_x)
    face_map_y = _face_reflection_map(system.bem_mesh, node_map_y)
    node_orbits, node_orbit_sizes = _orbits(node_map_x, node_map_y)
    face_orbits, _face_orbit_sizes = _orbits(face_map_x, face_map_y)
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
    try
        driven_state = _solve(right_factor, b_matrix)
        driven_boundary_output = _boundary_output(system, layout, driven_state) .+ e_matrix
        driven_velocity_output = isempty(layout.mechanical_range) ?
                                 zeros(Complex{T}, 0, size(b_matrix, 2)) :
                                 Matrix(view(driven_state, layout.mechanical_range, :))
        driven_current_output = isempty(layout.electrical_range) ?
                                zeros(Complex{T}, 0, size(b_matrix, 2)) :
                                Matrix(view(driven_state, layout.electrical_range, :))
        for sector in PARITY_SECTORS
            training_pressure = _normalize_columns!(_parity_project(
                pressure_training,
                node_map_x,
                node_map_y,
                sector.sign_x,
                sector.sign_y,
            ))
            validation_pressure = _normalize_columns!(_parity_project(
                pressure_validation,
                node_map_x,
                node_map_y,
                sector.sign_x,
                sector.sign_y,
            ))
            right_snapshots = _solve(
                right_factor,
                _right_hand_side(system, layout, training_pressure),
            )
            right_observations = _boundary_output(system, layout, right_snapshots)
            right_coefficients, right_spectrum = _snapshot_coefficients(
                right_observations,
                rank,
                T,
            )
            right_seed = right_snapshots * right_coefficients
            right_basis = Complex{T}.(Matrix(qr(right_seed).Q[:, 1:rank]))
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
                node_map_x,
                node_map_y,
                sector.sign_x,
                sector.sign_y,
            )))
            d_full = _parity_project(
                d_full,
                face_map_x,
                face_map_y,
                sector.sign_x,
                sector.sign_y,
            )
            # The isolated electrical/source response is stored exactly as a
            # direct affine term. The reduced state is reserved for pressure-
            # induced loading feedback, which is the response family used to
            # train and validate this basis.
            reduced_b = zeros(Complex{T}, rank, size(b_matrix, 2))
            projected_e = _parity_project(
                driven_boundary_output,
                face_map_x,
                face_map_y,
                sector.sign_x,
                sector.sign_y,
            )
            compact_c = zeros(Complex{T}, rank, length(node_orbits))
            for (column, (orbit, orbit_size)) in enumerate(zip(node_orbits, node_orbit_sizes))
                compact_c[:, column] .= orbit_size .* view(c_full, :, orbit[1])
            end
            compact_d = Matrix(d_full[[orbit[1] for orbit in face_orbits], :])
            compact_e = Matrix(projected_e[[orbit[1] for orbit in face_orbits], :])
            velocity_output = isempty(layout.mechanical_range) ?
                              zeros(Complex{T}, 0, rank) :
                              Matrix(view(right_basis, layout.mechanical_range, :))
            current_output = isempty(layout.electrical_range) ?
                             zeros(Complex{T}, 0, rank) :
                             Matrix(view(right_basis, layout.electrical_range, :))
            velocity_drive = sector.name == "even_even" ?
                             driven_velocity_output :
                             zeros(Complex{T}, size(driven_velocity_output))
            current_drive = sector.name == "even_even" ?
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
                sector.sign_x,
                sector.sign_y,
            )
            reduced_state = reduced_k \ (-compact_c * compact_pressure)
            compact_output = compact_d * reduced_state
            candidate_output = _reconstruct_parity_values(
                compact_output,
                face_orbits,
                sector.sign_x,
                sector.sign_y,
                length(system.bem_mesh.faces),
            )
            push!(
                validation,
                Dict(
                    "sector" => sector.name,
                    "boundary_output_error" => _curve_errors(
                        exact_validation_output,
                        candidate_output,
                    ),
                    "petrov_overlap_condition" => overlap_condition,
                    "right_snapshot_tail_ratio" => sqrt(
                        right_spectrum[rank] / first(right_spectrum),
                    ),
                    "petrov_identity_error" => norm(
                        petrov_identity - Matrix{Complex{T}}(I, rank, rank),
                    ) / sqrt(T(rank)),
                ),
            )
            push!(
                sector_models,
                (
                    k=Matrix(reduced_k),
                    c=compact_c,
                    d=compact_d,
                    b=Matrix(reduced_b),
                    e=compact_e,
                    velocity=velocity_output,
                    current=current_output,
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
        "rank_per_sector" => rank,
        "training_count_per_sector" => training_count,
        "validation_count_per_sector" => validation_count,
        "sector_names" => [sector.name for sector in PARITY_SECTORS],
        "sector_signs" => [[sector.sign_x, sector.sign_y] for sector in PARITY_SECTORS],
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
