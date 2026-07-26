module BeatEngineCoupled

using LinearAlgebra, SparseArrays, StaticArrays
using ..BeatEngineCore

export VolumeMesh,
    ConformingInterfaceMap,
    InterfaceOperators,
    load_gmsh41_volume,
    restrict_volume_mesh,
    physical_tag,
    assemble_p1_fem_matrices,
    assemble_boundary_mass_matrix,
    assemble_prescribed_velocity_load,
    sealed_cavity_modes,
    solve_prescribed_velocity_interior,
    build_conforming_interface_map,
    assemble_interface_operators,
    prepare_coupled_cache,
    release_coupled_cache!,
    build_coupled_system,
    solve_coupled_system,
    solve_coupled

struct VolumeMesh{T<:AbstractFloat}
    vertices::Vector{SVector{3,T}}
    tetrahedra::Vector{NTuple{4,Int}}
    tetra_physical_tags::Vector{Int}
    boundary_faces::Vector{NTuple{3,Int}}
    boundary_physical_tags::Vector{Int}
    physical_names::Dict{Tuple{Int,Int},String}
end

struct ConformingInterfaceMap
    fem_vertex_indices::Vector{Int}
    fem_to_bem_vertex_indices::Vector{Int}
    fem_face_indices::Vector{Int}
    bem_face_indices::Vector{Int}
    normal_sign::Vector{Int}
end

function restrict_volume_mesh(mesh::VolumeMesh{T}, selected_tags) where {T<:AbstractFloat}
    tags = Set(Int.(selected_tags))
    isempty(tags) && error("Bounded FEM region must select at least one physical volume group.")
    tetrahedron_indices = findall(tag -> tag in tags, mesh.tetra_physical_tags)
    isempty(tetrahedron_indices) && error(
        "Selected FEM volume groups contain no tetrahedra. Requested tags: $(join(sort(collect(tags)), ", ")).",
    )

    selected_tetrahedra = mesh.tetrahedra[tetrahedron_indices]
    face_occurrences = Dict{NTuple{3,Int},Int}()
    for tetrahedron in selected_tetrahedra
        for face in _tetrahedron_faces(tetrahedron)
            face_occurrences[face] = get(face_occurrences, face, 0) + 1
        end
    end
    exterior_faces = Set(face for (face, count) in face_occurrences if count == 1)
    boundary_face_indices = findall(
        face -> _sorted_face(face) in exterior_faces,
        mesh.boundary_faces,
    )

    active_vertices = sort(unique(vcat([collect(tetrahedron) for tetrahedron in selected_tetrahedra]...)))
    vertex_index_map = Dict(vertex => index for (index, vertex) in enumerate(active_vertices))
    boundary_face_index_map = Dict(
        face_index => index
        for (index, face_index) in enumerate(boundary_face_indices)
    )
    restricted_tetrahedra = [
        ntuple(local_index -> vertex_index_map[tetrahedron[local_index]], 4)
        for tetrahedron in selected_tetrahedra
    ]
    restricted_boundary_faces = [
        ntuple(local_index -> vertex_index_map[face[local_index]], 3)
        for face in mesh.boundary_faces[boundary_face_indices]
    ]
    restricted = VolumeMesh{T}(
        mesh.vertices[active_vertices],
        restricted_tetrahedra,
        mesh.tetra_physical_tags[tetrahedron_indices],
        restricted_boundary_faces,
        mesh.boundary_physical_tags[boundary_face_indices],
        mesh.physical_names,
    )
    return (
        mesh=restricted,
        vertex_index_map=vertex_index_map,
        boundary_face_index_map=boundary_face_index_map,
    )
end

struct InterfaceOperators{T<:AbstractFloat}
    fem_load::SparseMatrixCSC{T,Int}
    bem_flux::SparseMatrixCSC{T,Int}
    fem_trace::SparseMatrixCSC{T,Int}
    bem_trace::SparseMatrixCSC{T,Int}
end

function load_gmsh41_volume(filepath::String, scale::T) where {T<:AbstractFloat}
    lines = readlines(filepath)
    format_start = findfirst(==("\$MeshFormat"), lines)
    isnothing(format_start) && error("Gmsh mesh is missing \$MeshFormat.")
    format = split(strip(lines[format_start + 1]))
    startswith(format[1], "4.1") || error("Volume FEM import currently requires a Gmsh 4.1 ASCII mesh.")
    format[2] == "0" || error("Volume FEM import currently requires an ASCII Gmsh mesh.")

    physical_names = _parse_physical_names(lines)
    entity_physical_tags = _parse_entity_physical_tags(lines)
    vertices, node_index_map = _parse_gmsh41_nodes(lines, scale)
    tetrahedra, tetra_tags, boundary_faces, boundary_tags = _parse_gmsh41_elements(
        lines,
        node_index_map,
        entity_physical_tags,
    )
    isempty(tetrahedra) && error("FEM volume mesh contains no first-order tetrahedra.")
    isempty(boundary_faces) && error("FEM volume mesh contains no first-order boundary triangles.")
    return VolumeMesh{T}(
        vertices,
        tetrahedra,
        tetra_tags,
        boundary_faces,
        boundary_tags,
        physical_names,
    )
end

function physical_tag(mesh::VolumeMesh, dimension::Int, name::AbstractString)
    for ((candidate_dimension, tag), candidate_name) in mesh.physical_names
        candidate_dimension == dimension && candidate_name == name && return tag
    end
    available = sort([
        candidate_name
        for ((candidate_dimension, _), candidate_name) in mesh.physical_names
        if candidate_dimension == dimension
    ])
    error(
        "Physical group $(repr(String(name))) was not found in dimension $dimension. " *
        "Available groups: $(isempty(available) ? "(none)" : join(available, ", ")).",
    )
end

function assemble_p1_fem_matrices(mesh::VolumeMesh{T}) where {T<:AbstractFloat}
    stiffness_rows = Int[]
    stiffness_cols = Int[]
    stiffness_values = T[]
    mass_rows = Int[]
    mass_cols = Int[]
    mass_values = T[]
    reference_gradients = @SMatrix T[
        -1 1 0 0
        -1 0 1 0
        -1 0 0 1
    ]
    reference_mass = @SMatrix T[
        2 1 1 1
        1 2 1 1
        1 1 2 1
        1 1 1 2
    ]

    for tetrahedron in mesh.tetrahedra
        x1 = mesh.vertices[tetrahedron[1]]
        jacobian = SMatrix{3,3,T}(
            hcat(
                mesh.vertices[tetrahedron[2]] - x1,
                mesh.vertices[tetrahedron[3]] - x1,
                mesh.vertices[tetrahedron[4]] - x1,
            ),
        )
        determinant = det(jacobian)
        edge_scale = maximum(abs, jacobian)
        determinant_tolerance = eps(T) * edge_scale^3
        abs(determinant) > determinant_tolerance ||
            error("FEM volume mesh contains a numerically degenerate tetrahedron.")
        volume = abs(determinant) / T(6)
        gradients = inv(transpose(jacobian)) * reference_gradients
        local_stiffness = volume .* (transpose(gradients) * gradients)
        local_mass = (volume / T(20)) .* reference_mass

        for local_row in 1:4, local_col in 1:4
            push!(stiffness_rows, tetrahedron[local_row])
            push!(stiffness_cols, tetrahedron[local_col])
            push!(stiffness_values, local_stiffness[local_row, local_col])
            push!(mass_rows, tetrahedron[local_row])
            push!(mass_cols, tetrahedron[local_col])
            push!(mass_values, local_mass[local_row, local_col])
        end
    end

    dof_count = length(mesh.vertices)
    stiffness = sparse(stiffness_rows, stiffness_cols, stiffness_values, dof_count, dof_count)
    mass = sparse(mass_rows, mass_cols, mass_values, dof_count, dof_count)
    return stiffness, mass
end

function assemble_boundary_mass_matrix(
    mesh::VolumeMesh{T},
    face_indices,
    boundary_vertex_indices::AbstractVector{Int},
) where {T<:AbstractFloat}
    boundary_dof = Dict(vertex => index for (index, vertex) in enumerate(boundary_vertex_indices))
    rows = Int[]
    cols = Int[]
    values = T[]
    reference_mass = @SMatrix T[
        2 1 1
        1 2 1
        1 1 2
    ]

    for face_index in face_indices
        face = mesh.boundary_faces[face_index]
        area = _triangle_area(mesh.vertices, face)
        local_mass = (area / T(12)) .* reference_mass
        for local_row in 1:3, local_col in 1:3
            boundary_col = get(boundary_dof, face[local_col], 0)
            boundary_col == 0 && error("Boundary face references a vertex outside the requested boundary space.")
            push!(rows, face[local_row])
            push!(cols, boundary_col)
            push!(values, local_mass[local_row, local_col])
        end
    end
    return sparse(rows, cols, values, length(mesh.vertices), length(boundary_vertex_indices))
end

function assemble_prescribed_velocity_load(
    mesh::VolumeMesh{T},
    boundary_tag::Int,
    density::T,
    omega::T,
    velocity::Complex{T},
) where {T<:AbstractFloat}
    load = zeros(Complex{T}, length(mesh.vertices))
    normal_derivative = Complex{T}(0, density * omega) * velocity
    for face_index in eachindex(mesh.boundary_faces)
        mesh.boundary_physical_tags[face_index] == boundary_tag || continue
        face = mesh.boundary_faces[face_index]
        nodal_load = normal_derivative * _triangle_area(mesh.vertices, face) / T(3)
        for vertex in face
            load[vertex] += nodal_load
        end
    end
    return load
end

function sealed_cavity_modes(
    mesh::VolumeMesh{T},
    sound_speed::T;
    count::Int=6,
    zero_tolerance::T=T(1e-8),
) where {T<:AbstractFloat}
    count > 0 || error("Mode count must be positive.")
    stiffness, mass = assemble_p1_fem_matrices(mesh)
    decomposition = eigen(Symmetric(Matrix(stiffness)), Symmetric(Matrix(mass)))
    eigenvalues = sort(real.(decomposition.values))
    scale = maximum(abs, eigenvalues; init=one(T))
    positive = filter(value -> value > zero_tolerance * scale, eigenvalues)
    selected = positive[1:min(count, length(positive))]
    return sound_speed .* sqrt.(selected) ./ T(2pi)
end

function solve_prescribed_velocity_interior(
    mesh::VolumeMesh{T},
    frequency_hz::T,
    sound_speed::T,
    density::T,
    boundary_tag::Int;
    velocity::Complex{T}=Complex{T}(1, 0),
) where {T<:AbstractFloat}
    stiffness, mass = assemble_p1_fem_matrices(mesh)
    omega = T(2pi) * frequency_hz
    wavenumber = omega / sound_speed
    system = Complex{T}.(stiffness) - Complex{T}(wavenumber^2) .* Complex{T}.(mass)
    rhs = assemble_prescribed_velocity_load(mesh, boundary_tag, density, omega, velocity)
    pressure = system \ rhs
    relative_residual = norm(system * pressure - rhs) / max(norm(rhs), eps(T))
    return (
        pressure=pressure,
        stiffness=stiffness,
        mass=mass,
        rhs=rhs,
        relative_residual=relative_residual,
    )
end

function build_conforming_interface_map(
    fem_mesh::VolumeMesh{T},
    bem_mesh::BoundaryMesh{T},
    fem_interface_tag::Int,
    bem_interface_tag::Int;
    coordinate_tolerance::T=T(1e-10),
) where {T<:AbstractFloat}
    fem_face_indices = findall(==(fem_interface_tag), fem_mesh.boundary_physical_tags)
    bem_face_indices = findall(==(bem_interface_tag), bem_mesh.physical_tags)
    length(fem_face_indices) == length(bem_face_indices) || error(
        "FEM and BEM interface triangle counts differ: $(length(fem_face_indices)) and $(length(bem_face_indices)).",
    )
    fem_vertices = sort(unique(vcat([collect(fem_mesh.boundary_faces[index]) for index in fem_face_indices]...)))
    bem_vertices = sort(unique(vcat([collect(bem_mesh.faces[index]) for index in bem_face_indices]...)))
    length(fem_vertices) == length(bem_vertices) || error(
        "FEM and BEM interface vertex counts differ: $(length(fem_vertices)) and $(length(bem_vertices)).",
    )

    mapped_bem_vertices = Int[]
    used_bem_vertices = Set{Int}()
    for fem_vertex in fem_vertices
        point = fem_mesh.vertices[fem_vertex]
        distances = [norm(point - bem_mesh.vertices[bem_vertex]) for bem_vertex in bem_vertices]
        local_index = argmin(distances)
        distances[local_index] <= coordinate_tolerance || error(
            "FEM interface vertex $fem_vertex has no BEM match within $coordinate_tolerance m.",
        )
        bem_vertex = bem_vertices[local_index]
        bem_vertex in used_bem_vertices && error("FEM-to-BEM interface vertex mapping is not one-to-one.")
        push!(used_bem_vertices, bem_vertex)
        push!(mapped_bem_vertices, bem_vertex)
    end
    fem_to_bem = Dict(fem_vertices[index] => mapped_bem_vertices[index] for index in eachindex(fem_vertices))
    bem_face_by_key = Dict(
        _sorted_face(bem_mesh.faces[index]) => index
        for index in bem_face_indices
    )
    mapped_bem_faces = Int[]
    normal_sign = Int[]
    for fem_face_index in fem_face_indices
        fem_face = fem_mesh.boundary_faces[fem_face_index]
        mapped_key = _sorted_face((
            fem_to_bem[fem_face[1]],
            fem_to_bem[fem_face[2]],
            fem_to_bem[fem_face[3]],
        ))
        bem_face_index = get(bem_face_by_key, mapped_key, 0)
        bem_face_index > 0 || error("FEM and BEM interface triangle connectivity differs.")
        push!(mapped_bem_faces, bem_face_index)
        fem_normal = _triangle_normal(fem_mesh.vertices, fem_face)
        bem_normal = bem_mesh.normals[bem_face_index]
        push!(normal_sign, dot(fem_normal, bem_normal) >= zero(T) ? 1 : -1)
    end

    return ConformingInterfaceMap(
        fem_vertices,
        mapped_bem_vertices,
        fem_face_indices,
        mapped_bem_faces,
        normal_sign,
    )
end

function assemble_interface_operators(
    fem_mesh::VolumeMesh{T},
    bem_mesh::BoundaryMesh{T},
    interface_map::ConformingInterfaceMap,
) where {T<:AbstractFloat}
    fem_load = assemble_boundary_mass_matrix(
        fem_mesh,
        interface_map.fem_face_indices,
        interface_map.fem_vertex_indices,
    )
    interface_dof = Dict(
        vertex => index
        for (index, vertex) in enumerate(interface_map.fem_vertex_indices)
    )

    bem_flux_rows = Int[]
    bem_flux_cols = Int[]
    bem_flux_values = T[]
    for local_face_index in eachindex(interface_map.fem_face_indices)
        fem_face = fem_mesh.boundary_faces[interface_map.fem_face_indices[local_face_index]]
        bem_face_index = interface_map.bem_face_indices[local_face_index]
        orientation = T(interface_map.normal_sign[local_face_index])
        for vertex in fem_face
            push!(bem_flux_rows, bem_face_index)
            push!(bem_flux_cols, interface_dof[vertex])
            push!(bem_flux_values, orientation / T(3))
        end
    end
    bem_flux = sparse(
        bem_flux_rows,
        bem_flux_cols,
        bem_flux_values,
        length(bem_mesh.faces),
        length(interface_map.fem_vertex_indices),
    )

    interface_count = length(interface_map.fem_vertex_indices)
    fem_trace = sparse(
        collect(1:interface_count),
        interface_map.fem_vertex_indices,
        ones(T, interface_count),
        interface_count,
        length(fem_mesh.vertices),
    )
    bem_trace = sparse(
        collect(1:interface_count),
        interface_map.fem_to_bem_vertex_indices,
        ones(T, interface_count),
        interface_count,
        length(bem_mesh.vertices),
    )
    return InterfaceOperators{T}(fem_load, bem_flux, fem_trace, bem_trace)
end

function prepare_coupled_cache(
    fem_mesh::VolumeMesh{T},
    bem_mesh::BoundaryMesh{T},
    interface_map::ConformingInterfaceMap;
    quadrature_order::Int=2,
    singular_order::Int=2,
    bem_backend::Symbol=:cpu,
) where {T<:AbstractFloat}
    bem_backend in (:cpu, :cuda) ||
        error("Unsupported coupled BEM backend: $bem_backend. Expected :cpu or :cuda.")
    stiffness, mass = assemble_p1_fem_matrices(fem_mesh)
    interface_operators = assemble_interface_operators(fem_mesh, bem_mesh, interface_map)
    p1 = build_p1_space(bem_mesh)
    dp0 = build_dp0_space(bem_mesh)
    rule = triangle_rule(T, quadrature_order)
    singular_cache = build_singular_correction_cache(bem_mesh, singular_order)
    cpu_assembly_cache = bem_backend == :cpu ? build_beat_cpu_assembly_cache(
        bem_mesh,
        p1,
        dp0,
        rule;
        singular_order=singular_order,
    ) : nothing
    device_cache = bem_backend == :cuda ? build_cuda_regular_assembly_cache(bem_mesh, rule) : nothing
    device_singular_cache = bem_backend == :cuda ? BeatEngineCore.build_cuda_singular_correction_cache(
        singular_cache,
        p1,
        dp0,
    ) : nothing
    cpu_field_cache = build_field_evaluation_cache(bem_mesh, rule)
    field_cache = bem_backend == :cuda ? build_cuda_field_evaluation_cache(cpu_field_cache) : cpu_field_cache
    return (
        bem_backend=bem_backend,
        stiffness=stiffness,
        mass=mass,
        interface_operators=interface_operators,
        p1=p1,
        dp0=dp0,
        rule=rule,
        cpu_assembly_cache=cpu_assembly_cache,
        device_cache=device_cache,
        device_singular_cache=device_singular_cache,
        singular_cache=singular_cache,
        identity_p1_p1=assemble_l2_identity_matrix(bem_mesh, p1, dp0, rule, :p1, :p1),
        identity_p1_dp0=assemble_l2_identity_matrix(bem_mesh, p1, dp0, rule, :p1, :dp0),
        field_cache=field_cache,
    )
end

function _unsafe_free_cuda_fields!(value, fields)
    isnothing(value) && return nothing
    cuda = BeatEngineCore.cuda_module()
    for field in fields
        cuda.unsafe_free!(getproperty(value, field))
    end
    return nothing
end

function release_coupled_cache!(cache)
    cache.bem_backend == :cuda || return nothing
    _unsafe_free_cuda_fields!(
        cache.device_cache,
        (
            :face_vertices,
            :normals,
            :areas,
            :faces,
            :curls,
            :rule_points,
            :rule_weights,
            :test_indices,
            :trial_indices,
        ),
    )
    _unsafe_free_cuda_fields!(
        cache.device_singular_cache,
        (
            :test_indices,
            :trial_indices,
            :rule_indices,
            :jac_scales,
            :normal_products,
            :p1_rows,
            :p1_cols,
            :dp0_cols,
            :rule_offsets,
            :rule_test_points,
            :rule_trial_points,
            :rule_weights,
        ),
    )
    _unsafe_free_cuda_fields!(
        cache.field_cache,
        (
            :source_points,
            :source_normals,
            :source_weights,
            :source_faces,
            :source_elements,
            :basis_values,
        ),
    )
    return nothing
end

function _cuda_coupled_bem_blocks(
    operators,
    identity_p1_p1,
    identity_p1_dp0,
    bem_flux,
    wavenumber::T;
    validation_diagnostics::Bool,
) where {T<:AbstractFloat}
    cuda = BeatEngineCore.cuda_module()
    coupling = Complex{T}(0, 1) / wavenumber
    d_identity_p1_p1 = d_identity_p1_dp0 = d_bem_flux = nothing
    d_lhs = d_interface_block = d_interface_temp = d_rhs_operator = nothing
    try
        d_identity_p1_p1 = cuda.CuArray(Complex{T}.(identity_p1_p1))
        d_identity_p1_dp0 = cuda.CuArray(Complex{T}.(identity_p1_dp0))
        d_bem_flux = cuda.CuArray(Complex{T}.(Matrix(bem_flux)))
        d_lhs = (
            Complex{T}(0.5) .* d_identity_p1_p1 .-
            operators.double_layer .+
            coupling .* operators.hypersingular
        )
        d_interface_block = similar(
            operators.single_layer,
            Complex{T},
            size(operators.single_layer, 1),
            size(d_bem_flux, 2),
        )
        d_interface_temp = similar(d_interface_block)
        mul!(d_interface_block, operators.single_layer, d_bem_flux)
        mul!(d_interface_temp, operators.adjoint_double_layer, d_bem_flux)
        d_interface_block .+= coupling .* d_interface_temp
        mul!(d_interface_temp, d_identity_p1_dp0, d_bem_flux)
        d_interface_block .+= Complex{T}(0.5) * coupling .* d_interface_temp
        if validation_diagnostics
            d_rhs_operator = (
                -operators.single_layer .-
                coupling .* (
                    operators.adjoint_double_layer .+
                    Complex{T}(0.5) .* d_identity_p1_dp0
                )
            )
        end
        cuda.synchronize()
        return (
            bem_lhs=Complex{T}.(Array(d_lhs)),
            bem_rhs_operator=isnothing(d_rhs_operator) ? nothing : Complex{T}.(Array(d_rhs_operator)),
            bem_interface_block=Complex{T}.(Array(d_interface_block)),
        )
    finally
        for value in (
            d_identity_p1_p1,
            d_identity_p1_dp0,
            d_bem_flux,
            d_lhs,
            d_interface_block,
            d_interface_temp,
            d_rhs_operator,
        )
            isnothing(value) || cuda.unsafe_free!(value)
        end
    end
end

function build_coupled_system(
    fem_mesh::VolumeMesh{T},
    bem_mesh::BoundaryMesh{T},
    interface_map::ConformingInterfaceMap,
    frequency_hz::T,
    sound_speed::T,
    density::T;
    quadrature_order::Int=2,
    singular_order::Int=2,
    cache=nothing,
    validation_diagnostics::Bool=true,
    bem_backend::Symbol=:cpu,
) where {T<:AbstractFloat}
    fem_stage_started = time_ns()
    prepared = isnothing(cache) ? prepare_coupled_cache(
        fem_mesh,
        bem_mesh,
        interface_map;
        quadrature_order=quadrature_order,
        singular_order=singular_order,
        bem_backend=bem_backend,
    ) : cache
    prepared.bem_backend == bem_backend ||
        error("Coupled cache backend does not match requested BEM backend.")
    omega = T(2pi) * frequency_hz
    wavenumber = omega / sound_speed
    fem_system = Complex{T}.(prepared.stiffness) - Complex{T}(wavenumber^2) .* Complex{T}.(prepared.mass)
    interface_operators = prepared.interface_operators
    fem_system_s = (time_ns() - fem_stage_started) / 1.0e9

    bem_operator_started = time_ns()
    operators = assemble_regular_galerkin_operators(
        bem_mesh,
        prepared.p1,
        prepared.dp0,
        wavenumber,
        prepared.rule;
        skip_singular=false,
        singular_order=singular_order,
        backend=bem_backend,
        singular_cache=prepared.singular_cache,
        cpu_cache=prepared.cpu_assembly_cache,
        device_cache=prepared.device_cache,
        return_device=bem_backend == :cuda,
        accelerator_quadrature=bem_backend == :cuda,
        device_singular_cache=prepared.device_singular_cache,
    )
    bem_operator_s = (time_ns() - bem_operator_started) / 1.0e9
    bem_matrix_started = time_ns()
    bem_blocks = if bem_backend == :cuda
        try
            _cuda_coupled_bem_blocks(
                operators,
                prepared.identity_p1_p1,
                prepared.identity_p1_dp0,
                interface_operators.bem_flux,
                wavenumber;
                validation_diagnostics=validation_diagnostics,
            )
        finally
            release_operator_storage!(operators)
        end
    else
        bem_lhs, bem_rhs_operator = burton_miller_neumann_matrices(
            operators,
            prepared.identity_p1_p1,
            prepared.identity_p1_dp0,
            wavenumber,
        )
        (
            bem_lhs=bem_lhs,
            bem_rhs_operator=bem_rhs_operator,
            bem_interface_block=-(bem_rhs_operator * Complex{T}.(interface_operators.bem_flux)),
        )
    end
    bem_lhs = bem_blocks.bem_lhs
    bem_rhs_operator = bem_blocks.bem_rhs_operator
    bem_matrix_s = (time_ns() - bem_matrix_started) / 1.0e9

    block_assembly_started = time_ns()
    fem_count = length(fem_mesh.vertices)
    bem_count = length(bem_mesh.vertices)
    interface_count = length(interface_map.fem_vertex_indices)
    fem_range = 1:fem_count
    bem_range = (fem_count + 1):(fem_count + bem_count)
    flux_range = (fem_count + bem_count + 1):(fem_count + bem_count + interface_count)
    coupled = zeros(Complex{T}, fem_count + bem_count + interface_count, fem_count + bem_count + interface_count)
    coupled[fem_range, fem_range] = Matrix(fem_system)
    coupled[fem_range, flux_range] = -Complex{T}.(Matrix(interface_operators.fem_load))
    coupled[bem_range, bem_range] = bem_lhs
    coupled[bem_range, flux_range] = bem_blocks.bem_interface_block
    coupled[flux_range, fem_range] = Complex{T}.(Matrix(interface_operators.fem_trace))
    coupled[flux_range, bem_range] = -Complex{T}.(Matrix(interface_operators.bem_trace))
    block_assembly_s = (time_ns() - block_assembly_started) / 1.0e9

    coupled_factorization_started = time_ns()
    factorization = validation_diagnostics ? lu(coupled) : lu!(coupled)
    coupled_factorization_s = (time_ns() - coupled_factorization_started) / 1.0e9
    replay_factorization_started = time_ns()
    bem_factorization = validation_diagnostics ? lu(bem_lhs) : nothing
    replay_factorization_s = (time_ns() - replay_factorization_started) / 1.0e9
    return (
        fem_mesh=fem_mesh,
        bem_mesh=bem_mesh,
        interface_map=interface_map,
        interface_operators=interface_operators,
        density=density,
        omega=omega,
        wavenumber=wavenumber,
        field_cache=prepared.field_cache,
        coupled=validation_diagnostics ? coupled : nothing,
        factorization=factorization,
        fem_range=fem_range,
        bem_range=bem_range,
        flux_range=flux_range,
        bem_lhs=validation_diagnostics ? bem_lhs : nothing,
        bem_factorization=bem_factorization,
        bem_rhs_operator=validation_diagnostics ? bem_rhs_operator : nothing,
        bem_backend=bem_backend,
        cache=prepared,
        owns_cache=isnothing(cache),
        validation_diagnostics=validation_diagnostics,
        timings=(
            fem_system_s=fem_system_s,
            bem_operator_s=bem_operator_s,
            bem_matrix_s=bem_matrix_s,
            block_assembly_s=block_assembly_s,
            coupled_factorization_s=coupled_factorization_s,
            replay_factorization_s=replay_factorization_s,
        ),
    )
end

function solve_coupled_system(
    system,
    radiator_tag::Int;
    radiator_velocity=ComplexF64(1, 0),
)
    T = typeof(real(zero(eltype(system.factorization))))
    velocity = Complex{T}(radiator_velocity)
    fem_rhs = assemble_prescribed_velocity_load(
        system.fem_mesh,
        radiator_tag,
        system.density,
        system.omega,
        velocity,
    )
    rhs = zeros(Complex{T}, size(system.factorization, 1))
    rhs[system.fem_range] = fem_rhs
    solution = system.factorization \ rhs
    fem_pressure = solution[system.fem_range]
    bem_pressure = solution[system.bem_range]
    interface_flux = solution[system.flux_range]
    bem_neumann = Complex{T}.(system.interface_operators.bem_flux) * interface_flux

    pressure_jump = (
        system.interface_operators.fem_trace * fem_pressure -
        system.interface_operators.bem_trace * bem_pressure
    )
    pressure_scale = max(
        norm(system.interface_operators.fem_trace * fem_pressure),
        norm(system.interface_operators.bem_trace * bem_pressure),
        eps(T),
    )
    fem_integrated_flux = zero(Complex{T})
    bem_integrated_flux_along_fem_normal = zero(Complex{T})
    interface_dof = Dict(
        vertex => index
        for (index, vertex) in enumerate(system.interface_map.fem_vertex_indices)
    )
    for local_face_index in eachindex(system.interface_map.fem_face_indices)
        fem_face = system.fem_mesh.boundary_faces[system.interface_map.fem_face_indices[local_face_index]]
        fem_flux_average = sum(interface_flux[interface_dof[vertex]] for vertex in fem_face) / T(3)
        bem_face_index = system.interface_map.bem_face_indices[local_face_index]
        fem_integrated_flux += _triangle_area(system.fem_mesh.vertices, fem_face) * fem_flux_average
        bem_integrated_flux_along_fem_normal += (
            T(system.interface_map.normal_sign[local_face_index]) *
            system.bem_mesh.areas[bem_face_index] *
            bem_neumann[bem_face_index]
        )
    end
    flux_scale = max(abs(fem_integrated_flux), abs(bem_integrated_flux_along_fem_normal), eps(T))
    replay_error = nothing
    relative_residual = nothing
    if system.validation_diagnostics
        replay_pressure = system.bem_factorization \ (system.bem_rhs_operator * bem_neumann)
        replay_scale = max(norm(bem_pressure), norm(replay_pressure), eps(T))
        replay_error = norm(bem_pressure - replay_pressure) / replay_scale
        relative_residual = norm(system.coupled * solution - rhs) / max(norm(rhs), eps(T))
    end
    return (
        fem_pressure=fem_pressure,
        bem_pressure=bem_pressure,
        interface_flux=interface_flux,
        bem_neumann=bem_neumann,
        relative_residual=relative_residual,
        pressure_continuity_error=norm(pressure_jump) / pressure_scale,
        flux_conservation_error=abs(fem_integrated_flux - bem_integrated_flux_along_fem_normal) / flux_scale,
        all_bem_replay_error=replay_error,
        interface_map=system.interface_map,
        interface_operators=system.interface_operators,
    )
end

function solve_coupled(
    fem_mesh::VolumeMesh{T},
    bem_mesh::BoundaryMesh{T},
    interface_map::ConformingInterfaceMap,
    frequency_hz::T,
    sound_speed::T,
    density::T,
    radiator_tag::Int;
    radiator_velocity::Complex{T}=Complex{T}(1, 0),
    quadrature_order::Int=2,
    singular_order::Int=2,
) where {T<:AbstractFloat}
    system = build_coupled_system(
        fem_mesh,
        bem_mesh,
        interface_map,
        frequency_hz,
        sound_speed,
        density;
        quadrature_order=quadrature_order,
        singular_order=singular_order,
    )
    return solve_coupled_system(
        system,
        radiator_tag;
        radiator_velocity=radiator_velocity,
    )
end

function _parse_physical_names(lines)
    start_index = findfirst(==("\$PhysicalNames"), lines)
    isnothing(start_index) && return Dict{Tuple{Int,Int},String}()
    count = parse(Int, strip(lines[start_index + 1]))
    names = Dict{Tuple{Int,Int},String}()
    for line_index in (start_index + 2):(start_index + 1 + count)
        parts = split(strip(lines[line_index]); limit=3)
        names[(parse(Int, parts[1]), parse(Int, parts[2]))] = strip(parts[3], ['"'])
    end
    return names
end

function _parse_entity_physical_tags(lines)
    start_index = findfirst(==("\$Entities"), lines)
    isnothing(start_index) && error("Gmsh 4.1 volume mesh is missing \$Entities.")
    counts = parse.(Int, split(strip(lines[start_index + 1])))
    length(counts) == 4 || error("Invalid Gmsh \$Entities header.")
    entity_tags = Dict{Tuple{Int,Int},Int}()
    line_index = start_index + 2
    for dimension in 0:3
        for _ in 1:counts[dimension + 1]
            parts = split(strip(lines[line_index]))
            physical_count_index = dimension == 0 ? 5 : 8
            physical_count = parse(Int, parts[physical_count_index])
            if physical_count > 0
                entity_tags[(dimension, parse(Int, parts[1]))] = parse(Int, parts[physical_count_index + 1])
            end
            line_index += 1
        end
    end
    return entity_tags
end

function _parse_gmsh41_nodes(lines, scale::T) where {T<:AbstractFloat}
    start_index = findfirst(==("\$Nodes"), lines)
    end_index = findfirst(==("\$EndNodes"), lines)
    (isnothing(start_index) || isnothing(end_index)) && error("Gmsh volume mesh is missing \$Nodes.")
    tokens = split(join(lines[(start_index + 1):(end_index - 1)], " "))
    cursor = 1
    block_count = parse(Int, tokens[cursor])
    node_count = parse(Int, tokens[cursor + 1])
    cursor += 4
    coordinates_by_tag = Dict{Int,SVector{3,T}}()
    for _ in 1:block_count
        entity_dimension = parse(Int, tokens[cursor])
        _entity_tag = parse(Int, tokens[cursor + 1])
        parametric = parse(Int, tokens[cursor + 2])
        block_node_count = parse(Int, tokens[cursor + 3])
        cursor += 4
        node_tags = parse.(Int, tokens[cursor:(cursor + block_node_count - 1)])
        cursor += block_node_count
        for node_tag in node_tags
            coordinates_by_tag[node_tag] = SVector{3,T}(
                parse(T, tokens[cursor]) * scale,
                parse(T, tokens[cursor + 1]) * scale,
                parse(T, tokens[cursor + 2]) * scale,
            )
            cursor += 3
            parametric == 1 && (cursor += entity_dimension)
        end
    end
    length(coordinates_by_tag) == node_count || error("Gmsh node count does not match parsed coordinates.")
    sorted_tags = sort(collect(keys(coordinates_by_tag)))
    vertices = [coordinates_by_tag[tag] for tag in sorted_tags]
    node_index_map = Dict(tag => index for (index, tag) in enumerate(sorted_tags))
    return vertices, node_index_map
end

function _parse_gmsh41_elements(lines, node_index_map, entity_physical_tags)
    start_index = findfirst(==("\$Elements"), lines)
    end_index = findfirst(==("\$EndElements"), lines)
    (isnothing(start_index) || isnothing(end_index)) && error("Gmsh volume mesh is missing \$Elements.")
    tokens = split(join(lines[(start_index + 1):(end_index - 1)], " "))
    cursor = 1
    block_count = parse(Int, tokens[cursor])
    _element_count = parse(Int, tokens[cursor + 1])
    cursor += 4
    node_counts = Dict(1 => 2, 2 => 3, 4 => 4, 15 => 1)
    tetrahedra = NTuple{4,Int}[]
    tetra_tags = Int[]
    boundary_faces = NTuple{3,Int}[]
    boundary_tags = Int[]
    for _ in 1:block_count
        entity_dimension = parse(Int, tokens[cursor])
        entity_tag = parse(Int, tokens[cursor + 1])
        element_type = parse(Int, tokens[cursor + 2])
        block_element_count = parse(Int, tokens[cursor + 3])
        cursor += 4
        nodes_per_element = get(node_counts, element_type, 0)
        nodes_per_element > 0 || error("Unsupported Gmsh element type $element_type in FEM mesh.")
        physical = get(entity_physical_tags, (entity_dimension, entity_tag), 0)
        for _ in 1:block_element_count
            cursor += 1 # Element tag.
            nodes = ntuple(index -> node_index_map[parse(Int, tokens[cursor + index - 1])], nodes_per_element)
            cursor += nodes_per_element
            if element_type == 2
                push!(boundary_faces, nodes)
                push!(boundary_tags, physical)
            elseif element_type == 4
                push!(tetrahedra, nodes)
                push!(tetra_tags, physical)
            end
        end
    end
    return tetrahedra, tetra_tags, boundary_faces, boundary_tags
end

function _triangle_area(vertices, face)
    return norm(cross(vertices[face[2]] - vertices[face[1]], vertices[face[3]] - vertices[face[1]])) / 2
end

function _triangle_normal(vertices, face)
    normal = cross(vertices[face[2]] - vertices[face[1]], vertices[face[3]] - vertices[face[1]])
    magnitude = norm(normal)
    magnitude > 0 || error("Interface contains a degenerate triangle.")
    return normal / magnitude
end

function _sorted_face(face::NTuple{3,Int})
    values = sort(collect(face))
    return (values[1], values[2], values[3])
end

function _tetrahedron_faces(tetrahedron::NTuple{4,Int})
    return (
        _sorted_face((tetrahedron[1], tetrahedron[2], tetrahedron[3])),
        _sorted_face((tetrahedron[1], tetrahedron[2], tetrahedron[4])),
        _sorted_face((tetrahedron[1], tetrahedron[3], tetrahedron[4])),
        _sorted_face((tetrahedron[2], tetrahedron[3], tetrahedron[4])),
    )
end

end
