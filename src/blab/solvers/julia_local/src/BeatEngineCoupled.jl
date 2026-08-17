module BeatEngineCoupled

using LinearAlgebra, SparseArrays, StaticArrays
using ..BeatEngineCore

const CUDSS_MODULE = try
    @eval import CUDSS
    CUDSS
catch
    nothing
end

function _cudss_module()
    CUDSS_MODULE === nothing && error(
        "CUDA FEM static condensation requires CUDSS.jl in the active Julia environment.",
    )
    return CUDSS_MODULE
end

export VolumeMesh,
    ConformingInterfaceMap,
    InterfaceOperators,
    ElectrodynamicTransducer,
    load_gmsh41_volume,
    restrict_volume_mesh,
    physical_tag,
    assemble_p1_fem_matrices,
    assemble_fem_dynamic_stiffness,
    assemble_boundary_mass_matrix,
    miki_rigid_backed_surface_admittance,
    assemble_prescribed_velocity_load,
    sealed_cavity_modes,
    solve_prescribed_velocity_interior,
    build_conforming_interface_map,
    offset_interface_map,
    assemble_interface_operators,
    assemble_transducer_operators,
    prepare_coupled_cache,
    release_coupled_cache!,
    build_coupled_system,
    release_coupled_system!,
    solve_coupled_excitations,
    solve_coupled_system,
    solve_coupled_systems,
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

function offset_interface_map(
    map::ConformingInterfaceMap;
    fem_vertex_offset::Int=0,
    fem_face_offset::Int=0,
    bem_vertex_offset::Int=0,
    bem_face_offset::Int=0,
)
    return ConformingInterfaceMap(
        map.fem_vertex_indices .+ fem_vertex_offset,
        map.fem_to_bem_vertex_indices .+ bem_vertex_offset,
        map.fem_face_indices .+ fem_face_offset,
        map.bem_face_indices .+ bem_face_offset,
        map.normal_sign,
    )
end

struct ElectrodynamicTransducer{T<:AbstractFloat}
    id::String
    fem_boundary_tags::Vector{Int}
    fem_motion_signs::Vector{T}
    bem_boundary_tags::Vector{Int}
    bem_motion_signs::Vector{T}
    motion_axis::SVector{3,T}
    surface_completion_factor::T
    physical_driver_orbit_count::Int
    re_ohm::T
    le_h::T
    bl_n_per_a::T
    mmd_kg::T
    cms_m_per_n::T
    rms_n_s_per_m::T
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

function assemble_fem_dynamic_stiffness(
    stiffness::SparseMatrixCSC{T,Ti},
    mass::SparseMatrixCSC{T,Ti},
    wavenumber::T;
    bulk_loss_factor::T=zero(T),
    bulk_loss_mass=nothing,
) where {T<:AbstractFloat,Ti<:Integer}
    size(stiffness) == size(mass) || error("FEM stiffness and mass matrices must have matching shapes.")
    isfinite(wavenumber) && wavenumber >= zero(T) || error("FEM wavenumber must be finite and nonnegative.")
    isfinite(bulk_loss_factor) && bulk_loss_factor >= zero(T) || error(
        "FEM bulk loss factor must be finite and nonnegative.",
    )
    isnothing(bulk_loss_mass) || size(bulk_loss_mass) == size(mass) || error(
        "FEM bulk-loss mass matrix must match the ordinary mass matrix.",
    )
    squared_wavenumber = wavenumber^2
    system = Complex{T}.(stiffness) - squared_wavenumber .* mass
    if isnothing(bulk_loss_mass)
        return system - Complex{T}(0, bulk_loss_factor * squared_wavenumber) .* mass
    end
    return system - Complex{T}(0, squared_wavenumber) .* bulk_loss_mass
end

function miki_rigid_backed_surface_admittance(
    frequency_hz::T,
    sound_speed::T,
    density::T,
    thickness_m::T,
    flow_resistivity_pa_s_per_m2::T,
) where {T<:AbstractFloat}
    frequency_hz > zero(T) || error("Miki wall impedance requires a positive frequency.")
    sound_speed > zero(T) && density > zero(T) || error("Miki wall impedance requires a positive medium.")
    thickness_m > zero(T) || error("Wall-lining thickness must be positive.")
    flow_resistivity_pa_s_per_m2 > zero(T) || error("Wall-lining airflow resistivity must be positive.")
    normalized_frequency = frequency_hz / flow_resistivity_pa_s_per_m2
    impedance_power = normalized_frequency^T(-0.632)
    wavenumber_power = normalized_frequency^T(-0.618)
    characteristic_impedance_positive = density * sound_speed * Complex{T}(
        one(T) + T(0.070) * impedance_power,
        -T(0.107) * impedance_power,
    )
    free_wavenumber = T(2pi) * frequency_hz / sound_speed
    porous_wavenumber_positive = free_wavenumber * Complex{T}(
        one(T) + T(0.109) * wavenumber_power,
        -T(0.160) * wavenumber_power,
    )
    surface_impedance_positive = -Complex{T}(0, 1) * characteristic_impedance_positive *
                                 cot(porous_wavenumber_positive * thickness_m)
    # Miki is conventionally written for exp(+i*omega*t); Boundary Lab uses exp(-i*omega*t).
    surface_impedance = conj(surface_impedance_positive)
    admittance = inv(surface_impedance)
    real(admittance) >= zero(T) || error("Miki wall impedance produced a non-passive admittance.")
    return admittance
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
    bulk_loss_factor::T=zero(T),
) where {T<:AbstractFloat}
    stiffness, mass = assemble_p1_fem_matrices(mesh)
    omega = T(2pi) * frequency_hz
    wavenumber = omega / sound_speed
    system = assemble_fem_dynamic_stiffness(
        stiffness,
        mass,
        wavenumber;
        bulk_loss_factor=bulk_loss_factor,
    )
    rhs = assemble_prescribed_velocity_load(mesh, boundary_tag, density, omega, velocity)
    pressure = system \ rhs
    relative_residual = norm(system * pressure - rhs) / max(norm(rhs), eps(T))
    return (
        pressure=pressure,
        stiffness=stiffness,
        mass=mass,
        rhs=rhs,
        bulk_loss_factor=bulk_loss_factor,
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

function assemble_transducer_operators(
    fem_mesh::VolumeMesh{T},
    bem_mesh::BoundaryMesh{T},
    transducers::AbstractVector{ElectrodynamicTransducer{T}},
) where {T<:AbstractFloat}
    transducer_count = length(transducers)
    transducer_count == 0 && return (
        fem_surface=spzeros(T, length(fem_mesh.vertices), 0),
        fem_force=spzeros(T, length(fem_mesh.vertices), 0),
        bem_surface=spzeros(T, length(bem_mesh.vertices), 0),
        bem_force=spzeros(T, length(bem_mesh.vertices), 0),
        bem_normal_velocity=spzeros(T, length(bem_mesh.faces), 0),
    )

    fem_rows = Int[]
    fem_cols = Int[]
    fem_values = T[]
    fem_force_values = T[]
    bem_rows = Int[]
    bem_cols = Int[]
    bem_values = T[]
    bem_force_values = T[]
    bem_velocity_rows = Int[]
    bem_velocity_cols = Int[]
    bem_velocity_values = T[]

    fem_outward_normals = _outward_boundary_normals(fem_mesh)
    for (column, transducer) in enumerate(transducers)
        length(transducer.fem_boundary_tags) == length(transducer.fem_motion_signs) ||
            error("Transducer $(repr(transducer.id)) FEM boundary tags and signs differ in length.")
        length(transducer.bem_boundary_tags) == length(transducer.bem_motion_signs) ||
            error("Transducer $(repr(transducer.id)) BEM boundary tags and signs differ in length.")
        isempty(transducer.fem_boundary_tags) && isempty(transducer.bem_boundary_tags) &&
            error("Transducer $(repr(transducer.id)) has no acoustic moving boundary.")
        axis_norm = norm(transducer.motion_axis)
        axis_norm > zero(T) || error("Transducer $(repr(transducer.id)) has a zero motion axis.")
        motion_axis = transducer.motion_axis / axis_norm
        transducer.surface_completion_factor >= one(T) ||
            error("Transducer $(repr(transducer.id)) has an invalid surface completion factor.")
        transducer.physical_driver_orbit_count >= 1 ||
            error("Transducer $(repr(transducer.id)) has an invalid physical driver orbit count.")

        fem_sign_by_tag = Dict(zip(transducer.fem_boundary_tags, transducer.fem_motion_signs))
        length(fem_sign_by_tag) == length(transducer.fem_boundary_tags) ||
            error("Transducer $(repr(transducer.id)) repeats a FEM boundary tag.")
        matched_fem_tags = Set{Int}()
        for face_index in eachindex(fem_mesh.boundary_faces)
            tag = fem_mesh.boundary_physical_tags[face_index]
            sign = get(fem_sign_by_tag, tag, zero(T))
            sign == zero(T) && continue
            push!(matched_fem_tags, tag)
            face = fem_mesh.boundary_faces[face_index]
            projection = sign * dot(fem_outward_normals[face_index], motion_axis)
            nodal_area = projection * _triangle_area(fem_mesh.vertices, face) / T(3)
            for vertex in face
                push!(fem_rows, vertex)
                push!(fem_cols, column)
                push!(fem_values, nodal_area)
                push!(
                    fem_force_values,
                    transducer.surface_completion_factor * nodal_area,
                )
            end
        end
        missing_fem_tags = setdiff(Set(keys(fem_sign_by_tag)), matched_fem_tags)
        isempty(missing_fem_tags) || error(
            "Transducer $(repr(transducer.id)) FEM boundary tags contain no selected faces: " *
            join(sort(collect(missing_fem_tags)), ", "),
        )

        bem_sign_by_tag = Dict(zip(transducer.bem_boundary_tags, transducer.bem_motion_signs))
        length(bem_sign_by_tag) == length(transducer.bem_boundary_tags) ||
            error("Transducer $(repr(transducer.id)) repeats a BEM boundary tag.")
        matched_bem_tags = Set{Int}()
        for face_index in eachindex(bem_mesh.faces)
            tag = bem_mesh.physical_tags[face_index]
            sign = get(bem_sign_by_tag, tag, zero(T))
            sign == zero(T) && continue
            push!(matched_bem_tags, tag)
            face = bem_mesh.faces[face_index]
            projection = sign * dot(bem_mesh.normals[face_index], motion_axis)
            nodal_area = projection * bem_mesh.areas[face_index] / T(3)
            for vertex in face
                push!(bem_rows, vertex)
                push!(bem_cols, column)
                push!(bem_values, nodal_area)
                push!(
                    bem_force_values,
                    transducer.surface_completion_factor * nodal_area,
                )
            end
            push!(bem_velocity_rows, face_index)
            push!(bem_velocity_cols, column)
            push!(bem_velocity_values, projection)
        end
        missing_bem_tags = setdiff(Set(keys(bem_sign_by_tag)), matched_bem_tags)
        isempty(missing_bem_tags) || error(
            "Transducer $(repr(transducer.id)) BEM boundary tags contain no faces: " *
            join(sort(collect(missing_bem_tags)), ", "),
        )
    end

    return (
        fem_surface=sparse(
            fem_rows,
            fem_cols,
            fem_values,
            length(fem_mesh.vertices),
            transducer_count,
        ),
        fem_force=sparse(
            fem_rows,
            fem_cols,
            fem_force_values,
            length(fem_mesh.vertices),
            transducer_count,
        ),
        bem_surface=sparse(
            bem_rows,
            bem_cols,
            bem_values,
            length(bem_mesh.vertices),
            transducer_count,
        ),
        bem_force=sparse(
            bem_rows,
            bem_cols,
            bem_force_values,
            length(bem_mesh.vertices),
            transducer_count,
        ),
        bem_normal_velocity=sparse(
            bem_velocity_rows,
            bem_velocity_cols,
            bem_velocity_values,
            length(bem_mesh.faces),
            transducer_count,
        ),
    )
end

function prepare_coupled_cache(
    fem_mesh::VolumeMesh{T},
    bem_mesh::BoundaryMesh{T},
    interface_map::ConformingInterfaceMap;
    quadrature_order::Int=2,
    singular_order::Int=2,
    bem_backend::Symbol=:cpu,
    symmetry_mode::Symbol=:off,
    retained_fem_vertices=interface_map.fem_vertex_indices,
    bulk_loss_factor_by_vertex=zeros(T, length(fem_mesh.vertices)),
    wall_impedances=NamedTuple[],
) where {T<:AbstractFloat}
    bem_backend in (:cpu, :cuda, :rocm) ||
        error("Unsupported coupled BEM backend: $bem_backend. Expected :cpu, :cuda, or :rocm.")
    normalized_symmetry = BeatEngineCore.normalized_symmetry_mode(symmetry_mode)
    validate_symmetry_fundamental_domain!(bem_mesh, normalized_symmetry)

    fem_matrix_started = time_ns()
    stiffness, mass = assemble_p1_fem_matrices(fem_mesh)
    length(bulk_loss_factor_by_vertex) == length(fem_mesh.vertices) || error(
        "Per-region FEM bulk-loss vector must contain one value per FEM vertex.",
    )
    all(value -> isfinite(value) && zero(T) <= value <= one(T), bulk_loss_factor_by_vertex) || error(
        "Per-region FEM bulk-loss factors must be finite and between 0 and 1.",
    )
    bulk_loss_mass = spdiagm(0 => T.(bulk_loss_factor_by_vertex)) * mass
    wall_impedance_operators = [
        begin
            face_indices = findall(==(Int(spec.tag)), fem_mesh.boundary_physical_tags)
            isempty(face_indices) && error("Wall-impedance boundary tag $(spec.tag) has no FEM faces.")
            (
                matrix=assemble_boundary_mass_matrix(
                    fem_mesh,
                    face_indices,
                    collect(1:length(fem_mesh.vertices)),
                ),
                thickness_m=T(spec.thickness_m),
                flow_resistivity_pa_s_per_m2=T(spec.flow_resistivity_pa_s_per_m2),
            )
        end
        for spec in wall_impedances
    ]
    fem_matrix_cache_s = (time_ns() - fem_matrix_started) / 1.0e9

    interface_started = time_ns()
    interface_operators = assemble_interface_operators(fem_mesh, bem_mesh, interface_map)
    interface_operator_cache_s = (time_ns() - interface_started) / 1.0e9

    bem_space_started = time_ns()
    p1 = build_p1_space(bem_mesh)
    dp0 = build_dp0_space(bem_mesh)
    rule = triangle_rule(T, quadrature_order)
    bem_space_cache_s = (time_ns() - bem_space_started) / 1.0e9

    singular_started = time_ns()
    singular_cache = build_singular_correction_cache(bem_mesh, singular_order)
    bem_singular_cache_s = (time_ns() - singular_started) / 1.0e9

    cpu_assembly_started = time_ns()
    cpu_assembly_cache = bem_backend == :cpu ? build_beat_cpu_assembly_cache(
        bem_mesh,
        p1,
        dp0,
        rule;
        singular_order=singular_order,
        symmetry_mode=normalized_symmetry,
    ) : nothing
    bem_cpu_assembly_cache_s = (time_ns() - cpu_assembly_started) / 1.0e9

    device_regular_started = time_ns()
    device_cache = if bem_backend == :cuda
        build_cuda_regular_assembly_cache(bem_mesh, rule)
    elseif bem_backend == :rocm
        build_rocm_regular_assembly_cache(
            bem_mesh,
            p1,
            dp0,
            rule;
            singular_order=singular_order,
            symmetry_mode=normalized_symmetry,
        )
    else
        nothing
    end
    bem_backend == :cuda && BeatEngineCore.cuda_module().synchronize()
    bem_backend == :rocm && BeatEngineCore.amdgpu_module().synchronize()
    bem_device_regular_cache_s = (time_ns() - device_regular_started) / 1.0e9

    device_singular_started = time_ns()
    device_singular_cache = if bem_backend == :cuda
        BeatEngineCore.build_cuda_singular_correction_cache(singular_cache, p1, dp0)
    elseif bem_backend == :rocm
        build_rocm_singular_correction_cache(singular_cache)
    else
        nothing
    end
    bem_backend == :cuda && BeatEngineCore.cuda_module().synchronize()
    bem_backend == :rocm && BeatEngineCore.amdgpu_module().synchronize()
    bem_device_singular_cache_s = (time_ns() - device_singular_started) / 1.0e9

    device_image_started = time_ns()
    device_image_singular_cache = bem_backend == :cuda && normalized_symmetry != :off ?
        BeatEngineCore.build_cuda_image_singular_correction_cache(
            bem_mesh,
            p1,
            dp0,
            singular_order,
            eachindex(bem_mesh.faces),
            normalized_symmetry,
        ) : nothing
    bem_backend == :cuda && BeatEngineCore.cuda_module().synchronize()
    bem_device_image_cache_s = (time_ns() - device_image_started) / 1.0e9

    identity_started = time_ns()
    identity_p1_p1 = assemble_l2_identity_matrix(
        bem_mesh,
        p1,
        dp0,
        rule,
        :p1,
        :p1;
        symmetry_mode=normalized_symmetry,
    )
    identity_p1_dp0 = assemble_l2_identity_matrix(
        bem_mesh,
        p1,
        dp0,
        rule,
        :p1,
        :dp0;
        symmetry_mode=normalized_symmetry,
    )
    bem_identity_cache_s = (time_ns() - identity_started) / 1.0e9

    device_block_started = time_ns()
    device_identity_cache = nothing
    device_bem_flux = nothing
    device_sparse_blocks = nothing
    if bem_backend in (:cuda, :rocm)
        accelerator = bem_backend == :cuda ?
                      BeatEngineCore.cuda_module() : BeatEngineCore.amdgpu_module()
        device_identity_cache = if bem_backend == :cuda
            build_cuda_burton_miller_identity_cache(
                identity_p1_p1,
                identity_p1_dp0,
                T,
            )
        else
            build_rocm_burton_miller_identity_cache(
                identity_p1_p1,
                identity_p1_dp0,
                T,
            )
        end
        device_bem_flux = if bem_backend == :cuda
            accelerator.CuArray(Complex{T}.(Matrix(interface_operators.bem_flux)))
        else
            accelerator.ROCArray(Complex{T}.(Matrix(interface_operators.bem_flux)))
        end
        build_sparse_cache = bem_backend == :cuda ?
                             build_cuda_sparse_scatter_cache :
                             build_rocm_sparse_scatter_cache
        gamma = Int.(collect(retained_fem_vertices))
        device_sparse_blocks = (
            stiffness=build_sparse_cache(stiffness),
            mass=build_sparse_cache(mass),
            bulk_loss_mass=build_sparse_cache(bulk_loss_mass),
            wall_impedance=[
                build_sparse_cache(operator.matrix)
                for operator in wall_impedance_operators
            ],
            fem_load=build_sparse_cache(interface_operators.fem_load),
            fem_trace=build_sparse_cache(interface_operators.fem_trace),
            bem_trace=build_sparse_cache(interface_operators.bem_trace),
            condensed_fem_load=build_sparse_cache(
                interface_operators.fem_load[gamma, :],
            ),
            condensed_fem_trace=build_sparse_cache(
                interface_operators.fem_trace[:, gamma],
            ),
        )
        accelerator.synchronize()
    end
    device_block_cache_s = (time_ns() - device_block_started) / 1.0e9

    field_started = time_ns()
    cpu_field_cache = build_field_evaluation_cache(
        bem_mesh,
        rule;
        symmetry_mode=normalized_symmetry,
    )
    field_cache = if bem_backend == :cuda
        build_cuda_field_evaluation_cache(cpu_field_cache)
    elseif bem_backend == :rocm
        build_rocm_field_evaluation_cache(cpu_field_cache)
    else
        cpu_field_cache
    end
    bem_backend == :cuda && BeatEngineCore.cuda_module().synchronize()
    bem_backend == :rocm && BeatEngineCore.amdgpu_module().synchronize()
    field_cache_s = (time_ns() - field_started) / 1.0e9

    return (
        bem_backend=bem_backend,
        symmetry_mode=normalized_symmetry,
        retained_fem_vertices=Int.(collect(retained_fem_vertices)),
        stiffness=stiffness,
        mass=mass,
        bulk_loss_mass=bulk_loss_mass,
        bulk_loss_factor_by_vertex=T.(collect(bulk_loss_factor_by_vertex)),
        wall_impedance_operators=wall_impedance_operators,
        interface_operators=interface_operators,
        p1=p1,
        dp0=dp0,
        rule=rule,
        cpu_assembly_cache=cpu_assembly_cache,
        device_cache=device_cache,
        device_singular_cache=device_singular_cache,
        device_image_singular_cache=device_image_singular_cache,
        singular_cache=singular_cache,
        identity_p1_p1=identity_p1_p1,
        identity_p1_dp0=identity_p1_dp0,
        device_identity_cache=device_identity_cache,
        device_bem_flux=device_bem_flux,
        device_sparse_blocks=device_sparse_blocks,
        field_cache=field_cache,
        timings=(
            fem_matrix_cache_s=fem_matrix_cache_s,
            interface_operator_cache_s=interface_operator_cache_s,
            bem_space_cache_s=bem_space_cache_s,
            bem_singular_cache_s=bem_singular_cache_s,
            bem_cpu_assembly_cache_s=bem_cpu_assembly_cache_s,
            bem_device_regular_cache_s=bem_device_regular_cache_s,
            bem_device_singular_cache_s=bem_device_singular_cache_s,
            bem_device_image_cache_s=bem_device_image_cache_s,
            bem_identity_cache_s=bem_identity_cache_s,
            device_block_cache_s=device_block_cache_s,
            field_cache_s=field_cache_s,
        ),
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
    if cache.bem_backend == :rocm
        release_rocm_regular_assembly_cache!(cache.device_cache)
        release_rocm_singular_correction_cache!(cache.device_singular_cache)
        release_rocm_field_evaluation_cache!(cache.field_cache)
        release_rocm_burton_miller_identity_cache!(cache.device_identity_cache)
        BeatEngineCore.amdgpu_module().unsafe_free!(cache.device_bem_flux)
        for (name, sparse_cache) in pairs(cache.device_sparse_blocks)
            if name == :wall_impedance
                for wall_cache in sparse_cache
                    release_rocm_sparse_scatter_cache!(wall_cache)
                end
            elseif !isnothing(sparse_cache)
                release_rocm_sparse_scatter_cache!(sparse_cache)
            end
        end
        return nothing
    end
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
        cache.device_image_singular_cache,
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
            :transform_signs,
            :curl_signs,
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
    release_cuda_burton_miller_identity_cache!(cache.device_identity_cache)
    BeatEngineCore.cuda_module().unsafe_free!(cache.device_bem_flux)
    for (name, sparse_cache) in pairs(cache.device_sparse_blocks)
        if name == :wall_impedance
            for wall_cache in sparse_cache
                release_cuda_sparse_scatter_cache!(wall_cache)
            end
        else
            release_cuda_sparse_scatter_cache!(sparse_cache)
        end
    end
    return nothing
end

function _accelerator_coupled_bem_blocks(
    accelerator,
    device_array,
    operators,
    identity_p1_p1,
    identity_p1_dp0,
    bem_flux,
    wavenumber::T;
    validation_diagnostics::Bool,
    keep_device::Bool=false,
    inputs_on_device::Bool=false,
    bem_motion_flux=nothing,
    bem_prescribed_neumann=nothing,
) where {T<:AbstractFloat}
    coupling = Complex{T}(0, 1) / wavenumber
    d_identity_p1_p1 = d_identity_p1_dp0 = d_bem_flux = nothing
    d_lhs = d_interface_block = d_interface_temp = d_rhs_operator = nothing
    d_motion_block = d_motion_temp = nothing
    d_prescribed_rhs = d_prescribed_temp = nothing
    try
        if inputs_on_device
            d_identity_p1_p1 = identity_p1_p1
            d_identity_p1_dp0 = identity_p1_dp0
            d_bem_flux = bem_flux
        else
            d_identity_p1_p1 = device_array(Complex{T}.(identity_p1_p1))
            d_identity_p1_dp0 = device_array(Complex{T}.(identity_p1_dp0))
            d_bem_flux = device_array(Complex{T}.(Matrix(bem_flux)))
        end
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
        if size(d_bem_flux, 2) > 0
            d_interface_temp = similar(d_interface_block)
            mul!(d_interface_block, operators.single_layer, d_bem_flux)
            mul!(d_interface_temp, operators.adjoint_double_layer, d_bem_flux)
            d_interface_block .+= coupling .* d_interface_temp
            mul!(d_interface_temp, d_identity_p1_dp0, d_bem_flux)
            d_interface_block .+= Complex{T}(0.5) * coupling .* d_interface_temp
        end
        if !isnothing(bem_motion_flux)
            d_motion_block = similar(
                operators.single_layer,
                Complex{T},
                size(operators.single_layer, 1),
                size(bem_motion_flux, 2),
            )
            d_motion_temp = similar(d_motion_block)
            mul!(d_motion_block, operators.single_layer, bem_motion_flux)
            mul!(d_motion_temp, operators.adjoint_double_layer, bem_motion_flux)
            d_motion_block .+= coupling .* d_motion_temp
            mul!(d_motion_temp, d_identity_p1_dp0, bem_motion_flux)
            d_motion_block .+= Complex{T}(0.5) * coupling .* d_motion_temp
        end
        if !isnothing(bem_prescribed_neumann)
            d_prescribed_rhs = similar(
                operators.single_layer,
                Complex{T},
                size(operators.single_layer, 1),
                size(bem_prescribed_neumann, 2),
            )
            d_prescribed_temp = similar(d_prescribed_rhs)
            mul!(d_prescribed_rhs, operators.single_layer, bem_prescribed_neumann)
            mul!(d_prescribed_temp, operators.adjoint_double_layer, bem_prescribed_neumann)
            d_prescribed_rhs .+= coupling .* d_prescribed_temp
            mul!(d_prescribed_temp, d_identity_p1_dp0, bem_prescribed_neumann)
            d_prescribed_rhs .+= Complex{T}(0.5) * coupling .* d_prescribed_temp
            # Prescribed Neumann data is a known RHS term; interface and motion
            # blocks use the opposite sign because their amplitudes are unknowns.
            d_prescribed_rhs .*= -one(Complex{T})
        end
        if validation_diagnostics
            d_rhs_operator = (
                -operators.single_layer .-
                coupling .* (
                    operators.adjoint_double_layer .+
                    Complex{T}(0.5) .* d_identity_p1_dp0
                )
            )
        end
        accelerator.synchronize()
        return (
            bem_lhs=keep_device ? d_lhs : Complex{T}.(Array(d_lhs)),
            bem_rhs_operator=if isnothing(d_rhs_operator)
                nothing
            elseif keep_device
                d_rhs_operator
            else
                Complex{T}.(Array(d_rhs_operator))
            end,
            bem_interface_block=keep_device ?
                                d_interface_block :
                                Complex{T}.(Array(d_interface_block)),
            bem_motion_block=if isnothing(d_motion_block)
                nothing
            elseif keep_device
                d_motion_block
            else
                Complex{T}.(Array(d_motion_block))
            end,
            bem_prescribed_rhs=if isnothing(d_prescribed_rhs)
                nothing
            elseif keep_device
                d_prescribed_rhs
            else
                Complex{T}.(Array(d_prescribed_rhs))
            end,
        )
    finally
        input_allocations = inputs_on_device ? () : (
            d_identity_p1_p1,
            d_identity_p1_dp0,
            d_bem_flux,
        )
        for value in (
            input_allocations...,
            d_interface_temp,
            d_motion_temp,
            d_prescribed_temp,
        )
            isnothing(value) || accelerator.unsafe_free!(value)
        end
        if !keep_device
            for value in (
                d_lhs,
                d_interface_block,
                d_motion_block,
                d_prescribed_rhs,
                d_rhs_operator,
            )
                isnothing(value) || accelerator.unsafe_free!(value)
            end
        end
    end
end

function _cuda_coupled_bem_blocks(args...; kwargs...)
    cuda = BeatEngineCore.cuda_module()
    return _accelerator_coupled_bem_blocks(cuda, cuda.CuArray, args...; kwargs...)
end

function _rocm_coupled_bem_blocks(args...; kwargs...)
    amdgpu = BeatEngineCore.amdgpu_module()
    return _accelerator_coupled_bem_blocks(amdgpu, amdgpu.ROCArray, args...; kwargs...)
end

function _build_cuda_fem_condensation(
    fem_system::SparseMatrixCSC{Complex{T}},
    interface_operators::InterfaceOperators{T},
    retained_vertices,
) where {T<:AbstractFloat}
    cuda = BeatEngineCore.cuda_module()
    cudss = _cudss_module()
    fem_count = size(fem_system, 1)
    retained_vertices = Int.(collect(retained_vertices))
    retained_set = Set(retained_vertices)
    interior_vertices = [
        vertex for vertex in 1:fem_count
        if !(vertex in retained_set)
    ]
    interior_load = interface_operators.fem_load[interior_vertices, :]
    nnz(interior_load) == 0 || error(
        "FEM static condensation currently requires interface loads to have support only on interface nodes.",
    )

    permutation = vcat(interior_vertices, retained_vertices)
    permuted_system = fem_system[permutation, permutation]
    device_system = cuda.CUSPARSE.CuSparseMatrixCSR(permuted_system)
    solver = cudss.CudssSolver(device_system, "G", 'F')
    cudss.cudss_set(solver, "schur_mode", 1)
    cudss.cudss_set(
        solver,
        "user_schur_indices",
        vcat(
            zeros(Cint, length(interior_vertices)),
            ones(Cint, length(retained_vertices)),
        ),
    )
    device_rhs = cuda.zeros(Complex{T}, fem_count)
    device_solution = similar(device_rhs)
    analysis_started = time_ns()
    cudss.cudss("analysis", solver, device_solution, device_rhs; asynchronous=false)
    analysis_s = (time_ns() - analysis_started) / 1.0e9
    factorization_started = time_ns()
    cudss.cudss(
        "factorization",
        solver,
        device_solution,
        device_rhs;
        asynchronous=false,
    )
    factorization_s = (time_ns() - factorization_started) / 1.0e9

    schur_shape = cudss.cudss_get(solver, "schur_shape")
    device_schur = cuda.zeros(Complex{T}, schur_shape[1], schur_shape[2])
    schur_descriptor = cudss.CudssMatrix(device_schur)
    schur_started = time_ns()
    cudss.cudss_set(solver, "schur_matrix", schur_descriptor.matrix)
    cudss.cudss_get(solver, "schur_matrix")
    cuda.synchronize()
    schur_extraction_s = (time_ns() - schur_started) / 1.0e9
    cuda.unsafe_free!(device_rhs)
    cuda.unsafe_free!(device_solution)

    return (
        backend=:cuda_cudss,
        solver=solver,
        device_system=device_system,
        device_schur=device_schur,
        schur_descriptor=schur_descriptor,
        permutation=permutation,
        interior_count=length(interior_vertices),
        retained_count=length(retained_vertices),
        timings=(
            analysis_s=analysis_s,
            factorization_s=factorization_s,
            schur_extraction_s=schur_extraction_s,
        ),
    )
end

function _release_cuda_fem_condensation!(condensation)
    isnothing(condensation) && return nothing
    cuda = BeatEngineCore.cuda_module()
    finalize(condensation.solver.data)
    finalize(condensation.solver.config)
    finalize(condensation.solver.matrix)
    finalize(condensation.schur_descriptor)
    cuda.unsafe_free!(condensation.device_schur)
    cuda.unsafe_free!(condensation.device_system.rowPtr)
    cuda.unsafe_free!(condensation.device_system.colVal)
    cuda.unsafe_free!(condensation.device_system.nzVal)
    return nothing
end

const ROCM_HYBRID_SCHUR_BLOCK_SIZE = 64

function _densify_sparse_columns!(
    dense::AbstractMatrix{ComplexF64},
    source::SparseMatrixCSC{ComplexF64,Int},
    columns,
)
    fill!(dense, zero(ComplexF64))
    row_indices = rowvals(source)
    values = nonzeros(source)
    for (local_column, source_column) in enumerate(columns)
        for index in nzrange(source, source_column)
            dense[row_indices[index], local_column] = values[index]
        end
    end
    return dense
end

function _resolved_schur_block_size(requested::Int, retained_count::Int)
    retained_count == 0 && return 0
    return max(1, min(requested, fld(retained_count, Threads.nthreads())))
end

function _blocked_umfpack_schur_complement(
    factorization,
    interior_to_retained::SparseMatrixCSC{ComplexF64,Int},
    retained_to_interior::SparseMatrixCSC{ComplexF64,Int},
    retained_system::Matrix{ComplexF64};
    block_size::Int=ROCM_HYBRID_SCHUR_BLOCK_SIZE,
)
    block_size > 0 || throw(ArgumentError("Schur-complement block size must be positive."))
    retained_count = size(interior_to_retained, 2)
    retained_count == 0 && return (
        schur=copy(retained_system),
        block_size=0,
        thread_count=1,
    )

    resolved_block_size = _resolved_schur_block_size(block_size, retained_count)
    block_starts = collect(1:resolved_block_size:retained_count)
    thread_count = max(1, min(Threads.nthreads(), length(block_starts)))
    schur = copy(retained_system)

    interior_count = size(interior_to_retained, 1)
    buffer_columns = min(resolved_block_size, retained_count)
    tasks = map(1:thread_count) do task_index
        Threads.@spawn begin
            task_factorization = copy(factorization)
            dense_columns = Matrix{ComplexF64}(undef, interior_count, buffer_columns)
            solved_columns = Matrix{ComplexF64}(undef, interior_count, buffer_columns)
            for block_index in task_index:thread_count:length(block_starts)
                block_start = block_starts[block_index]
                columns = block_start:min(
                    block_start + resolved_block_size - 1,
                    retained_count,
                )
                block_rhs = view(dense_columns, :, 1:length(columns))
                block_solution = view(solved_columns, :, 1:length(columns))
                _densify_sparse_columns!(block_rhs, interior_to_retained, columns)
                ldiv!(block_solution, task_factorization, block_rhs)
                mul!(
                    view(schur, :, columns),
                    retained_to_interior,
                    block_solution,
                    -one(ComplexF64),
                    one(ComplexF64),
                )
            end
        end
    end
    foreach(wait, tasks)
    return (
        schur=schur,
        block_size=resolved_block_size,
        thread_count=thread_count,
    )
end

function _build_rocm_hybrid_fem_condensation(
    fem_system::SparseMatrixCSC{Complex{T}},
    interface_operators::InterfaceOperators{T},
    retained_vertices,
) where {T<:AbstractFloat}
    amdgpu = BeatEngineCore.amdgpu_module()
    fem_count = size(fem_system, 1)
    retained_vertices = Int.(collect(retained_vertices))
    retained_set = Set(retained_vertices)
    interior_vertices = [vertex for vertex in 1:fem_count if !(vertex in retained_set)]
    interior_load = interface_operators.fem_load[interior_vertices, :]
    nnz(interior_load) == 0 || error(
        "FEM static condensation requires interface loads to have support only on retained nodes.",
    )

    partition_started = time_ns()
    interior_system = SparseMatrixCSC{ComplexF64,Int}(
        fem_system[interior_vertices, interior_vertices],
    )
    interior_to_retained = SparseMatrixCSC{ComplexF64,Int}(
        fem_system[interior_vertices, retained_vertices],
    )
    retained_to_interior = SparseMatrixCSC{ComplexF64,Int}(
        fem_system[retained_vertices, interior_vertices],
    )
    retained_system = Matrix{ComplexF64}(fem_system[retained_vertices, retained_vertices])
    partition_s = (time_ns() - partition_started) / 1.0e9

    factorization = nothing
    factorization_started = time_ns()
    if !isempty(interior_vertices)
        factorization = lu(interior_system)
    end
    factorization_s = (time_ns() - factorization_started) / 1.0e9

    schur_started = time_ns()
    schur_result = if isempty(interior_vertices)
        (
            schur=retained_system,
            block_size=0,
            thread_count=1,
        )
    else
        _blocked_umfpack_schur_complement(
            factorization,
            interior_to_retained,
            retained_to_interior,
            retained_system,
        )
    end
    schur_extraction_s = (time_ns() - schur_started) / 1.0e9

    upload_started = time_ns()
    device_schur = amdgpu.ROCArray(Complex{T}.(schur_result.schur))
    amdgpu.synchronize()
    upload_s = (time_ns() - upload_started) / 1.0e9
    return (
        backend=:rocm_hybrid,
        factorization=factorization,
        device_schur=device_schur,
        interior_vertices=interior_vertices,
        retained_vertices=retained_vertices,
        interior_to_retained=interior_to_retained,
        retained_to_interior=retained_to_interior,
        interior_count=length(interior_vertices),
        retained_count=length(retained_vertices),
        schur_block_size=schur_result.block_size,
        schur_thread_count=schur_result.thread_count,
        timings=(
            analysis_s=0.0,
            partition_s=partition_s,
            factorization_s=factorization_s,
            schur_extraction_s=schur_extraction_s,
            upload_s=upload_s,
        ),
    )
end

function _release_rocm_hybrid_fem_condensation!(condensation)
    isnothing(condensation) && return nothing
    isnothing(condensation.factorization) || finalize(condensation.factorization)
    BeatEngineCore.amdgpu_module().unsafe_free!(condensation.device_schur)
    return nothing
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
    symmetry_mode::Symbol=:off,
    static_condensation::Bool=false,
    bulk_loss_factor::T=zero(T),
    bulk_loss_factor_by_vertex=nothing,
    wall_impedances=NamedTuple[],
    transducers::AbstractVector{ElectrodynamicTransducer{T}}=ElectrodynamicTransducer{T}[],
    transducer_operators=nothing,
    prescribed_bem_normal_velocity=nothing,
) where {T<:AbstractFloat}
    static_condensation && !(bem_backend in (:cuda, :rocm)) && error(
        "FEM static condensation is currently available only for CUDA and ROCm coupled backends.",
    )
    static_condensation && validation_diagnostics && error(
        "FEM static condensation cannot be combined with full-matrix validation diagnostics.",
    )
    fem_stage_started = time_ns()
    resolved_transducer_operators = isnothing(transducer_operators) ?
                                    assemble_transducer_operators(
        fem_mesh,
        bem_mesh,
        transducers,
    ) : transducer_operators
    transducer_fem_vertices = if isempty(transducers)
        Int[]
    else
        unique(findnz(resolved_transducer_operators.fem_surface)[1])
    end
    retained_fem_vertices = sort(
        unique(vcat(interface_map.fem_vertex_indices, transducer_fem_vertices)),
    )
    prepared = isnothing(cache) ? prepare_coupled_cache(
        fem_mesh,
        bem_mesh,
        interface_map;
        quadrature_order=quadrature_order,
        singular_order=singular_order,
        bem_backend=bem_backend,
        symmetry_mode=symmetry_mode,
        retained_fem_vertices=retained_fem_vertices,
        bulk_loss_factor_by_vertex=(
            isnothing(bulk_loss_factor_by_vertex) ?
            fill(bulk_loss_factor, length(fem_mesh.vertices)) :
            bulk_loss_factor_by_vertex
        ),
        wall_impedances=wall_impedances,
    ) : cache
    prepared.bem_backend == bem_backend ||
        error("Coupled cache backend does not match requested BEM backend.")
    prepared.symmetry_mode == BeatEngineCore.normalized_symmetry_mode(symmetry_mode) ||
        error("Coupled cache symmetry mode does not match requested symmetry.")
    prepared.retained_fem_vertices == retained_fem_vertices ||
        error("Coupled cache retained FEM vertices do not match the current moving surfaces.")
    omega = T(2pi) * frequency_hz
    wavenumber = omega / sound_speed
    fem_system = assemble_fem_dynamic_stiffness(
        prepared.stiffness,
        prepared.mass,
        wavenumber;
        bulk_loss_mass=prepared.bulk_loss_mass,
    )
    wall_admittances = Complex{T}[
        miki_rigid_backed_surface_admittance(
            frequency_hz,
            sound_speed,
            density,
            operator.thickness_m,
            operator.flow_resistivity_pa_s_per_m2,
        )
        for operator in prepared.wall_impedance_operators
    ]
    for (operator, admittance) in zip(prepared.wall_impedance_operators, wall_admittances)
        fem_system -= Complex{T}(0, density * omega) * admittance .* operator.matrix
    end
    interface_operators = prepared.interface_operators
    transducer_count = length(transducers)
    size(resolved_transducer_operators.fem_surface, 2) == transducer_count ||
        error("FEM transducer operator count does not match the transducer list.")
    size(resolved_transducer_operators.bem_surface, 2) == transducer_count ||
        error("BEM transducer operator count does not match the transducer list.")
    normal_derivative_scale = Complex{T}(0, density * omega)
    bem_motion_flux = normal_derivative_scale .* Complex{T}.(
        resolved_transducer_operators.bem_normal_velocity
    )
    resolved_prescribed_bem_velocity = isnothing(prescribed_bem_normal_velocity) ?
                                       spzeros(T, length(bem_mesh.faces), 0) :
                                       T.(prescribed_bem_normal_velocity)
    size(resolved_prescribed_bem_velocity, 1) == length(bem_mesh.faces) || error(
        "Prescribed BEM normal velocity must contain one row per BEM face.",
    )
    bem_prescribed_neumann = normal_derivative_scale .* Complex{T}.(
        resolved_prescribed_bem_velocity
    )
    prescribed_bem_count = size(bem_prescribed_neumann, 2)
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
        device_image_singular_cache=prepared.device_image_singular_cache,
        symmetry_mode=prepared.symmetry_mode,
        return_device=bem_backend in (:cuda, :rocm),
        accelerator_quadrature=bem_backend in (:cuda, :rocm),
        device_singular_cache=prepared.device_singular_cache,
    )
    bem_operator_s = (time_ns() - bem_operator_started) / 1.0e9
    bem_matrix_started = time_ns()
    linear_backend = bem_backend in (:cuda, :rocm) && !validation_diagnostics ?
                     bem_backend : :cpu
    bem_blocks = if bem_backend in (:cuda, :rocm)
        accelerator = bem_backend == :cuda ?
                      BeatEngineCore.cuda_module() : BeatEngineCore.amdgpu_module()
        device_array = bem_backend == :cuda ? accelerator.CuArray : accelerator.ROCArray
        device_bem_motion_flux = nothing
        device_bem_prescribed_neumann = nothing
        try
            if transducer_count > 0
                device_bem_motion_flux = device_array(
                    Matrix(bem_motion_flux),
                )
            end
            if prescribed_bem_count > 0
                device_bem_prescribed_neumann = device_array(
                    Matrix(bem_prescribed_neumann),
                )
            end
            build_bem_blocks = bem_backend == :cuda ?
                               _cuda_coupled_bem_blocks : _rocm_coupled_bem_blocks
            build_bem_blocks(
                operators,
                prepared.device_identity_cache.identity_p1_p1,
                prepared.device_identity_cache.identity_p1_dp0,
                prepared.device_bem_flux,
                wavenumber;
                validation_diagnostics=validation_diagnostics,
                keep_device=linear_backend in (:cuda, :rocm),
                inputs_on_device=true,
                bem_motion_flux=device_bem_motion_flux,
                bem_prescribed_neumann=device_bem_prescribed_neumann,
            )
        finally
            release_operator_storage!(operators)
            isnothing(device_bem_motion_flux) ||
                accelerator.unsafe_free!(device_bem_motion_flux)
            isnothing(device_bem_prescribed_neumann) ||
                accelerator.unsafe_free!(device_bem_prescribed_neumann)
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
            bem_motion_block=transducer_count == 0 ?
                             nothing :
                             -(bem_rhs_operator * bem_motion_flux),
            bem_prescribed_rhs=prescribed_bem_count == 0 ?
                               nothing :
                               bem_rhs_operator * bem_prescribed_neumann,
        )
    end
    bem_lhs = bem_blocks.bem_lhs
    bem_rhs_operator = bem_blocks.bem_rhs_operator
    bem_prescribed_rhs = isnothing(bem_blocks.bem_prescribed_rhs) ?
                         zeros(Complex{T}, length(bem_mesh.vertices), 0) :
                         Complex{T}.(Array(bem_blocks.bem_prescribed_rhs))
    bem_matrix_s = (time_ns() - bem_matrix_started) / 1.0e9

    condensation_started = time_ns()
    condensation = if !static_condensation
        nothing
    elseif bem_backend == :cuda
        _build_cuda_fem_condensation(
            fem_system,
            interface_operators,
            retained_fem_vertices,
        )
    else
        _build_rocm_hybrid_fem_condensation(
            fem_system,
            interface_operators,
            retained_fem_vertices,
        )
    end
    fem_condensation_s = (time_ns() - condensation_started) / 1.0e9

    block_assembly_started = time_ns()
    fem_count = length(fem_mesh.vertices)
    bem_count = length(bem_mesh.vertices)
    interface_count = length(interface_map.fem_vertex_indices)
    retained_fem_count = length(retained_fem_vertices)
    formulation = static_condensation ? :fem_interface_condensed : :monolithic
    fem_range = 1:fem_count
    gamma_range = static_condensation ? (1:retained_fem_count) : nothing
    bem_range = static_condensation ?
                ((retained_fem_count + 1):(retained_fem_count + bem_count)) :
                ((fem_count + 1):(fem_count + bem_count))
    flux_range = static_condensation ?
                 (
        (retained_fem_count + bem_count + 1):
        (retained_fem_count + bem_count + interface_count)
    ) :
                 ((fem_count + bem_count + 1):(fem_count + bem_count + interface_count))
    acoustic_system_count = static_condensation ?
                            (retained_fem_count + interface_count + bem_count) :
                            (fem_count + bem_count + interface_count)
    mechanical_range = transducer_count == 0 ?
                       (1:0) :
                       ((acoustic_system_count + 1):(acoustic_system_count + transducer_count))
    electrical_range = transducer_count == 0 ?
                       (1:0) :
                       (
        (acoustic_system_count + transducer_count + 1):
        (acoustic_system_count + 2 * transducer_count)
    )
    system_count = acoustic_system_count + 2 * transducer_count
    coupled_fem_range = static_condensation ? gamma_range : fem_range
    coupled_fem_vertices = static_condensation ? retained_fem_vertices : collect(fem_range)
    mechanical_impedance = Complex{T}[
        Complex{T}(
            transducer.rms_n_s_per_m,
            -omega * transducer.mmd_kg + inv(omega * transducer.cms_m_per_n),
        )
        for transducer in transducers
    ]
    electrical_impedance = Complex{T}[
        Complex{T}(transducer.re_ohm, -omega * transducer.le_h)
        for transducer in transducers
    ]
    force_factor = T[transducer.bl_n_per_a for transducer in transducers]
    coupled = if linear_backend in (:cuda, :rocm)
        accelerator = linear_backend == :cuda ?
                      BeatEngineCore.cuda_module() : BeatEngineCore.amdgpu_module()
        device_array = linear_backend == :cuda ? accelerator.CuArray : accelerator.ROCArray
        scatter_sparse! = linear_backend == :cuda ?
                          scatter_cuda_sparse_to_dense! : scatter_rocm_sparse_to_dense!
        d_coupled = accelerator.zeros(Complex{T}, system_count, system_count)
        transducer_allocations = Any[]
        try
            if static_condensation
                view(d_coupled, gamma_range, gamma_range) .= condensation.device_schur
                scatter_sparse!(
                    d_coupled,
                    prepared.device_sparse_blocks.condensed_fem_load;
                    column_offset=first(flux_range) - 1,
                    alpha=-one(Complex{T}),
                )
                scatter_sparse!(
                    d_coupled,
                    prepared.device_sparse_blocks.condensed_fem_trace;
                    row_offset=first(flux_range) - 1,
                )
            else
                scatter_sparse!(
                    d_coupled,
                    prepared.device_sparse_blocks.stiffness,
                )
                scatter_sparse!(
                    d_coupled,
                    prepared.device_sparse_blocks.mass;
                    alpha=Complex{T}(-(wavenumber^2), 0),
                    add=true,
                )
                scatter_sparse!(
                    d_coupled,
                    prepared.device_sparse_blocks.bulk_loss_mass;
                    alpha=Complex{T}(0, -(wavenumber^2)),
                    add=true,
                )
                for (wall_cache, admittance) in zip(
                    prepared.device_sparse_blocks.wall_impedance,
                    wall_admittances,
                )
                    scatter_sparse!(
                        d_coupled,
                        wall_cache;
                        alpha=-Complex{T}(0, density * omega) * admittance,
                        add=true,
                    )
                end
                scatter_sparse!(
                    d_coupled,
                    prepared.device_sparse_blocks.fem_load;
                    column_offset=first(flux_range) - 1,
                    alpha=-one(Complex{T}),
                )
                scatter_sparse!(
                    d_coupled,
                    prepared.device_sparse_blocks.fem_trace;
                    row_offset=first(flux_range) - 1,
                )
            end
            view(d_coupled, bem_range, bem_range) .= bem_lhs
            view(d_coupled, bem_range, flux_range) .= bem_blocks.bem_interface_block
            scatter_sparse!(
                d_coupled,
                prepared.device_sparse_blocks.bem_trace;
                row_offset=first(flux_range) - 1,
                column_offset=first(bem_range) - 1,
                alpha=-one(Complex{T}),
            )
            if transducer_count > 0
                d_fem_motion = device_array(
                    -normal_derivative_scale .* Complex{T}.(
                        Matrix(
                            resolved_transducer_operators.fem_surface[
                                coupled_fem_vertices,
                                :,
                            ],
                        )
                    ),
                )
                d_fem_force = device_array(
                    -Complex{T}.(
                        transpose(
                            Matrix(
                                resolved_transducer_operators.fem_force[
                                    coupled_fem_vertices,
                                    :,
                                ],
                            ),
                        )
                    ),
                )
                d_bem_force = device_array(
                    Complex{T}.(
                        transpose(Matrix(resolved_transducer_operators.bem_force))
                    ),
                )
                d_mechanical = device_array(Matrix(Diagonal(mechanical_impedance)))
                d_electrical = device_array(Matrix(Diagonal(electrical_impedance)))
                d_force_factor = device_array(Matrix(Diagonal(Complex{T}.(force_factor))))
                append!(
                    transducer_allocations,
                    (
                        d_fem_motion,
                        d_fem_force,
                        d_bem_force,
                        d_mechanical,
                        d_electrical,
                        d_force_factor,
                    ),
                )
                view(d_coupled, coupled_fem_range, mechanical_range) .= d_fem_motion
                view(d_coupled, bem_range, mechanical_range) .= bem_blocks.bem_motion_block
                view(d_coupled, mechanical_range, coupled_fem_range) .= d_fem_force
                view(d_coupled, mechanical_range, bem_range) .= d_bem_force
                view(d_coupled, mechanical_range, mechanical_range) .= d_mechanical
                view(d_coupled, mechanical_range, electrical_range) .= -d_force_factor
                view(d_coupled, electrical_range, mechanical_range) .= d_force_factor
                view(d_coupled, electrical_range, electrical_range) .= d_electrical
            end
            accelerator.synchronize()
        finally
            accelerator.unsafe_free!(bem_lhs)
            accelerator.unsafe_free!(bem_blocks.bem_interface_block)
            isnothing(bem_blocks.bem_motion_block) ||
                accelerator.unsafe_free!(bem_blocks.bem_motion_block)
            isnothing(bem_blocks.bem_prescribed_rhs) ||
                accelerator.unsafe_free!(bem_blocks.bem_prescribed_rhs)
            for allocation in transducer_allocations
                accelerator.unsafe_free!(allocation)
            end
        end
        d_coupled
    else
        host_coupled = zeros(Complex{T}, system_count, system_count)
        host_coupled[fem_range, fem_range] = Matrix(fem_system)
        host_coupled[fem_range, flux_range] =
            -Complex{T}.(Matrix(interface_operators.fem_load))
        host_coupled[bem_range, bem_range] = bem_lhs
        host_coupled[bem_range, flux_range] = bem_blocks.bem_interface_block
        host_coupled[flux_range, fem_range] =
            Complex{T}.(Matrix(interface_operators.fem_trace))
        host_coupled[flux_range, bem_range] =
            -Complex{T}.(Matrix(interface_operators.bem_trace))
        if transducer_count > 0
            host_coupled[fem_range, mechanical_range] =
                -normal_derivative_scale .* Complex{T}.(
                    Matrix(resolved_transducer_operators.fem_surface)
                )
            host_coupled[bem_range, mechanical_range] = bem_blocks.bem_motion_block
            host_coupled[mechanical_range, fem_range] =
                -Complex{T}.(
                    transpose(Matrix(resolved_transducer_operators.fem_force))
                )
            host_coupled[mechanical_range, bem_range] =
                Complex{T}.(
                    transpose(Matrix(resolved_transducer_operators.bem_force))
                )
            host_coupled[mechanical_range, mechanical_range] =
                Matrix(Diagonal(mechanical_impedance))
            host_coupled[mechanical_range, electrical_range] =
                -Matrix(Diagonal(Complex{T}.(force_factor)))
            host_coupled[electrical_range, mechanical_range] =
                Matrix(Diagonal(Complex{T}.(force_factor)))
            host_coupled[electrical_range, electrical_range] =
                Matrix(Diagonal(electrical_impedance))
        end
        host_coupled
    end
    block_assembly_s = (time_ns() - block_assembly_started) / 1.0e9

    coupled_factorization_started = time_ns()
    factorization = if validation_diagnostics
        lu(coupled)
    elseif linear_backend == :rocm
        rocm_dense_lu!(coupled)
    else
        lu!(coupled)
    end
    linear_backend == :cuda && BeatEngineCore.cuda_module().synchronize()
    linear_backend == :rocm && BeatEngineCore.amdgpu_module().synchronize()
    coupled_factorization_s = (time_ns() - coupled_factorization_started) / 1.0e9
    replay_factorization_started = time_ns()
    bem_factorization = validation_diagnostics ? lu(bem_lhs) : nothing
    replay_factorization_s = (time_ns() - replay_factorization_started) / 1.0e9
    return (
        fem_mesh=fem_mesh,
        bem_mesh=bem_mesh,
        interface_map=interface_map,
        interface_operators=interface_operators,
        transducers=transducers,
        transducer_operators=resolved_transducer_operators,
        density=density,
        bulk_loss_factor=maximum(prepared.bulk_loss_factor_by_vertex; init=zero(T)),
        bulk_loss_factor_by_vertex=prepared.bulk_loss_factor_by_vertex,
        wall_admittances=wall_admittances,
        omega=omega,
        wavenumber=wavenumber,
        field_cache=prepared.field_cache,
        coupled=validation_diagnostics ? coupled : nothing,
        factorization=factorization,
        formulation=formulation,
        condensation=condensation,
        condensation_timings=isnothing(condensation) ? nothing : condensation.timings,
        fem_range=fem_range,
        gamma_range=gamma_range,
        retained_fem_vertices=retained_fem_vertices,
        bem_range=bem_range,
        flux_range=flux_range,
        mechanical_range=mechanical_range,
        electrical_range=electrical_range,
        bem_lhs=validation_diagnostics ? bem_lhs : nothing,
        bem_factorization=bem_factorization,
        bem_rhs_operator=validation_diagnostics ? bem_rhs_operator : nothing,
        prescribed_bem_rhs=bem_prescribed_rhs,
        prescribed_bem_neumann=bem_prescribed_neumann,
        bem_backend=bem_backend,
        linear_backend=linear_backend,
        symmetry_mode=prepared.symmetry_mode,
        cache=prepared,
        owns_cache=isnothing(cache),
        validation_diagnostics=validation_diagnostics,
        scalar_type=T,
        full_system_order=fem_count + bem_count + interface_count + 2 * transducer_count,
        solved_system_order=system_count,
        timings=(
            fem_system_s=fem_system_s,
            bem_operator_s=bem_operator_s,
            bem_matrix_s=bem_matrix_s,
            fem_condensation_s=fem_condensation_s,
            block_assembly_s=block_assembly_s,
            coupled_factorization_s=coupled_factorization_s,
            replay_factorization_s=replay_factorization_s,
        ),
    )
end

function release_coupled_system!(system)
    if system.linear_backend in (:cuda, :rocm)
        accelerator = system.linear_backend == :cuda ?
                      BeatEngineCore.cuda_module() : BeatEngineCore.amdgpu_module()
        accelerator.unsafe_free!(system.factorization.factors)
        accelerator.unsafe_free!(system.factorization.ipiv)
    end
    if system.linear_backend == :cuda
        _release_cuda_fem_condensation!(system.condensation)
    elseif system.linear_backend == :rocm && system.formulation == :fem_interface_condensed
        _release_rocm_hybrid_fem_condensation!(system.condensation)
    end
    system.owns_cache && release_coupled_cache!(system.cache)
    return nothing
end

function _coupled_solution_from_parts(
    system,
    fem_pressure,
    bem_pressure,
    interface_flux;
    diaphragm_velocity=zeros(Complex{system.scalar_type}, length(system.transducers)),
    voice_coil_current=zeros(Complex{system.scalar_type}, length(system.transducers)),
    prescribed_bem_neumann=zeros(
        Complex{system.scalar_type},
        length(system.bem_mesh.faces),
    ),
    solution=nothing,
    rhs=nothing,
    fem_rhs_condensation_s=nothing,
    fem_reconstruction_s=nothing,
)
    T = system.scalar_type
    bem_neumann = (
        Complex{T}.(system.interface_operators.bem_flux) * interface_flux +
        Complex{T}(0, system.density * system.omega) .*
        (
            Complex{T}.(system.transducer_operators.bem_normal_velocity) *
            diaphragm_velocity
        ) +
        prescribed_bem_neumann
    )

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
        isnothing(solution) && error("Validation diagnostics require the monolithic solution vector.")
        isnothing(rhs) && error("Validation diagnostics require the monolithic right-hand side.")
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
        diaphragm_velocity=diaphragm_velocity,
        voice_coil_current=voice_coil_current,
        fem_rhs_condensation_s=fem_rhs_condensation_s,
        fem_reconstruction_s=fem_reconstruction_s,
        relative_residual=relative_residual,
        pressure_continuity_error=norm(pressure_jump) / pressure_scale,
        flux_conservation_error=abs(fem_integrated_flux - bem_integrated_flux_along_fem_normal) / flux_scale,
        all_bem_replay_error=replay_error,
        interface_map=system.interface_map,
        interface_operators=system.interface_operators,
    )
end

function _coupled_solution_from_vector(
    system,
    solution,
    rhs;
    prescribed_bem_neumann=zeros(
        Complex{system.scalar_type},
        length(system.bem_mesh.faces),
    ),
)
    return _coupled_solution_from_parts(
        system,
        solution[system.fem_range],
        solution[system.bem_range],
        solution[system.flux_range];
        diaphragm_velocity=solution[system.mechanical_range],
        voice_coil_current=solution[system.electrical_range],
        prescribed_bem_neumann=prescribed_bem_neumann,
        solution=solution,
        rhs=rhs,
    )
end

function _solve_rocm_hybrid_condensed_excitations(
    system,
    fem_rhs,
    bem_rhs,
    electrical_rhs,
    prescribed_bem_neumann,
)
    T = system.scalar_type
    amdgpu = BeatEngineCore.amdgpu_module()
    condensation = system.condensation
    excitation_count = size(fem_rhs, 2)

    rhs_condensation_started = time_ns()
    reduced_fem_rhs = if condensation.interior_count == 0
        Complex{T}.(fem_rhs[condensation.retained_vertices, :])
    else
        interior_rhs = ComplexF64.(fem_rhs[condensation.interior_vertices, :])
        interior_forward = condensation.factorization \ interior_rhs
        Complex{T}.(
            fem_rhs[condensation.retained_vertices, :] .-
            condensation.retained_to_interior * interior_forward
        )
    end
    host_rhs = zeros(Complex{T}, size(system.factorization, 1), excitation_count)
    host_rhs[system.gamma_range, :] = reduced_fem_rhs
    host_rhs[system.bem_range, :] = bem_rhs
    host_rhs[system.electrical_range, :] = electrical_rhs
    fem_rhs_condensation_s = (time_ns() - rhs_condensation_started) / 1.0e9

    d_rhs = amdgpu.ROCArray(host_rhs)
    d_solution = nothing
    host_solution = nothing
    try
        d_solution = solve_rocm_dense_factorization(system.factorization, d_rhs)
        amdgpu.synchronize()
        host_solution = Complex{T}.(Array(d_solution))
    finally
        amdgpu.unsafe_free!(d_rhs)
        isnothing(d_solution) || amdgpu.unsafe_free!(d_solution)
    end

    reconstruction_started = time_ns()
    retained_pressure = view(host_solution, system.gamma_range, :)
    fem_pressure = zeros(Complex{T}, length(system.fem_mesh.vertices), excitation_count)
    fem_pressure[condensation.retained_vertices, :] .= retained_pressure
    if condensation.interior_count > 0
        interior_rhs = (
            ComplexF64.(fem_rhs[condensation.interior_vertices, :]) .-
            condensation.interior_to_retained * ComplexF64.(retained_pressure)
        )
        fem_pressure[condensation.interior_vertices, :] .= Complex{T}.(
            condensation.factorization \ interior_rhs
        )
    end
    fem_reconstruction_s = (time_ns() - reconstruction_started) / 1.0e9
    bem_pressure = view(host_solution, system.bem_range, :)
    interface_flux = view(host_solution, system.flux_range, :)
    diaphragm_velocity = view(host_solution, system.mechanical_range, :)
    voice_coil_current = view(host_solution, system.electrical_range, :)
    return [
        _coupled_solution_from_parts(
            system,
            fem_pressure[:, column],
            bem_pressure[:, column],
            interface_flux[:, column];
            diaphragm_velocity=diaphragm_velocity[:, column],
            voice_coil_current=voice_coil_current[:, column],
            prescribed_bem_neumann=prescribed_bem_neumann[:, column],
            fem_rhs_condensation_s=fem_rhs_condensation_s,
            fem_reconstruction_s=fem_reconstruction_s,
        )
        for column in axes(fem_pressure, 2)
    ]
end

function solve_coupled_excitations(system, excitations)
    T = system.scalar_type
    requested = collect(excitations)
    isempty(requested) && error("At least one coupled excitation is required.")
    excitation_count = length(requested)
    fem_rhs = zeros(
        Complex{T},
        length(system.fem_mesh.vertices),
        excitation_count,
    )
    bem_rhs = zeros(
        Complex{T},
        length(system.bem_mesh.vertices),
        excitation_count,
    )
    prescribed_bem_neumann = zeros(
        Complex{T},
        length(system.bem_mesh.faces),
        excitation_count,
    )
    electrical_rhs = zeros(Complex{T}, length(system.transducers), excitation_count)
    for (column, excitation) in enumerate(requested)
        kind = Symbol(excitation.kind)
        amplitude = Complex{T}(excitation.amplitude)
        if kind == :normal_velocity
            fem_boundary_tags = hasproperty(excitation, :fem_boundary_tags) ?
                                Int.(excitation.fem_boundary_tags) :
                                [Int(excitation.radiator_tag)]
            fem_boundary_weights = hasproperty(excitation, :fem_boundary_weights) ?
                                   T.(excitation.fem_boundary_weights) :
                                   ones(T, length(fem_boundary_tags))
            length(fem_boundary_tags) == length(fem_boundary_weights) || error(
                "Prescribed FEM boundary tags and weights must have the same length.",
            )
            all(weight -> isfinite(weight) && weight > zero(T), fem_boundary_weights) ||
                error("Prescribed FEM boundary weights must be finite and greater than zero.")
            for (tag, weight) in zip(fem_boundary_tags, fem_boundary_weights)
                fem_rhs[:, column] .+= assemble_prescribed_velocity_load(
                    system.fem_mesh,
                    tag,
                    system.density,
                    system.omega,
                    amplitude * weight,
                )
            end
            bem_source_index = hasproperty(excitation, :bem_source_index) ?
                               Int(excitation.bem_source_index) :
                               0
            if bem_source_index > 0
                bem_source_index <= size(system.prescribed_bem_rhs, 2) || error(
                    "Prescribed BEM source index $bem_source_index is unavailable.",
                )
                bem_rhs[:, column] .= amplitude .* view(
                    system.prescribed_bem_rhs,
                    :,
                    bem_source_index,
                )
                prescribed_bem_neumann[:, column] .= amplitude .* view(
                    system.prescribed_bem_neumann,
                    :,
                    bem_source_index,
                )
            end
            isempty(fem_boundary_tags) && bem_source_index == 0 && error(
                "A prescribed-velocity excitation must own at least one FEM or BEM moving boundary.",
            )
        elseif kind == :voltage
            transducer_index = Int(excitation.transducer_index)
            1 <= transducer_index <= length(system.transducers) || error(
                "Voltage excitation references invalid transducer index $transducer_index.",
            )
            electrical_rhs[transducer_index, column] = amplitude
        else
            error("Unsupported coupled excitation kind: $kind.")
        end
    end
    if system.formulation == :fem_interface_condensed && system.linear_backend == :rocm
        return _solve_rocm_hybrid_condensed_excitations(
            system,
            fem_rhs,
            bem_rhs,
            electrical_rhs,
            prescribed_bem_neumann,
        )
    elseif system.formulation == :fem_interface_condensed
        cuda = BeatEngineCore.cuda_module()
        cudss = _cudss_module()
        condensation = system.condensation
        permuted_rhs = fem_rhs[condensation.permutation, :]
        d_fem_rhs = cuda.CuArray(permuted_rhs)
        d_forward = similar(d_fem_rhs)
        d_rhs = cuda.zeros(
            Complex{T},
            size(system.factorization, 1),
            excitation_count,
        )
        d_bem_rhs = nothing
        d_electrical_rhs = nothing
        d_solution = nothing
        try
            cudss.cudss(
                "solve_fwd_schur",
                condensation.solver,
                d_forward,
                d_fem_rhs;
                asynchronous=false,
            )
            schur_rows = (condensation.interior_count + 1):length(condensation.permutation)
            view(d_rhs, system.gamma_range, :) .= view(d_forward, schur_rows, :)
            if size(system.prescribed_bem_rhs, 2) > 0
                d_bem_rhs = cuda.CuArray(bem_rhs)
                view(d_rhs, system.bem_range, :) .= d_bem_rhs
            end
            if !isempty(system.transducers)
                d_electrical_rhs = cuda.CuArray(electrical_rhs)
                view(d_rhs, system.electrical_range, :) .= d_electrical_rhs
            end
            d_solution = system.factorization \ d_rhs
            view(d_forward, schur_rows, :) .= view(d_solution, system.gamma_range, :)
            cudss.cudss(
                "solve_bwd_schur",
                condensation.solver,
                d_fem_rhs,
                d_forward;
                asynchronous=false,
            )
            view(d_fem_rhs, schur_rows, :) .= view(d_solution, system.gamma_range, :)
            cuda.synchronize()
            permuted_pressure = Complex{T}.(Array(d_fem_rhs))
            fem_pressure = similar(permuted_pressure)
            fem_pressure[condensation.permutation, :] = permuted_pressure
            bem_pressure = Complex{T}.(Array(view(d_solution, system.bem_range, :)))
            interface_flux = Complex{T}.(Array(view(d_solution, system.flux_range, :)))
            diaphragm_velocity = Complex{T}.(
                Array(view(d_solution, system.mechanical_range, :)),
            )
            voice_coil_current = Complex{T}.(
                Array(view(d_solution, system.electrical_range, :)),
            )
            return [
                _coupled_solution_from_parts(
                    system,
                    fem_pressure[:, column],
                    bem_pressure[:, column],
                    interface_flux[:, column];
                    diaphragm_velocity=diaphragm_velocity[:, column],
                    voice_coil_current=voice_coil_current[:, column],
                    prescribed_bem_neumann=prescribed_bem_neumann[:, column],
                )
                for column in axes(fem_pressure, 2)
            ]
        finally
            cuda.unsafe_free!(d_fem_rhs)
            cuda.unsafe_free!(d_forward)
            cuda.unsafe_free!(d_rhs)
            isnothing(d_bem_rhs) || cuda.unsafe_free!(d_bem_rhs)
            isnothing(d_electrical_rhs) || cuda.unsafe_free!(d_electrical_rhs)
            isnothing(d_solution) || cuda.unsafe_free!(d_solution)
        end
    end

    rhs = zeros(Complex{T}, size(system.factorization, 1), excitation_count)
    rhs[system.fem_range, :] = fem_rhs
    rhs[system.bem_range, :] = bem_rhs
    rhs[system.electrical_range, :] = electrical_rhs
    solution = if system.linear_backend in (:cuda, :rocm)
        accelerator = system.linear_backend == :cuda ?
                      BeatEngineCore.cuda_module() : BeatEngineCore.amdgpu_module()
        device_array = system.linear_backend == :cuda ?
                       accelerator.CuArray : accelerator.ROCArray
        d_rhs = device_array(rhs)
        d_solution = nothing
        try
            d_solution = system.linear_backend == :rocm ?
                         solve_rocm_dense_factorization(system.factorization, d_rhs) :
                         system.factorization \ d_rhs
            accelerator.synchronize()
            Complex{T}.(Array(d_solution))
        finally
            accelerator.unsafe_free!(d_rhs)
            isnothing(d_solution) || accelerator.unsafe_free!(d_solution)
        end
    else
        system.factorization \ rhs
    end
    return [
        _coupled_solution_from_vector(
            system,
            solution[:, column],
            rhs[:, column];
            prescribed_bem_neumann=prescribed_bem_neumann[:, column],
        )
        for column in axes(solution, 2)
    ]
end

function solve_coupled_systems(
    system,
    radiator_tags;
    radiator_velocities=nothing,
)
    T = system.scalar_type
    tags = Int.(collect(radiator_tags))
    isempty(tags) && error("At least one radiator tag is required.")
    velocities = isnothing(radiator_velocities) ?
                 fill(Complex{T}(1, 0), length(tags)) :
                 Complex{T}.(collect(radiator_velocities))
    length(velocities) == length(tags) ||
        error("Radiator tags and velocities must have the same length.")
    return solve_coupled_excitations(
        system,
        [
            (
                kind=:normal_velocity,
                radiator_tag=tag,
                transducer_index=0,
                amplitude=velocity,
            )
            for (tag, velocity) in zip(tags, velocities)
        ],
    )
end

function solve_coupled_system(
    system,
    radiator_tag::Int;
    radiator_velocity=ComplexF64(1, 0),
)
    return only(
        solve_coupled_systems(
            system,
            [radiator_tag];
            radiator_velocities=[radiator_velocity],
        ),
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
    symmetry_mode::Symbol=:off,
    bulk_loss_factor::T=zero(T),
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
        symmetry_mode=symmetry_mode,
        bulk_loss_factor=bulk_loss_factor,
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

function _outward_boundary_normals(mesh::VolumeMesh{T}) where {T<:AbstractFloat}
    face_counts = Dict{NTuple{3,Int},Int}()
    opposite_vertex = Dict{NTuple{3,Int},Int}()
    for tetrahedron in mesh.tetrahedra
        faces = _tetrahedron_faces(tetrahedron)
        opposites = (
            tetrahedron[4],
            tetrahedron[3],
            tetrahedron[2],
            tetrahedron[1],
        )
        for (face, opposite) in zip(faces, opposites)
            face_counts[face] = get(face_counts, face, 0) + 1
            opposite_vertex[face] = opposite
        end
    end

    return [
        begin
            key = _sorted_face(face)
            get(face_counts, key, 0) == 1 ||
                error("FEM boundary triangle is not an exterior tetrahedral facet.")
            normal = _triangle_normal(mesh.vertices, face)
            centroid = (
                mesh.vertices[face[1]] +
                mesh.vertices[face[2]] +
                mesh.vertices[face[3]]
            ) / T(3)
            inward = mesh.vertices[opposite_vertex[key]] - centroid
            dot(normal, inward) > zero(T) ? -normal : normal
        end
        for face in mesh.boundary_faces
    ]
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
