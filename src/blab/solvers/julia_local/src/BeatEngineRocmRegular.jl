function _rocm_geometry_arrays(mesh::BoundaryMesh{T}) where {T<:AbstractFloat}
    face_count = length(mesh.faces)
    face_vertices = Matrix{T}(undef, face_count, 9)
    normals = Matrix{T}(undef, face_count, 3)
    curls = Matrix{T}(undef, face_count, 9)
    faces = Matrix{Int32}(undef, face_count, 3)
    areas = Vector{T}(undef, face_count)

    for element_index in 1:face_count
        vertices = mesh.face_vertices[element_index]
        normal = mesh.normals[element_index]
        element_curls = surface_curls(vertices, normal)
        face = mesh.faces[element_index]
        areas[element_index] = mesh.areas[element_index]
        for local_index in 1:3
            column = 3 * (local_index - 1)
            face_vertices[element_index, column + 1] = vertices[local_index][1]
            face_vertices[element_index, column + 2] = vertices[local_index][2]
            face_vertices[element_index, column + 3] = vertices[local_index][3]
            normals[element_index, local_index] = normal[local_index]
            faces[element_index, local_index] = Int32(face[local_index])
            curls[element_index, column + 1] = element_curls[local_index][1]
            curls[element_index, column + 2] = element_curls[local_index][2]
            curls[element_index, column + 3] = element_curls[local_index][3]
        end
    end
    return face_vertices, normals, areas, faces, curls
end

function _rocm_rule_arrays(rule::TriangleRule{T}) where {T<:AbstractFloat}
    rule_count = length(rule.points)
    points = Matrix{T}(undef, rule_count, 2)
    weights = Vector{T}(undef, rule_count)
    for rule_index in 1:rule_count
        points[rule_index, 1] = rule.points[rule_index][1]
        points[rule_index, 2] = rule.points[rule_index][2]
        weights[rule_index] = rule.weights[rule_index]
    end
    return points, weights
end

function _rocm_incident_element_arrays(
    p1_space::P1Space,
    dp0_space::DP0Space,
    element_indices,
)
    incidents = [Tuple{Int32,Int32}[] for _ in 1:p1_space.global_dof_count]
    dp0_elements = zeros(Int32, dp0_space.global_dof_count)
    for element_index in element_indices
        p1_dofs = p1_space.local_to_global[element_index]
        for local_index in 1:3
            push!(incidents[p1_dofs[local_index]], (Int32(element_index), Int32(local_index)))
        end
        dp0_dof = dp0_space.local_to_global[element_index]
        dp0_elements[dp0_dof] == 0 || error("ROCm native assembly requires one element per DP0 degree of freedom.")
        dp0_elements[dp0_dof] = Int32(element_index)
    end

    offsets = Vector{Int32}(undef, length(incidents) + 1)
    incident_elements = Int32[]
    incident_local_indices = Int32[]
    offsets[1] = 1
    for row in eachindex(incidents)
        for (element_index, local_index) in incidents[row]
            push!(incident_elements, element_index)
            push!(incident_local_indices, local_index)
        end
        offsets[row + 1] = Int32(length(incident_elements) + 1)
    end
    return offsets, incident_elements, incident_local_indices, dp0_elements
end

function build_rocm_regular_assembly_cache(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    rule::TriangleRule{T};
    singular_order::Int=2,
    element_indices=eachindex(mesh.faces),
    threaded::Bool=true,
    assembly_mode=nothing,
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    normalized_symmetry_mode(symmetry_mode) == :off ||
        error("The initial BEAT Engine ROCm backend supports only symmetry_mode=:off.")
    _require_rocm!()
    indices = collect(element_indices)
    host_cache = if _normalized_rocm_assembly_mode(assembly_mode) == :host_staged
        build_beat_cpu_assembly_cache(
            mesh,
            p1_space,
            dp0_space,
            rule;
            singular_order=singular_order,
            element_indices=indices,
            threaded=threaded,
            symmetry_mode=:off,
        )
    else
        nothing
    end
    face_vertices, normals, areas, faces, curls = _rocm_geometry_arrays(mesh)
    rule_points, rule_weights = _rocm_rule_arrays(rule)
    vertex_offsets, incident_elements, incident_local_indices, dp0_elements =
        _rocm_incident_element_arrays(p1_space, dp0_space, indices)

    return RocmRegularAssemblyCache{T,typeof(host_cache)}(
        host_cache,
        AMDGPU.ROCArray(face_vertices),
        AMDGPU.ROCArray(normals),
        AMDGPU.ROCArray(areas),
        AMDGPU.ROCArray(faces),
        AMDGPU.ROCArray(curls),
        AMDGPU.ROCArray(rule_points),
        AMDGPU.ROCArray(rule_weights),
        AMDGPU.ROCArray(vertex_offsets),
        AMDGPU.ROCArray(incident_elements),
        AMDGPU.ROCArray(incident_local_indices),
        AMDGPU.ROCArray(dp0_elements),
        indices,
        length(mesh.faces),
        p1_space.global_dof_count,
        dp0_space.global_dof_count,
        length(rule.points),
    )
end

function release_rocm_regular_assembly_cache!(cache::RocmRegularAssemblyCache)
    AMDGPU.unsafe_free!(cache.face_vertices)
    AMDGPU.unsafe_free!(cache.normals)
    AMDGPU.unsafe_free!(cache.areas)
    AMDGPU.unsafe_free!(cache.faces)
    AMDGPU.unsafe_free!(cache.curls)
    AMDGPU.unsafe_free!(cache.rule_points)
    AMDGPU.unsafe_free!(cache.rule_weights)
    AMDGPU.unsafe_free!(cache.vertex_offsets)
    AMDGPU.unsafe_free!(cache.incident_elements)
    AMDGPU.unsafe_free!(cache.incident_local_indices)
    AMDGPU.unsafe_free!(cache.dp0_elements)
    return nothing
end
