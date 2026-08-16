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

function _rocm_local_dof_arrays(
    p1_space::P1Space,
    dp0_space::DP0Space,
    element_indices,
    face_count::Int,
)
    p1_dofs = zeros(Int32, face_count, 3)
    element_dp0_dofs = zeros(Int32, face_count)
    for element_index in element_indices
        local_p1_dofs = p1_space.local_to_global[element_index]
        for local_index in 1:3
            p1_dofs[element_index, local_index] = Int32(local_p1_dofs[local_index])
        end
        element_dp0_dofs[element_index] = Int32(dp0_space.local_to_global[element_index])
    end
    return p1_dofs, element_dp0_dofs
end

function _rocm_element_color_arrays(mesh::BoundaryMesh, element_indices)
    groups = _beat_cpu_element_color_groups(mesh, element_indices)
    offsets = Vector{Int}(undef, length(groups) + 1)
    elements = Int32[]
    offsets[1] = 1
    for (color, group) in enumerate(groups)
        append!(elements, Int32.(group))
        offsets[color + 1] = length(elements) + 1
    end
    return offsets, elements
end

function _rocm_image_singular_host_cache(
    mesh::BoundaryMesh{T},
    singular_order::Int,
    element_indices,
    transform::SymmetryTransform;
    tolerance::T=T(1e-8),
) where {T<:AbstractFloat}
    pairs_by_test = [SingularCorrectionPair{T}[] for _ in eachindex(mesh.faces)]
    base_rules = Dict(
        :coincident => duffy_rule(T, singular_order, :coincident),
        :edge_adjacent => duffy_rule(T, singular_order, :edge_adjacent),
        :vertex_adjacent => duffy_rule(T, singular_order, :vertex_adjacent),
    )
    rules = DuffyRule{T}[]
    rule_indices = Dict{NTuple{5,Int},Int}()
    pairs = SingularCorrectionPair{T}[]
    for (test_index, trial_index) in image_singular_candidates(mesh, element_indices, transform; tolerance=tolerance)
        trial_vertices = reflect_vertices(transform, mesh.face_vertices[trial_index])
        info = geometric_adjacency_info(mesh.face_vertices[test_index], trial_vertices; tolerance=tolerance)
        info.kind == :regular && continue
        rule_index = rule_for_singular_orientation!(rules, rule_indices, base_rules, info)
        jac_scale = (T(2) * mesh.areas[test_index]) * (T(2) * mesh.areas[trial_index])
        normal_product = dot(mesh.normals[test_index], reflect_normal(transform, mesh.normals[trial_index]))
        pair = SingularCorrectionPair(test_index, trial_index, rule_index, jac_scale, normal_product)
        push!(pairs_by_test[test_index], pair)
        push!(pairs, pair)
    end
    curls = [surface_curls(mesh.face_vertices[index], mesh.normals[index]) for index in eachindex(mesh.faces)]
    return SingularCorrectionCache(pairs_by_test, pairs, rules, curls, length(pairs))
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
    normalized_mode = normalized_symmetry_mode(symmetry_mode)
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
            symmetry_mode=normalized_mode,
        )
    else
        nothing
    end
    face_vertices, normals, areas, faces, curls = _rocm_geometry_arrays(mesh)
    rule_points, rule_weights = _rocm_rule_arrays(rule)
    vertex_offsets, incident_elements, incident_local_indices, dp0_elements =
        _rocm_incident_element_arrays(p1_space, dp0_space, indices)
    p1_dofs, element_dp0_dofs =
        _rocm_local_dof_arrays(p1_space, dp0_space, indices, length(mesh.faces))
    color_offsets, color_elements = _rocm_element_color_arrays(mesh, indices)
    image_transforms = collect(symmetry_image_transforms(normalized_mode))
    image_singular_caches = RocmSingularCorrectionCache{T}[]
    image_singular_pair_count = 0
    for transform in image_transforms
        host_image_cache = _rocm_image_singular_host_cache(mesh, singular_order, indices, transform)
        push!(image_singular_caches, build_rocm_singular_correction_cache(host_image_cache))
        image_singular_pair_count += host_image_cache.pair_count
    end

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
        AMDGPU.ROCArray(p1_dofs),
        AMDGPU.ROCArray(element_dp0_dofs),
        AMDGPU.ROCArray(color_elements),
        color_offsets,
        indices,
        length(mesh.faces),
        p1_space.global_dof_count,
        dp0_space.global_dof_count,
        length(rule.points),
        normalized_mode,
        image_transforms,
        image_singular_caches,
        image_singular_pair_count,
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
    AMDGPU.unsafe_free!(cache.p1_dofs)
    AMDGPU.unsafe_free!(cache.element_dp0_dofs)
    AMDGPU.unsafe_free!(cache.color_elements)
    for image_cache in cache.image_singular_caches
        release_rocm_singular_correction_cache!(image_cache)
    end
    return nothing
end
