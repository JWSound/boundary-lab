function build_rocm_singular_correction_cache(cache::SingularCorrectionCache{T}) where {T<:AbstractFloat}
    _require_rocm!()
    face_count = length(cache.pairs_by_test)
    pair_offsets = Vector{Int32}(undef, face_count + 1)
    trial_indices = Int32[]
    rule_indices = Int32[]
    jac_scales = T[]
    normal_products = T[]
    pair_offsets[1] = 1
    for test_index in 1:face_count
        for pair in cache.pairs_by_test[test_index]
            push!(trial_indices, Int32(pair.trial_index))
            push!(rule_indices, Int32(pair.rule_index))
            push!(jac_scales, pair.jac_scale)
            push!(normal_products, pair.normal_product)
        end
        pair_offsets[test_index + 1] = Int32(length(trial_indices) + 1)
    end

    total_rule_points = sum(length(rule.weights) for rule in cache.rules)
    rule_offsets = Vector{Int32}(undef, length(cache.rules) + 1)
    rule_test_points = Matrix{T}(undef, total_rule_points, 2)
    rule_trial_points = Matrix{T}(undef, total_rule_points, 2)
    rule_weights = Vector{T}(undef, total_rule_points)
    offset = 1
    for (rule_index, rule) in enumerate(cache.rules)
        rule_offsets[rule_index] = Int32(offset)
        for q in eachindex(rule.weights)
            target = offset + q - 1
            rule_test_points[target, 1] = rule.test_points[q][1]
            rule_test_points[target, 2] = rule.test_points[q][2]
            rule_trial_points[target, 1] = rule.trial_points[q][1]
            rule_trial_points[target, 2] = rule.trial_points[q][2]
            rule_weights[target] = rule.weights[q]
        end
        offset += length(rule.weights)
    end
    rule_offsets[end] = Int32(offset)

    return RocmSingularCorrectionCache{T}(
        AMDGPU.ROCArray(pair_offsets),
        AMDGPU.ROCArray(trial_indices),
        AMDGPU.ROCArray(rule_indices),
        AMDGPU.ROCArray(jac_scales),
        AMDGPU.ROCArray(normal_products),
        AMDGPU.ROCArray(rule_offsets),
        AMDGPU.ROCArray(rule_test_points),
        AMDGPU.ROCArray(rule_trial_points),
        AMDGPU.ROCArray(rule_weights),
        cache.pair_count,
    )
end

function _release_rocm_singular_correction_cache!(cache::RocmSingularCorrectionCache)
    AMDGPU.unsafe_free!(cache.pair_offsets)
    AMDGPU.unsafe_free!(cache.trial_indices)
    AMDGPU.unsafe_free!(cache.rule_indices)
    AMDGPU.unsafe_free!(cache.jac_scales)
    AMDGPU.unsafe_free!(cache.normal_products)
    AMDGPU.unsafe_free!(cache.rule_offsets)
    AMDGPU.unsafe_free!(cache.rule_test_points)
    AMDGPU.unsafe_free!(cache.rule_trial_points)
    AMDGPU.unsafe_free!(cache.rule_weights)
    return nothing
end

release_rocm_singular_correction_cache!(cache::RocmSingularCorrectionCache) =
    _release_rocm_singular_correction_cache!(cache)

@inline function _rocm_find_singular_pair(pair_offsets, trial_indices, test_index, trial_index)
    pair_position = pair_offsets[test_index]
    pair_stop = pair_offsets[test_index + 1] - 1
    while pair_position <= pair_stop
        trial_indices[pair_position] == trial_index && return pair_position
        pair_position += 1
    end
    return zero(pair_position)
end

function _rocm_singular_slp_adjoint_entries_kernel!(
    single_layer,
    adjoint_double_layer,
    face_vertices,
    normals,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    dp0_elements,
    pair_offsets,
    trial_indices,
    rule_indices,
    jac_scales,
    rule_offsets,
    rule_test_points,
    rule_trial_points,
    rule_weights,
    k,
    p1_dof_count,
    dp0_dof_count,
    face_count,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
)
    linear_index = _rocm_global_linear_index()
    total_entries = p1_dof_count * dp0_dof_count
    linear_index > total_entries && return nothing
    row = ((linear_index - 1) % p1_dof_count) + 1
    column = div(linear_index - 1, p1_dof_count) + 1
    trial_index = dp0_elements[column]
    trial_index == 0 && return nothing

    four_pi = typeof(k)(12.566370614359172)
    rule_point_count = length(rule_weights)
    slp_re = zero(k)
    slp_im = zero(k)
    adj_re = zero(k)
    adj_im = zero(k)
    incident_position = vertex_offsets[row]
    incident_stop = vertex_offsets[row + 1] - 1
    while incident_position <= incident_stop
        test_index = incident_elements[incident_position]
        local_row = incident_local_indices[incident_position]
        pair_position = _rocm_find_singular_pair(pair_offsets, trial_indices, test_index, trial_index)
        if pair_position != 0
            rule_index = rule_indices[pair_position]
            q = rule_offsets[rule_index]
            q_stop = rule_offsets[rule_index + 1] - 1
            jac_scale = jac_scales[pair_position]
            test_nx = normals[test_index]
            test_ny = normals[test_index + face_count]
            test_nz = normals[test_index + 2 * face_count]
            while q <= q_stop
                test_xi = rule_test_points[q]
                test_eta = rule_test_points[q + rule_point_count]
                test_basis1 = one(k) - test_xi - test_eta
                test_basis2 = test_xi
                test_basis3 = test_eta
                test_basis = _rocm_basis_value(local_row, test_basis1, test_basis2, test_basis3)
                x, y, z = _rocm_face_point(
                    face_vertices,
                    test_index,
                    face_count,
                    test_basis1,
                    test_basis2,
                    test_basis3,
                )
                trial_xi = rule_trial_points[q]
                trial_eta = rule_trial_points[q + rule_point_count]
                trial_basis1 = one(k) - trial_xi - trial_eta
                trial_basis2 = trial_xi
                trial_basis3 = trial_eta
                sx, sy, sz = _rocm_face_point(
                    face_vertices,
                    trial_index,
                    face_count,
                    trial_basis1,
                    trial_basis2,
                    trial_basis3,
                )
                sx *= trial_sign_x
                sy *= trial_sign_y
                sz *= trial_sign_z

                dx = sx - x
                dy = sy - y
                dz = sz - z
                radius = sqrt(dx * dx + dy * dy + dz * dz)
                if radius > zero(k)
                    inv_radius = one(k) / radius
                    phase = k * radius
                    green_scale = inv_radius / four_pi
                    green_re = cos(phase) * green_scale
                    green_im = sin(phase) * green_scale
                    weight = rule_weights[q] * jac_scale * test_basis
                    slp_re += green_re * weight
                    slp_im += green_im * weight
                    test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
                    grad_re = -green_re * inv_radius - green_im * k
                    grad_im = green_re * k - green_im * inv_radius
                    adj_re += grad_re * test_dot * weight
                    adj_im += grad_im * test_dot * weight
                end
                q += 1
            end
        end
        incident_position += 1
    end

    single_layer[linear_index] += Complex(slp_re, slp_im)
    adjoint_double_layer[linear_index] += Complex(adj_re, adj_im)
    return nothing
end

function _rocm_singular_dlp_hyp_entries_kernel!(
    double_layer,
    hypersingular,
    face_vertices,
    normals,
    curls,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    pair_offsets,
    trial_indices,
    rule_indices,
    jac_scales,
    normal_products,
    rule_offsets,
    rule_test_points,
    rule_trial_points,
    rule_weights,
    k,
    p1_dof_count,
    face_count,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    linear_index = _rocm_global_linear_index()
    total_entries = p1_dof_count * p1_dof_count
    linear_index > total_entries && return nothing
    row = ((linear_index - 1) % p1_dof_count) + 1
    column = div(linear_index - 1, p1_dof_count) + 1
    four_pi = typeof(k)(12.566370614359172)
    rule_point_count = length(rule_weights)
    k2 = k * k
    dlp_re = zero(k)
    dlp_im = zero(k)
    hyp_re = zero(k)
    hyp_im = zero(k)

    test_position = vertex_offsets[row]
    test_stop = vertex_offsets[row + 1] - 1
    while test_position <= test_stop
        test_index = incident_elements[test_position]
        local_row = incident_local_indices[test_position]
        test_curl_offset = 3 * (local_row - 1)
        test_curl_x = curls[test_index + test_curl_offset * face_count]
        test_curl_y = curls[test_index + (test_curl_offset + 1) * face_count]
        test_curl_z = curls[test_index + (test_curl_offset + 2) * face_count]
        trial_position = vertex_offsets[column]
        trial_stop = vertex_offsets[column + 1] - 1
        while trial_position <= trial_stop
            trial_index = incident_elements[trial_position]
            local_column = incident_local_indices[trial_position]
            pair_position = _rocm_find_singular_pair(pair_offsets, trial_indices, test_index, trial_index)
            if pair_position != 0
                rule_index = rule_indices[pair_position]
                q = rule_offsets[rule_index]
                q_stop = rule_offsets[rule_index + 1] - 1
                jac_scale = jac_scales[pair_position]
                normal_product = normal_products[pair_position]
                trial_nx = trial_sign_x * normals[trial_index]
                trial_ny = trial_sign_y * normals[trial_index + face_count]
                trial_nz = trial_sign_z * normals[trial_index + 2 * face_count]
                trial_curl_offset = 3 * (local_column - 1)
                curl_product =
                    test_curl_x * trial_curl_sign_x * curls[trial_index + trial_curl_offset * face_count] +
                    test_curl_y * trial_curl_sign_y * curls[trial_index + (trial_curl_offset + 1) * face_count] +
                    test_curl_z * trial_curl_sign_z * curls[trial_index + (trial_curl_offset + 2) * face_count]

                while q <= q_stop
                    test_xi = rule_test_points[q]
                    test_eta = rule_test_points[q + rule_point_count]
                    test_basis1 = one(k) - test_xi - test_eta
                    test_basis2 = test_xi
                    test_basis3 = test_eta
                    test_basis = _rocm_basis_value(local_row, test_basis1, test_basis2, test_basis3)
                    x, y, z = _rocm_face_point(
                        face_vertices,
                        test_index,
                        face_count,
                        test_basis1,
                        test_basis2,
                        test_basis3,
                    )
                    trial_xi = rule_trial_points[q]
                    trial_eta = rule_trial_points[q + rule_point_count]
                    trial_basis1 = one(k) - trial_xi - trial_eta
                    trial_basis2 = trial_xi
                    trial_basis3 = trial_eta
                    trial_basis = _rocm_basis_value(
                        local_column,
                        trial_basis1,
                        trial_basis2,
                        trial_basis3,
                    )
                    sx, sy, sz = _rocm_face_point(
                        face_vertices,
                        trial_index,
                        face_count,
                        trial_basis1,
                        trial_basis2,
                        trial_basis3,
                    )
                    sx *= trial_sign_x
                    sy *= trial_sign_y
                    sz *= trial_sign_z

                    dx = sx - x
                    dy = sy - y
                    dz = sz - z
                    radius = sqrt(dx * dx + dy * dy + dz * dz)
                    if radius > zero(k)
                        inv_radius = one(k) / radius
                        phase = k * radius
                        green_scale = inv_radius / four_pi
                        green_re = cos(phase) * green_scale
                        green_im = sin(phase) * green_scale
                        weight = rule_weights[q] * jac_scale
                        basis_product = test_basis * trial_basis
                        trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
                        grad_re = -green_re * inv_radius - green_im * k
                        grad_im = green_re * k - green_im * inv_radius
                        dlp_weight = basis_product * trial_dot * weight
                        dlp_re += grad_re * dlp_weight
                        dlp_im += grad_im * dlp_weight
                        hyper_factor = curl_product - k2 * basis_product * normal_product
                        hyp_re += hyper_factor * green_re * weight
                        hyp_im += hyper_factor * green_im * weight
                    end
                    q += 1
                end
            end
            trial_position += 1
        end
        test_position += 1
    end

    double_layer[linear_index] += Complex(dlp_re, dlp_im)
    hypersingular[linear_index] += Complex(hyp_re, hyp_im)
    return nothing
end

function _launch_rocm_singular_entry_kernels!(
    operators,
    regular_cache::RocmRegularAssemblyCache,
    singular_cache::RocmSingularCorrectionCache,
    k,
    transform::SymmetryTransform=SymmetryTransform(:identity, SVector{3,Int}(1, 1, 1), 1),
)
    groupsize = 128
    slp_entries = regular_cache.p1_dof_count * regular_cache.dp0_dof_count
    p1_entries = regular_cache.p1_dof_count * regular_cache.p1_dof_count
    sx = typeof(k)(transform.signs[1])
    sy = typeof(k)(transform.signs[2])
    sz = typeof(k)(transform.signs[3])
    csx = typeof(k)(transform.determinant * transform.signs[1])
    csy = typeof(k)(transform.determinant * transform.signs[2])
    csz = typeof(k)(transform.determinant * transform.signs[3])
    AMDGPU.@roc groupsize=groupsize gridsize=cld(slp_entries, groupsize) _rocm_singular_slp_adjoint_entries_kernel!(
        operators.single_layer,
        operators.adjoint_double_layer,
        regular_cache.face_vertices,
        regular_cache.normals,
        regular_cache.vertex_offsets,
        regular_cache.incident_elements,
        regular_cache.incident_local_indices,
        regular_cache.dp0_elements,
        singular_cache.pair_offsets,
        singular_cache.trial_indices,
        singular_cache.rule_indices,
        singular_cache.jac_scales,
        singular_cache.rule_offsets,
        singular_cache.rule_test_points,
        singular_cache.rule_trial_points,
        singular_cache.rule_weights,
        k,
        regular_cache.p1_dof_count,
        regular_cache.dp0_dof_count,
        regular_cache.face_count,
        sx,
        sy,
        sz,
    )
    AMDGPU.@roc groupsize=groupsize gridsize=cld(p1_entries, groupsize) _rocm_singular_dlp_hyp_entries_kernel!(
        operators.double_layer,
        operators.hypersingular,
        regular_cache.face_vertices,
        regular_cache.normals,
        regular_cache.curls,
        regular_cache.vertex_offsets,
        regular_cache.incident_elements,
        regular_cache.incident_local_indices,
        singular_cache.pair_offsets,
        singular_cache.trial_indices,
        singular_cache.rule_indices,
        singular_cache.jac_scales,
        singular_cache.normal_products,
        singular_cache.rule_offsets,
        singular_cache.rule_test_points,
        singular_cache.rule_trial_points,
        singular_cache.rule_weights,
        k,
        regular_cache.p1_dof_count,
        regular_cache.face_count,
        sx,
        sy,
        sz,
        csx,
        csy,
        csz,
    )
    return nothing
end
