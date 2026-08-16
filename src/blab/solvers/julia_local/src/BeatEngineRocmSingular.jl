function build_rocm_singular_correction_cache(cache::SingularCorrectionCache{T}) where {T<:AbstractFloat}
    _require_rocm!()
    face_count = length(cache.pairs_by_test)
    pair_offsets = Vector{Int32}(undef, face_count + 1)
    test_indices = Int32[]
    trial_indices = Int32[]
    rule_indices = Int32[]
    jac_scales = T[]
    normal_products = T[]
    pair_offsets[1] = 1
    for test_index in 1:face_count
        for pair in cache.pairs_by_test[test_index]
            push!(test_indices, Int32(test_index))
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
        AMDGPU.ROCArray(test_indices),
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
    AMDGPU.unsafe_free!(cache.test_indices)
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

function _rocm_singular_slp_adjoint_blocks_kernel!(
    slp_values,
    adjoint_values,
    test_indices,
    trial_indices,
    rule_indices,
    jac_scales,
    rule_offsets,
    rule_test_points,
    rule_trial_points,
    rule_weights,
    face_vertices,
    normals,
    k,
    face_count,
    pair_count,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
)
    pair_position = _rocm_global_linear_index()
    pair_position > pair_count && return nothing
    test_index = test_indices[pair_position]
    trial_index = trial_indices[pair_position]
    rule_index = rule_indices[pair_position]
    q = rule_offsets[rule_index]
    q_stop = rule_offsets[rule_index + 1] - 1
    rule_point_count = length(rule_weights)
    jac_scale = jac_scales[pair_position]
    four_pi = typeof(k)(12.566370614359172)
    test_nx = normals[test_index]
    test_ny = normals[test_index + face_count]
    test_nz = normals[test_index + 2 * face_count]
    slp1_re = zero(k); slp1_im = zero(k)
    slp2_re = zero(k); slp2_im = zero(k)
    slp3_re = zero(k); slp3_im = zero(k)
    adj1_re = zero(k); adj1_im = zero(k)
    adj2_re = zero(k); adj2_im = zero(k)
    adj3_re = zero(k); adj3_im = zero(k)
    while q <= q_stop
        test_xi = rule_test_points[q]
        test_eta = rule_test_points[q + rule_point_count]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        x, y, z = _rocm_face_point(face_vertices, test_index, face_count, tb1, tb2, tb3)
        trial_xi = rule_trial_points[q]
        trial_eta = rule_trial_points[q + rule_point_count]
        rb1 = one(k) - trial_xi - trial_eta
        rb2 = trial_xi
        rb3 = trial_eta
        sx, sy, sz = _rocm_face_point(face_vertices, trial_index, face_count, rb1, rb2, rb3)
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
            test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
            grad_re = (-green_re * inv_radius - green_im * k) * test_dot
            grad_im = (green_re * k - green_im * inv_radius) * test_dot
            slp1_re += tb1 * green_re * weight; slp1_im += tb1 * green_im * weight
            slp2_re += tb2 * green_re * weight; slp2_im += tb2 * green_im * weight
            slp3_re += tb3 * green_re * weight; slp3_im += tb3 * green_im * weight
            adj1_re += tb1 * grad_re * weight; adj1_im += tb1 * grad_im * weight
            adj2_re += tb2 * grad_re * weight; adj2_im += tb2 * grad_im * weight
            adj3_re += tb3 * grad_re * weight; adj3_im += tb3 * grad_im * weight
        end
        q += 1
    end
    slp_values[pair_position] = Complex(slp1_re, slp1_im)
    slp_values[pair_position + pair_count] = Complex(slp2_re, slp2_im)
    slp_values[pair_position + 2 * pair_count] = Complex(slp3_re, slp3_im)
    adjoint_values[pair_position] = Complex(adj1_re, adj1_im)
    adjoint_values[pair_position + pair_count] = Complex(adj2_re, adj2_im)
    adjoint_values[pair_position + 2 * pair_count] = Complex(adj3_re, adj3_im)
    return nothing
end

function _rocm_singular_dlp_hyp_blocks_kernel!(
    dlp_values,
    hypersingular_values,
    test_indices,
    trial_indices,
    rule_indices,
    jac_scales,
    normal_products,
    rule_offsets,
    rule_test_points,
    rule_trial_points,
    rule_weights,
    face_vertices,
    normals,
    curls,
    k,
    face_count,
    pair_count,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    linear_index = _rocm_global_linear_index()
    linear_index > 3 * pair_count && return nothing
    pair_position = ((linear_index - 1) % pair_count) + 1
    local_column = div(linear_index - 1, pair_count) + 1
    test_index = test_indices[pair_position]
    trial_index = trial_indices[pair_position]
    rule_index = rule_indices[pair_position]
    q = rule_offsets[rule_index]
    q_stop = rule_offsets[rule_index + 1] - 1
    rule_point_count = length(rule_weights)
    jac_scale = jac_scales[pair_position]
    normal_product = normal_products[pair_position]
    four_pi = typeof(k)(12.566370614359172)
    k2 = k * k
    trial_nx = trial_sign_x * normals[trial_index]
    trial_ny = trial_sign_y * normals[trial_index + face_count]
    trial_nz = trial_sign_z * normals[trial_index + 2 * face_count]
    trial_curl_offset = 3 * (local_column - 1)
    trial_curl_x = trial_curl_sign_x * curls[trial_index + trial_curl_offset * face_count]
    trial_curl_y = trial_curl_sign_y * curls[trial_index + (trial_curl_offset + 1) * face_count]
    trial_curl_z = trial_curl_sign_z * curls[trial_index + (trial_curl_offset + 2) * face_count]
    test_curl1_x = curls[test_index]
    test_curl1_y = curls[test_index + face_count]
    test_curl1_z = curls[test_index + 2 * face_count]
    test_curl2_x = curls[test_index + 3 * face_count]
    test_curl2_y = curls[test_index + 4 * face_count]
    test_curl2_z = curls[test_index + 5 * face_count]
    test_curl3_x = curls[test_index + 6 * face_count]
    test_curl3_y = curls[test_index + 7 * face_count]
    test_curl3_z = curls[test_index + 8 * face_count]
    curl1 = test_curl1_x * trial_curl_x + test_curl1_y * trial_curl_y + test_curl1_z * trial_curl_z
    curl2 = test_curl2_x * trial_curl_x + test_curl2_y * trial_curl_y + test_curl2_z * trial_curl_z
    curl3 = test_curl3_x * trial_curl_x + test_curl3_y * trial_curl_y + test_curl3_z * trial_curl_z
    dlp1_re = zero(k); dlp1_im = zero(k)
    dlp2_re = zero(k); dlp2_im = zero(k)
    dlp3_re = zero(k); dlp3_im = zero(k)
    hyp1_re = zero(k); hyp1_im = zero(k)
    hyp2_re = zero(k); hyp2_im = zero(k)
    hyp3_re = zero(k); hyp3_im = zero(k)
    while q <= q_stop
        test_xi = rule_test_points[q]
        test_eta = rule_test_points[q + rule_point_count]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        x, y, z = _rocm_face_point(face_vertices, test_index, face_count, tb1, tb2, tb3)
        trial_xi = rule_trial_points[q]
        trial_eta = rule_trial_points[q + rule_point_count]
        rb1 = one(k) - trial_xi - trial_eta
        rb2 = trial_xi
        rb3 = trial_eta
        trial_basis = _rocm_basis_value(local_column, rb1, rb2, rb3)
        sx, sy, sz = _rocm_face_point(face_vertices, trial_index, face_count, rb1, rb2, rb3)
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
            trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
            grad_re = (-green_re * inv_radius - green_im * k) * trial_dot
            grad_im = (green_re * k - green_im * inv_radius) * trial_dot
            basis1 = tb1 * trial_basis
            basis2 = tb2 * trial_basis
            basis3 = tb3 * trial_basis
            dlp1_re += basis1 * grad_re * weight; dlp1_im += basis1 * grad_im * weight
            dlp2_re += basis2 * grad_re * weight; dlp2_im += basis2 * grad_im * weight
            dlp3_re += basis3 * grad_re * weight; dlp3_im += basis3 * grad_im * weight
            factor1 = curl1 - k2 * basis1 * normal_product
            factor2 = curl2 - k2 * basis2 * normal_product
            factor3 = curl3 - k2 * basis3 * normal_product
            hyp1_re += factor1 * green_re * weight; hyp1_im += factor1 * green_im * weight
            hyp2_re += factor2 * green_re * weight; hyp2_im += factor2 * green_im * weight
            hyp3_re += factor3 * green_re * weight; hyp3_im += factor3 * green_im * weight
        end
        q += 1
    end
    value_offset = (local_column - 1) * 3
    dlp_values[pair_position + value_offset * pair_count] = Complex(dlp1_re, dlp1_im)
    dlp_values[pair_position + (value_offset + 1) * pair_count] = Complex(dlp2_re, dlp2_im)
    dlp_values[pair_position + (value_offset + 2) * pair_count] = Complex(dlp3_re, dlp3_im)
    hypersingular_values[pair_position + value_offset * pair_count] = Complex(hyp1_re, hyp1_im)
    hypersingular_values[pair_position + (value_offset + 1) * pair_count] = Complex(hyp2_re, hyp2_im)
    hypersingular_values[pair_position + (value_offset + 2) * pair_count] = Complex(hyp3_re, hyp3_im)
    return nothing
end

function _rocm_singular_slp_adjoint_gather_kernel!(
    single_layer,
    adjoint_double_layer,
    slp_values,
    adjoint_values,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    dp0_elements,
    pair_offsets,
    trial_indices,
    pair_count,
    p1_dof_count,
    dp0_dof_count,
)
    linear_index = _rocm_global_linear_index()
    linear_index > p1_dof_count * dp0_dof_count && return nothing
    row = ((linear_index - 1) % p1_dof_count) + 1
    column = div(linear_index - 1, p1_dof_count) + 1
    trial_index = dp0_elements[column]
    trial_index == 0 && return nothing
    slp = zero(eltype(single_layer))
    adj = zero(eltype(adjoint_double_layer))
    position = vertex_offsets[row]
    position_stop = vertex_offsets[row + 1] - 1
    while position <= position_stop
        test_index = incident_elements[position]
        local_row = incident_local_indices[position]
        pair_position = _rocm_find_singular_pair(pair_offsets, trial_indices, test_index, trial_index)
        if pair_position != 0
            value_index = pair_position + (local_row - 1) * pair_count
            slp += slp_values[value_index]
            adj += adjoint_values[value_index]
        end
        position += 1
    end
    single_layer[linear_index] += slp
    adjoint_double_layer[linear_index] += adj
    return nothing
end

function _rocm_singular_dlp_hyp_gather_kernel!(
    double_layer,
    hypersingular,
    dlp_values,
    hypersingular_values,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    pair_offsets,
    trial_indices,
    pair_count,
    p1_dof_count,
)
    linear_index = _rocm_global_linear_index()
    linear_index > p1_dof_count * p1_dof_count && return nothing
    row = ((linear_index - 1) % p1_dof_count) + 1
    column = div(linear_index - 1, p1_dof_count) + 1
    dlp = zero(eltype(double_layer))
    hyp = zero(eltype(hypersingular))
    test_position = vertex_offsets[row]
    test_stop = vertex_offsets[row + 1] - 1
    while test_position <= test_stop
        test_index = incident_elements[test_position]
        local_row = incident_local_indices[test_position]
        trial_position = vertex_offsets[column]
        trial_stop = vertex_offsets[column + 1] - 1
        while trial_position <= trial_stop
            trial_index = incident_elements[trial_position]
            local_column = incident_local_indices[trial_position]
            pair_position = _rocm_find_singular_pair(pair_offsets, trial_indices, test_index, trial_index)
            if pair_position != 0
                value_index = pair_position + ((local_column - 1) * 3 + local_row - 1) * pair_count
                dlp += dlp_values[value_index]
                hyp += hypersingular_values[value_index]
            end
            trial_position += 1
        end
        test_position += 1
    end
    double_layer[linear_index] += dlp
    hypersingular[linear_index] += hyp
    return nothing
end

function _launch_rocm_singular_block_gather_kernels!(
    operators,
    regular_cache::RocmRegularAssemblyCache,
    singular_cache::RocmSingularCorrectionCache,
    k,
    transform::SymmetryTransform=SymmetryTransform(:identity, SVector{3,Int}(1, 1, 1), 1),
)
    groupsize = _rocm_kernel_groupsize()
    slp_entries = regular_cache.p1_dof_count * regular_cache.dp0_dof_count
    p1_entries = regular_cache.p1_dof_count * regular_cache.p1_dof_count
    sx = typeof(k)(transform.signs[1])
    sy = typeof(k)(transform.signs[2])
    sz = typeof(k)(transform.signs[3])
    csx = typeof(k)(transform.determinant * transform.signs[1])
    csy = typeof(k)(transform.determinant * transform.signs[2])
    csz = typeof(k)(transform.determinant * transform.signs[3])
    pair_count = singular_cache.pair_count
    pair_count == 0 && return nothing
    slp_values = AMDGPU.zeros(eltype(operators.single_layer), pair_count, 3)
    adjoint_values = AMDGPU.zeros(eltype(operators.adjoint_double_layer), pair_count, 3)
    dlp_values = AMDGPU.zeros(eltype(operators.double_layer), pair_count, 9)
    hypersingular_values = AMDGPU.zeros(eltype(operators.hypersingular), pair_count, 9)
    AMDGPU.@roc groupsize=groupsize gridsize=cld(pair_count, groupsize) _rocm_singular_slp_adjoint_blocks_kernel!(
        slp_values,
        adjoint_values,
        singular_cache.test_indices,
        singular_cache.trial_indices,
        singular_cache.rule_indices,
        singular_cache.jac_scales,
        singular_cache.rule_offsets,
        singular_cache.rule_test_points,
        singular_cache.rule_trial_points,
        singular_cache.rule_weights,
        regular_cache.face_vertices,
        regular_cache.normals,
        k,
        regular_cache.face_count,
        pair_count,
        sx,
        sy,
        sz,
    )
    AMDGPU.@roc groupsize=groupsize gridsize=cld(3 * pair_count, groupsize) _rocm_singular_dlp_hyp_blocks_kernel!(
        dlp_values,
        hypersingular_values,
        singular_cache.test_indices,
        singular_cache.trial_indices,
        singular_cache.rule_indices,
        singular_cache.jac_scales,
        singular_cache.normal_products,
        singular_cache.rule_offsets,
        singular_cache.rule_test_points,
        singular_cache.rule_trial_points,
        singular_cache.rule_weights,
        regular_cache.face_vertices,
        regular_cache.normals,
        regular_cache.curls,
        k,
        regular_cache.face_count,
        pair_count,
        sx,
        sy,
        sz,
        csx,
        csy,
        csz,
    )
    AMDGPU.@roc groupsize=groupsize gridsize=cld(slp_entries, groupsize) _rocm_singular_slp_adjoint_gather_kernel!(
        operators.single_layer,
        operators.adjoint_double_layer,
        slp_values,
        adjoint_values,
        regular_cache.vertex_offsets,
        regular_cache.incident_elements,
        regular_cache.incident_local_indices,
        regular_cache.dp0_elements,
        singular_cache.pair_offsets,
        singular_cache.trial_indices,
        pair_count,
        regular_cache.p1_dof_count,
        regular_cache.dp0_dof_count,
    )
    AMDGPU.@roc groupsize=groupsize gridsize=cld(p1_entries, groupsize) _rocm_singular_dlp_hyp_gather_kernel!(
        operators.double_layer,
        operators.hypersingular,
        dlp_values,
        hypersingular_values,
        regular_cache.vertex_offsets,
        regular_cache.incident_elements,
        regular_cache.incident_local_indices,
        singular_cache.pair_offsets,
        singular_cache.trial_indices,
        pair_count,
        regular_cache.p1_dof_count,
    )
    AMDGPU.synchronize()
    AMDGPU.unsafe_free!(slp_values)
    AMDGPU.unsafe_free!(adjoint_values)
    AMDGPU.unsafe_free!(dlp_values)
    AMDGPU.unsafe_free!(hypersingular_values)
    return nothing
end
