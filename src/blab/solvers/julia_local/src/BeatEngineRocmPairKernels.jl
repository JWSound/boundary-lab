@inline function _rocm_pair_elements(
    color_elements,
    linear_index,
    test_start,
    test_count,
    trial_start,
)
    test_offset = (linear_index - 1) % test_count
    trial_offset = div(linear_index - 1, test_count)
    return color_elements[test_start + test_offset], color_elements[trial_start + trial_offset]
end

@inline function _rocm_pair_is_skipped(
    faces,
    face_count,
    test_index,
    trial_index,
    pair_offsets,
    singular_trial_indices,
    skip_mode,
)
    return skip_mode == 0 ?
        _rocm_faces_are_adjacent(faces, test_index, trial_index, face_count) :
        skip_mode == 1 &&
            _rocm_find_pair(pair_offsets, singular_trial_indices, test_index, trial_index) != 0
end

function _rocm_regular_slp_adjoint_dlp_pairs_kernel!(
    single_layer,
    adjoint_double_layer,
    double_layer,
    face_vertices,
    normals,
    areas,
    faces,
    p1_dofs,
    element_dp0_dofs,
    rule_points,
    rule_weights,
    color_elements,
    test_start,
    test_count,
    trial_start,
    trial_count,
    k,
    p1_dof_count,
    face_count,
    rule_count,
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
)
    linear_index = _rocm_global_linear_index()
    linear_index > test_count * trial_count && return nothing
    test_index, trial_index = _rocm_pair_elements(
        color_elements,
        linear_index,
        test_start,
        test_count,
        trial_start,
    )
    _rocm_pair_is_skipped(
        faces,
        face_count,
        test_index,
        trial_index,
        pair_offsets,
        singular_trial_indices,
        skip_mode,
    ) && return nothing

    four_pi = typeof(k)(12.566370614359172)
    slp_re = zero(SVector{3,typeof(k)})
    slp_im = zero(SVector{3,typeof(k)})
    adj_re = zero(SVector{3,typeof(k)})
    adj_im = zero(SVector{3,typeof(k)})
    dlp_re = zero(SVector{9,typeof(k)})
    dlp_im = zero(SVector{9,typeof(k)})
    test_nx = normals[test_index]
    test_ny = normals[test_index + face_count]
    test_nz = normals[test_index + 2 * face_count]
    trial_nx = trial_sign_x * normals[trial_index]
    trial_ny = trial_sign_y * normals[trial_index + face_count]
    trial_nz = trial_sign_z * normals[trial_index + 2 * face_count]
    jac_scale = typeof(k)(4) * areas[test_index] * areas[trial_index]

    test_q = 1
    while test_q <= rule_count
        test_xi = rule_points[test_q]
        test_eta = rule_points[test_q + rule_count]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        test_basis = SVector(tb1, tb2, tb3)
        x, y, z = _rocm_face_point(face_vertices, test_index, face_count, tb1, tb2, tb3)
        test_weight = rule_weights[test_q]

        trial_q = 1
        while trial_q <= rule_count
            trial_xi = rule_points[trial_q]
            trial_eta = rule_points[trial_q + rule_count]
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
                weight = test_weight * rule_weights[trial_q] * jac_scale
                weighted_basis = test_basis * weight
                slp_re += weighted_basis * green_re
                slp_im += weighted_basis * green_im

                grad_re = -green_re * inv_radius - green_im * k
                grad_im = green_re * k - green_im * inv_radius
                test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
                adj_re += weighted_basis * (grad_re * test_dot)
                adj_im += weighted_basis * (grad_im * test_dot)

                trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
                basis_products = SVector(
                    tb1 * rb1, tb2 * rb1, tb3 * rb1,
                    tb1 * rb2, tb2 * rb2, tb3 * rb2,
                    tb1 * rb3, tb2 * rb3, tb3 * rb3,
                )
                dlp_re += basis_products * (grad_re * trial_dot * weight)
                dlp_im += basis_products * (grad_im * trial_dot * weight)
            end
            trial_q += 1
        end
        test_q += 1
    end

    dp0_column = element_dp0_dofs[trial_index]
    local_row = 1
    while local_row <= 3
        row = p1_dofs[test_index + (local_row - 1) * face_count]
        operator_index = row + (dp0_column - 1) * p1_dof_count
        single_layer[operator_index] += Complex(slp_re[local_row], slp_im[local_row])
        adjoint_double_layer[operator_index] += Complex(adj_re[local_row], adj_im[local_row])
        local_row += 1
    end
    local_column = 1
    while local_column <= 3
        column = p1_dofs[trial_index + (local_column - 1) * face_count]
        local_row = 1
        while local_row <= 3
            row = p1_dofs[test_index + (local_row - 1) * face_count]
            local_index = local_row + 3 * (local_column - 1)
            operator_index = row + (column - 1) * p1_dof_count
            double_layer[operator_index] += Complex(dlp_re[local_index], dlp_im[local_index])
            local_row += 1
        end
        local_column += 1
    end
    return nothing
end

@inline function _rocm_pair_curl_products(
    curls,
    test_index,
    trial_index,
    face_count,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    t11 = curls[test_index]
    t12 = curls[test_index + face_count]
    t13 = curls[test_index + 2 * face_count]
    t21 = curls[test_index + 3 * face_count]
    t22 = curls[test_index + 4 * face_count]
    t23 = curls[test_index + 5 * face_count]
    t31 = curls[test_index + 6 * face_count]
    t32 = curls[test_index + 7 * face_count]
    t33 = curls[test_index + 8 * face_count]
    r11 = trial_curl_sign_x * curls[trial_index]
    r12 = trial_curl_sign_y * curls[trial_index + face_count]
    r13 = trial_curl_sign_z * curls[trial_index + 2 * face_count]
    r21 = trial_curl_sign_x * curls[trial_index + 3 * face_count]
    r22 = trial_curl_sign_y * curls[trial_index + 4 * face_count]
    r23 = trial_curl_sign_z * curls[trial_index + 5 * face_count]
    r31 = trial_curl_sign_x * curls[trial_index + 6 * face_count]
    r32 = trial_curl_sign_y * curls[trial_index + 7 * face_count]
    r33 = trial_curl_sign_z * curls[trial_index + 8 * face_count]
    return SVector(
        t11 * r11 + t12 * r12 + t13 * r13,
        t21 * r11 + t22 * r12 + t23 * r13,
        t31 * r11 + t32 * r12 + t33 * r13,
        t11 * r21 + t12 * r22 + t13 * r23,
        t21 * r21 + t22 * r22 + t23 * r23,
        t31 * r21 + t32 * r22 + t33 * r23,
        t11 * r31 + t12 * r32 + t13 * r33,
        t21 * r31 + t22 * r32 + t23 * r33,
        t31 * r31 + t32 * r32 + t33 * r33,
    )
end

function _rocm_regular_hyp_pairs_kernel!(
    hypersingular,
    face_vertices,
    normals,
    areas,
    faces,
    curls,
    p1_dofs,
    rule_points,
    rule_weights,
    color_elements,
    test_start,
    test_count,
    trial_start,
    trial_count,
    k,
    p1_dof_count,
    face_count,
    rule_count,
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    linear_index = _rocm_global_linear_index()
    linear_index > test_count * trial_count && return nothing
    test_index, trial_index = _rocm_pair_elements(
        color_elements,
        linear_index,
        test_start,
        test_count,
        trial_start,
    )
    _rocm_pair_is_skipped(
        faces,
        face_count,
        test_index,
        trial_index,
        pair_offsets,
        singular_trial_indices,
        skip_mode,
    ) && return nothing

    four_pi = typeof(k)(12.566370614359172)
    k2 = k * k
    hyp_re = zero(SVector{9,typeof(k)})
    hyp_im = zero(SVector{9,typeof(k)})
    trial_nx = trial_sign_x * normals[trial_index]
    trial_ny = trial_sign_y * normals[trial_index + face_count]
    trial_nz = trial_sign_z * normals[trial_index + 2 * face_count]
    normal_product = normals[test_index] * trial_nx +
        normals[test_index + face_count] * trial_ny +
        normals[test_index + 2 * face_count] * trial_nz
    curl_products = _rocm_pair_curl_products(
        curls,
        test_index,
        trial_index,
        face_count,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
    )
    jac_scale = typeof(k)(4) * areas[test_index] * areas[trial_index]

    test_q = 1
    while test_q <= rule_count
        test_xi = rule_points[test_q]
        test_eta = rule_points[test_q + rule_count]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        x, y, z = _rocm_face_point(face_vertices, test_index, face_count, tb1, tb2, tb3)
        test_weight = rule_weights[test_q]

        trial_q = 1
        while trial_q <= rule_count
            trial_xi = rule_points[trial_q]
            trial_eta = rule_points[trial_q + rule_count]
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
                weight = test_weight * rule_weights[trial_q] * jac_scale
                basis_products = SVector(
                    tb1 * rb1, tb2 * rb1, tb3 * rb1,
                    tb1 * rb2, tb2 * rb2, tb3 * rb2,
                    tb1 * rb3, tb2 * rb3, tb3 * rb3,
                )
                hyper_factors = curl_products - basis_products * (k2 * normal_product)
                hyp_re += hyper_factors * (green_re * weight)
                hyp_im += hyper_factors * (green_im * weight)
            end
            trial_q += 1
        end
        test_q += 1
    end

    local_column = 1
    while local_column <= 3
        column = p1_dofs[trial_index + (local_column - 1) * face_count]
        local_row = 1
        while local_row <= 3
            row = p1_dofs[test_index + (local_row - 1) * face_count]
            local_index = local_row + 3 * (local_column - 1)
            operator_index = row + (column - 1) * p1_dof_count
            hypersingular[operator_index] += Complex(hyp_re[local_index], hyp_im[local_index])
            local_row += 1
        end
        local_column += 1
    end
    return nothing
end

function _launch_rocm_colored_pair_kernels!(
    operators,
    cache::RocmRegularAssemblyCache,
    k,
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    groupsize = _rocm_kernel_groupsize()
    color_count = length(cache.color_offsets) - 1
    for test_color in 1:color_count
        test_start = cache.color_offsets[test_color]
        test_count = cache.color_offsets[test_color + 1] - test_start
        for trial_color in 1:color_count
            trial_start = cache.color_offsets[trial_color]
            trial_count = cache.color_offsets[trial_color + 1] - trial_start
            pair_count = test_count * trial_count
            pair_count == 0 && continue
            AMDGPU.@roc groupsize=groupsize gridsize=cld(pair_count, groupsize) _rocm_regular_slp_adjoint_dlp_pairs_kernel!(
                operators.single_layer,
                operators.adjoint_double_layer,
                operators.double_layer,
                cache.face_vertices,
                cache.normals,
                cache.areas,
                cache.faces,
                cache.p1_dofs,
                cache.element_dp0_dofs,
                cache.rule_points,
                cache.rule_weights,
                cache.color_elements,
                Int32(test_start),
                Int32(test_count),
                Int32(trial_start),
                Int32(trial_count),
                k,
                cache.p1_dof_count,
                cache.face_count,
                cache.rule_count,
                pair_offsets,
                singular_trial_indices,
                skip_mode,
                trial_sign_x,
                trial_sign_y,
                trial_sign_z,
            )
            AMDGPU.@roc groupsize=groupsize gridsize=cld(pair_count, groupsize) _rocm_regular_hyp_pairs_kernel!(
                operators.hypersingular,
                cache.face_vertices,
                cache.normals,
                cache.areas,
                cache.faces,
                cache.curls,
                cache.p1_dofs,
                cache.rule_points,
                cache.rule_weights,
                cache.color_elements,
                Int32(test_start),
                Int32(test_count),
                Int32(trial_start),
                Int32(trial_count),
                k,
                cache.p1_dof_count,
                cache.face_count,
                cache.rule_count,
                pair_offsets,
                singular_trial_indices,
                skip_mode,
                trial_sign_x,
                trial_sign_y,
                trial_sign_z,
                trial_curl_sign_x,
                trial_curl_sign_y,
                trial_curl_sign_z,
            )
        end
    end
    return nothing
end

function _launch_rocm_regular_pair_kernels!(operators, cache::RocmRegularAssemblyCache, k)
    return _launch_rocm_colored_pair_kernels!(
        operators,
        cache,
        k,
        cache.vertex_offsets,
        cache.incident_elements,
        Int32(0),
        one(k), one(k), one(k),
        one(k), one(k), one(k),
    )
end

function _launch_rocm_symmetry_regular_pair_kernels!(
    operators,
    cache::RocmRegularAssemblyCache,
    image_cache::RocmSingularCorrectionCache,
    transform::SymmetryTransform,
    k;
    skip_image_singular::Bool,
)
    sx = typeof(k)(transform.signs[1])
    sy = typeof(k)(transform.signs[2])
    sz = typeof(k)(transform.signs[3])
    csx = typeof(k)(transform.determinant * transform.signs[1])
    csy = typeof(k)(transform.determinant * transform.signs[2])
    csz = typeof(k)(transform.determinant * transform.signs[3])
    return _launch_rocm_colored_pair_kernels!(
        operators,
        cache,
        k,
        image_cache.pair_offsets,
        image_cache.trial_indices,
        skip_image_singular ? Int32(1) : Int32(2),
        sx, sy, sz,
        csx, csy, csz,
    )
end
