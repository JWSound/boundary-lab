function _cuda_regular_quadrature_slp_hyp_kernel!(
    slp_re,
    slp_im,
    hyp_re,
    hyp_im,
    face_vertices,
    normals,
    areas,
    faces,
    curls,
    test_indices,
    trial_indices,
    rule_points,
    rule_weights,
    k,
    p1_dof_count,
    face_count,
    rule_count,
    total_pairs,
    pairs_per_block,
)
    linear_thread = threadIdx().x - 1
    threads_per_pair = div(blockDim().x, pairs_per_block)
    pair_group = div(linear_thread, threads_per_pair) + 1
    tid = (linear_thread % threads_per_pair) + 1
    scratch_tid = threadIdx().x
    group_base = scratch_tid - tid
    pair = (blockIdx().x - 1) * pairs_per_block + pair_group
    pair > total_pairs && return nothing

    index_count = length(test_indices)
    test_loop_index = ((pair - 1) % index_count) + 1
    trial_loop_index = div(pair - 1, index_count) + 1
    test_index = test_indices[test_loop_index]
    trial_index = trial_indices[trial_loop_index]

    t1 = faces[test_index]
    t2 = faces[test_index + face_count]
    t3 = faces[test_index + 2 * face_count]
    r1 = faces[trial_index]
    r2 = faces[trial_index + face_count]
    r3 = faces[trial_index + 2 * face_count]

    adjacent = t1 == r1 || t1 == r2 || t1 == r3 ||
        t2 == r1 || t2 == r2 || t2 == r3 ||
        t3 == r1 || t3 == r2 || t3 == r3
    adjacent && return nothing

    tv1x = face_vertices[test_index]
    tv1y = face_vertices[test_index + face_count]
    tv1z = face_vertices[test_index + 2 * face_count]
    tv2x = face_vertices[test_index + 3 * face_count]
    tv2y = face_vertices[test_index + 4 * face_count]
    tv2z = face_vertices[test_index + 5 * face_count]
    tv3x = face_vertices[test_index + 6 * face_count]
    tv3y = face_vertices[test_index + 7 * face_count]
    tv3z = face_vertices[test_index + 8 * face_count]

    rv1x = face_vertices[trial_index]
    rv1y = face_vertices[trial_index + face_count]
    rv1z = face_vertices[trial_index + 2 * face_count]
    rv2x = face_vertices[trial_index + 3 * face_count]
    rv2y = face_vertices[trial_index + 4 * face_count]
    rv2z = face_vertices[trial_index + 5 * face_count]
    rv3x = face_vertices[trial_index + 6 * face_count]
    rv3y = face_vertices[trial_index + 7 * face_count]
    rv3z = face_vertices[trial_index + 8 * face_count]

    tnx = normals[test_index]
    tny = normals[test_index + face_count]
    tnz = normals[test_index + 2 * face_count]
    rnx = normals[trial_index]
    rny = normals[trial_index + face_count]
    rnz = normals[trial_index + 2 * face_count]
    normal_product = tnx * rnx + tny * rny + tnz * rnz

    tc11 = curls[test_index]
    tc12 = curls[test_index + face_count]
    tc13 = curls[test_index + 2 * face_count]
    tc21 = curls[test_index + 3 * face_count]
    tc22 = curls[test_index + 4 * face_count]
    tc23 = curls[test_index + 5 * face_count]
    tc31 = curls[test_index + 6 * face_count]
    tc32 = curls[test_index + 7 * face_count]
    tc33 = curls[test_index + 8 * face_count]

    rc11 = curls[trial_index]
    rc12 = curls[trial_index + face_count]
    rc13 = curls[trial_index + 2 * face_count]
    rc21 = curls[trial_index + 3 * face_count]
    rc22 = curls[trial_index + 4 * face_count]
    rc23 = curls[trial_index + 5 * face_count]
    rc31 = curls[trial_index + 6 * face_count]
    rc32 = curls[trial_index + 7 * face_count]
    rc33 = curls[trial_index + 8 * face_count]

    jac_scale = typeof(k)(4) * areas[test_index] * areas[trial_index]
    four_pi = typeof(k)(12.566370614359172)
    qpair_count = rule_count * rule_count
    qpair = tid
    accumulator_count = 24
    scratch = CUDA.@cuDynamicSharedMem(typeof(k), blockDim().x * accumulator_count)

    slp1_re = zero(k); slp2_re = zero(k); slp3_re = zero(k)
    slp1_im = zero(k); slp2_im = zero(k); slp3_im = zero(k)
    hyp11_re = zero(k); hyp12_re = zero(k); hyp13_re = zero(k)
    hyp21_re = zero(k); hyp22_re = zero(k); hyp23_re = zero(k)
    hyp31_re = zero(k); hyp32_re = zero(k); hyp33_re = zero(k)
    hyp11_im = zero(k); hyp12_im = zero(k); hyp13_im = zero(k)
    hyp21_im = zero(k); hyp22_im = zero(k); hyp23_im = zero(k)
    hyp31_im = zero(k); hyp32_im = zero(k); hyp33_im = zero(k)

    while qpair <= qpair_count
        tq = ((qpair - 1) % rule_count) + 1
        rq = div(qpair - 1, rule_count) + 1

        txi = rule_points[tq]
        teta = rule_points[tq + rule_count]
        tw = rule_weights[tq]
        tv1 = one(k) - txi - teta
        tv2 = txi
        tv3 = teta
        x = tv1 * tv1x + tv2 * tv2x + tv3 * tv3x
        y = tv1 * tv1y + tv2 * tv2y + tv3 * tv3y
        z = tv1 * tv1z + tv2 * tv2z + tv3 * tv3z

        rxi = rule_points[rq]
        reta = rule_points[rq + rule_count]
        rw = rule_weights[rq]
        rv1 = one(k) - rxi - reta
        rv2 = rxi
        rv3 = reta
        sx = rv1 * rv1x + rv2 * rv2x + rv3 * rv3x
        sy = rv1 * rv1y + rv2 * rv2y + rv3 * rv3y
        sz = rv1 * rv1z + rv2 * rv2z + rv3 * rv3z

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
            weight = tw * rw * jac_scale
            weighted_re = green_re * weight
            weighted_im = green_im * weight

            slp1_re += tv1 * weighted_re
            slp2_re += tv2 * weighted_re
            slp3_re += tv3 * weighted_re
            slp1_im += tv1 * weighted_im
            slp2_im += tv2 * weighted_im
            slp3_im += tv3 * weighted_im

            h11 = (tc11 * rc11 + tc12 * rc12 + tc13 * rc13) - k * k * tv1 * rv1 * normal_product
            h12 = (tc11 * rc21 + tc12 * rc22 + tc13 * rc23) - k * k * tv1 * rv2 * normal_product
            h13 = (tc11 * rc31 + tc12 * rc32 + tc13 * rc33) - k * k * tv1 * rv3 * normal_product
            h21 = (tc21 * rc11 + tc22 * rc12 + tc23 * rc13) - k * k * tv2 * rv1 * normal_product
            h22 = (tc21 * rc21 + tc22 * rc22 + tc23 * rc23) - k * k * tv2 * rv2 * normal_product
            h23 = (tc21 * rc31 + tc22 * rc32 + tc23 * rc33) - k * k * tv2 * rv3 * normal_product
            h31 = (tc31 * rc11 + tc32 * rc12 + tc33 * rc13) - k * k * tv3 * rv1 * normal_product
            h32 = (tc31 * rc21 + tc32 * rc22 + tc33 * rc23) - k * k * tv3 * rv2 * normal_product
            h33 = (tc31 * rc31 + tc32 * rc32 + tc33 * rc33) - k * k * tv3 * rv3 * normal_product

            hyp11_re += h11 * weighted_re
            hyp12_re += h12 * weighted_re
            hyp13_re += h13 * weighted_re
            hyp21_re += h21 * weighted_re
            hyp22_re += h22 * weighted_re
            hyp23_re += h23 * weighted_re
            hyp31_re += h31 * weighted_re
            hyp32_re += h32 * weighted_re
            hyp33_re += h33 * weighted_re
            hyp11_im += h11 * weighted_im
            hyp12_im += h12 * weighted_im
            hyp13_im += h13 * weighted_im
            hyp21_im += h21 * weighted_im
            hyp22_im += h22 * weighted_im
            hyp23_im += h23 * weighted_im
            hyp31_im += h31 * weighted_im
            hyp32_im += h32 * weighted_im
            hyp33_im += h33 * weighted_im
        end

        qpair += threads_per_pair
    end

    stride = blockDim().x
    scratch[scratch_tid + 0 * stride] = slp1_re
    scratch[scratch_tid + 1 * stride] = slp2_re
    scratch[scratch_tid + 2 * stride] = slp3_re
    scratch[scratch_tid + 3 * stride] = slp1_im
    scratch[scratch_tid + 4 * stride] = slp2_im
    scratch[scratch_tid + 5 * stride] = slp3_im
    scratch[scratch_tid + 6 * stride] = hyp11_re
    scratch[scratch_tid + 7 * stride] = hyp12_re
    scratch[scratch_tid + 8 * stride] = hyp13_re
    scratch[scratch_tid + 9 * stride] = hyp21_re
    scratch[scratch_tid + 10 * stride] = hyp22_re
    scratch[scratch_tid + 11 * stride] = hyp23_re
    scratch[scratch_tid + 12 * stride] = hyp31_re
    scratch[scratch_tid + 13 * stride] = hyp32_re
    scratch[scratch_tid + 14 * stride] = hyp33_re
    scratch[scratch_tid + 15 * stride] = hyp11_im
    scratch[scratch_tid + 16 * stride] = hyp12_im
    scratch[scratch_tid + 17 * stride] = hyp13_im
    scratch[scratch_tid + 18 * stride] = hyp21_im
    scratch[scratch_tid + 19 * stride] = hyp22_im
    scratch[scratch_tid + 20 * stride] = hyp23_im
    scratch[scratch_tid + 21 * stride] = hyp31_im
    scratch[scratch_tid + 22 * stride] = hyp32_im
    scratch[scratch_tid + 23 * stride] = hyp33_im
    sync_threads()

    offset = threads_per_pair >>> 1
    while offset > 0
        if tid <= offset
            for slot in 0:23
                scratch[scratch_tid + slot * stride] += scratch[scratch_tid + offset + slot * stride]
            end
        end
        sync_threads()
        offset >>>= 1
    end

    if tid == 1
        slp_col = trial_index
        _cuda_atomic_add!(slp_re, t1 + (slp_col - 1) * p1_dof_count, scratch[group_base + 1 + 0 * stride])
        _cuda_atomic_add!(slp_re, t2 + (slp_col - 1) * p1_dof_count, scratch[group_base + 1 + 1 * stride])
        _cuda_atomic_add!(slp_re, t3 + (slp_col - 1) * p1_dof_count, scratch[group_base + 1 + 2 * stride])
        _cuda_atomic_add!(slp_im, t1 + (slp_col - 1) * p1_dof_count, scratch[group_base + 1 + 3 * stride])
        _cuda_atomic_add!(slp_im, t2 + (slp_col - 1) * p1_dof_count, scratch[group_base + 1 + 4 * stride])
        _cuda_atomic_add!(slp_im, t3 + (slp_col - 1) * p1_dof_count, scratch[group_base + 1 + 5 * stride])

        _cuda_atomic_add!(hyp_re, t1 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 6 * stride])
        _cuda_atomic_add!(hyp_re, t1 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 7 * stride])
        _cuda_atomic_add!(hyp_re, t1 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 8 * stride])
        _cuda_atomic_add!(hyp_re, t2 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 9 * stride])
        _cuda_atomic_add!(hyp_re, t2 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 10 * stride])
        _cuda_atomic_add!(hyp_re, t2 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 11 * stride])
        _cuda_atomic_add!(hyp_re, t3 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 12 * stride])
        _cuda_atomic_add!(hyp_re, t3 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 13 * stride])
        _cuda_atomic_add!(hyp_re, t3 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 14 * stride])
        _cuda_atomic_add!(hyp_im, t1 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 15 * stride])
        _cuda_atomic_add!(hyp_im, t1 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 16 * stride])
        _cuda_atomic_add!(hyp_im, t1 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 17 * stride])
        _cuda_atomic_add!(hyp_im, t2 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 18 * stride])
        _cuda_atomic_add!(hyp_im, t2 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 19 * stride])
        _cuda_atomic_add!(hyp_im, t2 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 20 * stride])
        _cuda_atomic_add!(hyp_im, t3 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 21 * stride])
        _cuda_atomic_add!(hyp_im, t3 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 22 * stride])
        _cuda_atomic_add!(hyp_im, t3 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 23 * stride])
    end

    return nothing
end

function _cuda_regular_quadrature_dlp_adjoint_kernel!(
    dlp_re,
    dlp_im,
    adj_re,
    adj_im,
    face_vertices,
    normals,
    areas,
    faces,
    test_indices,
    trial_indices,
    rule_points,
    rule_weights,
    k,
    p1_dof_count,
    face_count,
    rule_count,
    total_pairs,
    pairs_per_block,
)
    linear_thread = threadIdx().x - 1
    threads_per_pair = div(blockDim().x, pairs_per_block)
    pair_group = div(linear_thread, threads_per_pair) + 1
    tid = (linear_thread % threads_per_pair) + 1
    scratch_tid = threadIdx().x
    group_base = scratch_tid - tid
    pair = (blockIdx().x - 1) * pairs_per_block + pair_group
    pair > total_pairs && return nothing

    index_count = length(test_indices)
    test_loop_index = ((pair - 1) % index_count) + 1
    trial_loop_index = div(pair - 1, index_count) + 1
    test_index = test_indices[test_loop_index]
    trial_index = trial_indices[trial_loop_index]

    t1 = faces[test_index]
    t2 = faces[test_index + face_count]
    t3 = faces[test_index + 2 * face_count]
    r1 = faces[trial_index]
    r2 = faces[trial_index + face_count]
    r3 = faces[trial_index + 2 * face_count]

    adjacent = t1 == r1 || t1 == r2 || t1 == r3 ||
        t2 == r1 || t2 == r2 || t2 == r3 ||
        t3 == r1 || t3 == r2 || t3 == r3
    adjacent && return nothing

    tv1x = face_vertices[test_index]
    tv1y = face_vertices[test_index + face_count]
    tv1z = face_vertices[test_index + 2 * face_count]
    tv2x = face_vertices[test_index + 3 * face_count]
    tv2y = face_vertices[test_index + 4 * face_count]
    tv2z = face_vertices[test_index + 5 * face_count]
    tv3x = face_vertices[test_index + 6 * face_count]
    tv3y = face_vertices[test_index + 7 * face_count]
    tv3z = face_vertices[test_index + 8 * face_count]

    rv1x = face_vertices[trial_index]
    rv1y = face_vertices[trial_index + face_count]
    rv1z = face_vertices[trial_index + 2 * face_count]
    rv2x = face_vertices[trial_index + 3 * face_count]
    rv2y = face_vertices[trial_index + 4 * face_count]
    rv2z = face_vertices[trial_index + 5 * face_count]
    rv3x = face_vertices[trial_index + 6 * face_count]
    rv3y = face_vertices[trial_index + 7 * face_count]
    rv3z = face_vertices[trial_index + 8 * face_count]

    tnx = normals[test_index]
    tny = normals[test_index + face_count]
    tnz = normals[test_index + 2 * face_count]
    rnx = normals[trial_index]
    rny = normals[trial_index + face_count]
    rnz = normals[trial_index + 2 * face_count]

    jac_scale = typeof(k)(4) * areas[test_index] * areas[trial_index]
    four_pi = typeof(k)(12.566370614359172)
    qpair_count = rule_count * rule_count
    qpair = tid
    accumulator_count = 24
    scratch = CUDA.@cuDynamicSharedMem(typeof(k), blockDim().x * accumulator_count)

    dlp11_re = zero(k); dlp12_re = zero(k); dlp13_re = zero(k)
    dlp21_re = zero(k); dlp22_re = zero(k); dlp23_re = zero(k)
    dlp31_re = zero(k); dlp32_re = zero(k); dlp33_re = zero(k)
    dlp11_im = zero(k); dlp12_im = zero(k); dlp13_im = zero(k)
    dlp21_im = zero(k); dlp22_im = zero(k); dlp23_im = zero(k)
    dlp31_im = zero(k); dlp32_im = zero(k); dlp33_im = zero(k)
    adj1_re = zero(k); adj2_re = zero(k); adj3_re = zero(k)
    adj1_im = zero(k); adj2_im = zero(k); adj3_im = zero(k)

    while qpair <= qpair_count
        tq = ((qpair - 1) % rule_count) + 1
        rq = div(qpair - 1, rule_count) + 1

        txi = rule_points[tq]
        teta = rule_points[tq + rule_count]
        tw = rule_weights[tq]
        tv1 = one(k) - txi - teta
        tv2 = txi
        tv3 = teta
        x = tv1 * tv1x + tv2 * tv2x + tv3 * tv3x
        y = tv1 * tv1y + tv2 * tv2y + tv3 * tv3y
        z = tv1 * tv1z + tv2 * tv2z + tv3 * tv3z

        rxi = rule_points[rq]
        reta = rule_points[rq + rule_count]
        rw = rule_weights[rq]
        rv1 = one(k) - rxi - reta
        rv2 = rxi
        rv3 = reta
        sx = rv1 * rv1x + rv2 * rv2x + rv3 * rv3x
        sy = rv1 * rv1y + rv2 * rv2y + rv3 * rv3y
        sz = rv1 * rv1z + rv2 * rv2z + rv3 * rv3z

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
            weight = tw * rw * jac_scale

            source_projection = (dx * rnx + dy * rny + dz * rnz) * inv_radius
            test_projection = -(dx * tnx + dy * tny + dz * tnz) * inv_radius
            factor_re = -inv_radius
            factor_im = k
            deriv_re = green_re * factor_re - green_im * factor_im
            deriv_im = green_re * factor_im + green_im * factor_re
            dlp_value_re = deriv_re * source_projection * weight
            dlp_value_im = deriv_im * source_projection * weight
            adj_value_re = deriv_re * test_projection * weight
            adj_value_im = deriv_im * test_projection * weight

            dlp11_re += tv1 * rv1 * dlp_value_re
            dlp12_re += tv1 * rv2 * dlp_value_re
            dlp13_re += tv1 * rv3 * dlp_value_re
            dlp21_re += tv2 * rv1 * dlp_value_re
            dlp22_re += tv2 * rv2 * dlp_value_re
            dlp23_re += tv2 * rv3 * dlp_value_re
            dlp31_re += tv3 * rv1 * dlp_value_re
            dlp32_re += tv3 * rv2 * dlp_value_re
            dlp33_re += tv3 * rv3 * dlp_value_re
            dlp11_im += tv1 * rv1 * dlp_value_im
            dlp12_im += tv1 * rv2 * dlp_value_im
            dlp13_im += tv1 * rv3 * dlp_value_im
            dlp21_im += tv2 * rv1 * dlp_value_im
            dlp22_im += tv2 * rv2 * dlp_value_im
            dlp23_im += tv2 * rv3 * dlp_value_im
            dlp31_im += tv3 * rv1 * dlp_value_im
            dlp32_im += tv3 * rv2 * dlp_value_im
            dlp33_im += tv3 * rv3 * dlp_value_im

            adj1_re += tv1 * adj_value_re
            adj2_re += tv2 * adj_value_re
            adj3_re += tv3 * adj_value_re
            adj1_im += tv1 * adj_value_im
            adj2_im += tv2 * adj_value_im
            adj3_im += tv3 * adj_value_im
        end

        qpair += threads_per_pair
    end

    stride = blockDim().x
    scratch[scratch_tid + 0 * stride] = dlp11_re
    scratch[scratch_tid + 1 * stride] = dlp12_re
    scratch[scratch_tid + 2 * stride] = dlp13_re
    scratch[scratch_tid + 3 * stride] = dlp21_re
    scratch[scratch_tid + 4 * stride] = dlp22_re
    scratch[scratch_tid + 5 * stride] = dlp23_re
    scratch[scratch_tid + 6 * stride] = dlp31_re
    scratch[scratch_tid + 7 * stride] = dlp32_re
    scratch[scratch_tid + 8 * stride] = dlp33_re
    scratch[scratch_tid + 9 * stride] = dlp11_im
    scratch[scratch_tid + 10 * stride] = dlp12_im
    scratch[scratch_tid + 11 * stride] = dlp13_im
    scratch[scratch_tid + 12 * stride] = dlp21_im
    scratch[scratch_tid + 13 * stride] = dlp22_im
    scratch[scratch_tid + 14 * stride] = dlp23_im
    scratch[scratch_tid + 15 * stride] = dlp31_im
    scratch[scratch_tid + 16 * stride] = dlp32_im
    scratch[scratch_tid + 17 * stride] = dlp33_im
    scratch[scratch_tid + 18 * stride] = adj1_re
    scratch[scratch_tid + 19 * stride] = adj2_re
    scratch[scratch_tid + 20 * stride] = adj3_re
    scratch[scratch_tid + 21 * stride] = adj1_im
    scratch[scratch_tid + 22 * stride] = adj2_im
    scratch[scratch_tid + 23 * stride] = adj3_im
    sync_threads()

    offset = threads_per_pair >>> 1
    while offset > 0
        if tid <= offset
            for slot in 0:23
                scratch[scratch_tid + slot * stride] += scratch[scratch_tid + offset + slot * stride]
            end
        end
        sync_threads()
        offset >>>= 1
    end

    if tid == 1
        _cuda_atomic_add!(dlp_re, t1 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 0 * stride])
        _cuda_atomic_add!(dlp_re, t1 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 1 * stride])
        _cuda_atomic_add!(dlp_re, t1 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 2 * stride])
        _cuda_atomic_add!(dlp_re, t2 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 3 * stride])
        _cuda_atomic_add!(dlp_re, t2 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 4 * stride])
        _cuda_atomic_add!(dlp_re, t2 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 5 * stride])
        _cuda_atomic_add!(dlp_re, t3 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 6 * stride])
        _cuda_atomic_add!(dlp_re, t3 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 7 * stride])
        _cuda_atomic_add!(dlp_re, t3 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 8 * stride])
        _cuda_atomic_add!(dlp_im, t1 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 9 * stride])
        _cuda_atomic_add!(dlp_im, t1 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 10 * stride])
        _cuda_atomic_add!(dlp_im, t1 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 11 * stride])
        _cuda_atomic_add!(dlp_im, t2 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 12 * stride])
        _cuda_atomic_add!(dlp_im, t2 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 13 * stride])
        _cuda_atomic_add!(dlp_im, t2 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 14 * stride])
        _cuda_atomic_add!(dlp_im, t3 + (r1 - 1) * p1_dof_count, scratch[group_base + 1 + 15 * stride])
        _cuda_atomic_add!(dlp_im, t3 + (r2 - 1) * p1_dof_count, scratch[group_base + 1 + 16 * stride])
        _cuda_atomic_add!(dlp_im, t3 + (r3 - 1) * p1_dof_count, scratch[group_base + 1 + 17 * stride])

        adj_col = trial_index
        _cuda_atomic_add!(adj_re, t1 + (adj_col - 1) * p1_dof_count, scratch[group_base + 1 + 18 * stride])
        _cuda_atomic_add!(adj_re, t2 + (adj_col - 1) * p1_dof_count, scratch[group_base + 1 + 19 * stride])
        _cuda_atomic_add!(adj_re, t3 + (adj_col - 1) * p1_dof_count, scratch[group_base + 1 + 20 * stride])
        _cuda_atomic_add!(adj_im, t1 + (adj_col - 1) * p1_dof_count, scratch[group_base + 1 + 21 * stride])
        _cuda_atomic_add!(adj_im, t2 + (adj_col - 1) * p1_dof_count, scratch[group_base + 1 + 22 * stride])
        _cuda_atomic_add!(adj_im, t3 + (adj_col - 1) * p1_dof_count, scratch[group_base + 1 + 23 * stride])
    end

    return nothing
end

