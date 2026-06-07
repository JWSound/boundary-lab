function _cuda_regular_quadrature_probe_kernel!(
    sink,
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
    face_count,
    rule_count,
    total_pairs,
    probe_mode,
)
    pair = blockIdx().x
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
    qpair = threadIdx().x
    scratch = CUDA.@cuDynamicSharedMem(typeof(k), blockDim().x)
    acc = zero(k)

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

            if probe_mode == 1
                acc += weighted_re + weighted_im
            else
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

                c11 = (tc11 * rc11 + tc12 * rc12 + tc13 * rc13) - k * k * tv1 * rv1 * normal_product
                c12 = (tc11 * rc21 + tc12 * rc22 + tc13 * rc23) - k * k * tv1 * rv2 * normal_product
                c13 = (tc11 * rc31 + tc12 * rc32 + tc13 * rc33) - k * k * tv1 * rv3 * normal_product
                c21 = (tc21 * rc11 + tc22 * rc12 + tc23 * rc13) - k * k * tv2 * rv1 * normal_product
                c22 = (tc21 * rc21 + tc22 * rc22 + tc23 * rc23) - k * k * tv2 * rv2 * normal_product
                c23 = (tc21 * rc31 + tc22 * rc32 + tc23 * rc33) - k * k * tv2 * rv3 * normal_product
                c31 = (tc31 * rc11 + tc32 * rc12 + tc33 * rc13) - k * k * tv3 * rv1 * normal_product
                c32 = (tc31 * rc21 + tc32 * rc22 + tc33 * rc23) - k * k * tv3 * rv2 * normal_product
                c33 = (tc31 * rc31 + tc32 * rc32 + tc33 * rc33) - k * k * tv3 * rv3 * normal_product

                basis_sum = tv1 + tv2 + tv3
                dlp_basis_sum = (
                    tv1 * rv1 + tv1 * rv2 + tv1 * rv3 +
                    tv2 * rv1 + tv2 * rv2 + tv2 * rv3 +
                    tv3 * rv1 + tv3 * rv2 + tv3 * rv3
                )
                hyp_coeff_sum = c11 + c12 + c13 + c21 + c22 + c23 + c31 + c32 + c33

                if probe_mode == 2
                    acc += weighted_re + weighted_im
                    acc += basis_sum * (weighted_re + weighted_im)
                    acc += basis_sum * (adj_value_re + adj_value_im)
                    acc += dlp_basis_sum * (dlp_value_re + dlp_value_im)
                    acc += hyp_coeff_sum * (weighted_re + weighted_im)
                elseif probe_mode == 3
                    acc += basis_sum * (weighted_re + weighted_im)
                elseif probe_mode == 4
                    acc += basis_sum * (adj_value_re + adj_value_im)
                elseif probe_mode == 5
                    acc += dlp_basis_sum * (dlp_value_re + dlp_value_im)
                elseif probe_mode == 6
                    acc += hyp_coeff_sum * (weighted_re + weighted_im)
                end
            end
        end

        qpair += blockDim().x
    end

    reduced = _cuda_block_sum(acc, scratch)
    threadIdx().x == 1 && _cuda_atomic_add!(sink, 1, reduced)
    return nothing
end

function _cuda_regular_quadrature_slp_adjoint_kernel!(
    slp_re,
    slp_im,
    adj_re,
    adj_im,
    face_vertices,
    normals,
    areas,
    faces,
    test_indices,
    test_offset,
    test_count,
    trial_indices,
    trial_count,
    rule_points,
    rule_weights,
    k,
    p1_dof_count,
    face_count,
    rule_count,
    total_pairs,
    atomic_writes,
)
    pair = blockIdx().x
    pair > total_pairs && return nothing

    test_loop_index = ((pair - 1) % test_count) + 1
    trial_loop_index = div(pair - 1, test_count) + 1
    test_index = test_indices[test_offset + test_loop_index - 1]
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

    jac_scale = typeof(k)(4) * areas[test_index] * areas[trial_index]
    four_pi = typeof(k)(12.566370614359172)
    qpair_count = rule_count * rule_count
    qpair = threadIdx().x
    accumulator_count = 12
    scratch = CUDA.@cuDynamicSharedMem(typeof(k), blockDim().x * accumulator_count)

    slp1_re = zero(k)
    slp2_re = zero(k)
    slp3_re = zero(k)
    slp1_im = zero(k)
    slp2_im = zero(k)
    slp3_im = zero(k)
    adj1_re = zero(k)
    adj2_re = zero(k)
    adj3_re = zero(k)
    adj1_im = zero(k)
    adj2_im = zero(k)
    adj3_im = zero(k)

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

            test_projection = -(dx * tnx + dy * tny + dz * tnz) * inv_radius
            factor_re = -inv_radius
            factor_im = k
            deriv_re = green_re * factor_re - green_im * factor_im
            deriv_im = green_re * factor_im + green_im * factor_re
            adj_value_re = deriv_re * test_projection * weight
            adj_value_im = deriv_im * test_projection * weight

            slp1_re += tv1 * weighted_re
            slp2_re += tv2 * weighted_re
            slp3_re += tv3 * weighted_re
            slp1_im += tv1 * weighted_im
            slp2_im += tv2 * weighted_im
            slp3_im += tv3 * weighted_im

            adj1_re += tv1 * adj_value_re
            adj2_re += tv2 * adj_value_re
            adj3_re += tv3 * adj_value_re
            adj1_im += tv1 * adj_value_im
            adj2_im += tv2 * adj_value_im
            adj3_im += tv3 * adj_value_im
        end

        qpair += blockDim().x
    end

    tid = threadIdx().x
    stride = blockDim().x
    scratch[tid + 0 * stride] = slp1_re
    scratch[tid + 1 * stride] = slp2_re
    scratch[tid + 2 * stride] = slp3_re
    scratch[tid + 3 * stride] = slp1_im
    scratch[tid + 4 * stride] = slp2_im
    scratch[tid + 5 * stride] = slp3_im
    scratch[tid + 6 * stride] = adj1_re
    scratch[tid + 7 * stride] = adj2_re
    scratch[tid + 8 * stride] = adj3_re
    scratch[tid + 9 * stride] = adj1_im
    scratch[tid + 10 * stride] = adj2_im
    scratch[tid + 11 * stride] = adj3_im
    sync_threads()

    offset = stride >>> 1
    while offset > 0
        if tid <= offset
            for slot in 0:11
                scratch[tid + slot * stride] += scratch[tid + offset + slot * stride]
            end
        end
        sync_threads()
        offset >>>= 1
    end

    if threadIdx().x == 1
        slp_col = trial_index
        index1 = t1 + (slp_col - 1) * p1_dof_count
        index2 = t2 + (slp_col - 1) * p1_dof_count
        index3 = t3 + (slp_col - 1) * p1_dof_count
        if atomic_writes
            _cuda_atomic_add!(slp_re, index1, scratch[1 + 0 * stride])
            _cuda_atomic_add!(slp_re, index2, scratch[1 + 1 * stride])
            _cuda_atomic_add!(slp_re, index3, scratch[1 + 2 * stride])
            _cuda_atomic_add!(slp_im, index1, scratch[1 + 3 * stride])
            _cuda_atomic_add!(slp_im, index2, scratch[1 + 4 * stride])
            _cuda_atomic_add!(slp_im, index3, scratch[1 + 5 * stride])
            _cuda_atomic_add!(adj_re, index1, scratch[1 + 6 * stride])
            _cuda_atomic_add!(adj_re, index2, scratch[1 + 7 * stride])
            _cuda_atomic_add!(adj_re, index3, scratch[1 + 8 * stride])
            _cuda_atomic_add!(adj_im, index1, scratch[1 + 9 * stride])
            _cuda_atomic_add!(adj_im, index2, scratch[1 + 10 * stride])
            _cuda_atomic_add!(adj_im, index3, scratch[1 + 11 * stride])
        else
            slp_re[index1] += scratch[1 + 0 * stride]
            slp_re[index2] += scratch[1 + 1 * stride]
            slp_re[index3] += scratch[1 + 2 * stride]
            slp_im[index1] += scratch[1 + 3 * stride]
            slp_im[index2] += scratch[1 + 4 * stride]
            slp_im[index3] += scratch[1 + 5 * stride]
            adj_re[index1] += scratch[1 + 6 * stride]
            adj_re[index2] += scratch[1 + 7 * stride]
            adj_re[index3] += scratch[1 + 8 * stride]
            adj_im[index1] += scratch[1 + 9 * stride]
            adj_im[index2] += scratch[1 + 10 * stride]
            adj_im[index3] += scratch[1 + 11 * stride]
        end
    end

    return nothing
end

function _cuda_regular_quadrature_dlp_hyp_kernel!(
    dlp_re,
    dlp_im,
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
)
    pair = blockIdx().x
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
    qpair = threadIdx().x
    accumulator_count = 36
    scratch = CUDA.@cuDynamicSharedMem(typeof(k), blockDim().x * accumulator_count)

    dlp11_re = zero(k); dlp12_re = zero(k); dlp13_re = zero(k)
    dlp21_re = zero(k); dlp22_re = zero(k); dlp23_re = zero(k)
    dlp31_re = zero(k); dlp32_re = zero(k); dlp33_re = zero(k)
    dlp11_im = zero(k); dlp12_im = zero(k); dlp13_im = zero(k)
    dlp21_im = zero(k); dlp22_im = zero(k); dlp23_im = zero(k)
    dlp31_im = zero(k); dlp32_im = zero(k); dlp33_im = zero(k)
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

            source_projection = (dx * rnx + dy * rny + dz * rnz) * inv_radius
            factor_re = -inv_radius
            factor_im = k
            deriv_re = green_re * factor_re - green_im * factor_im
            deriv_im = green_re * factor_im + green_im * factor_re
            dlp_value_re = deriv_re * source_projection * weight
            dlp_value_im = deriv_im * source_projection * weight

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

        qpair += blockDim().x
    end

    tid = threadIdx().x
    stride = blockDim().x
    scratch[tid + 0 * stride] = dlp11_re
    scratch[tid + 1 * stride] = dlp12_re
    scratch[tid + 2 * stride] = dlp13_re
    scratch[tid + 3 * stride] = dlp21_re
    scratch[tid + 4 * stride] = dlp22_re
    scratch[tid + 5 * stride] = dlp23_re
    scratch[tid + 6 * stride] = dlp31_re
    scratch[tid + 7 * stride] = dlp32_re
    scratch[tid + 8 * stride] = dlp33_re
    scratch[tid + 9 * stride] = dlp11_im
    scratch[tid + 10 * stride] = dlp12_im
    scratch[tid + 11 * stride] = dlp13_im
    scratch[tid + 12 * stride] = dlp21_im
    scratch[tid + 13 * stride] = dlp22_im
    scratch[tid + 14 * stride] = dlp23_im
    scratch[tid + 15 * stride] = dlp31_im
    scratch[tid + 16 * stride] = dlp32_im
    scratch[tid + 17 * stride] = dlp33_im
    scratch[tid + 18 * stride] = hyp11_re
    scratch[tid + 19 * stride] = hyp12_re
    scratch[tid + 20 * stride] = hyp13_re
    scratch[tid + 21 * stride] = hyp21_re
    scratch[tid + 22 * stride] = hyp22_re
    scratch[tid + 23 * stride] = hyp23_re
    scratch[tid + 24 * stride] = hyp31_re
    scratch[tid + 25 * stride] = hyp32_re
    scratch[tid + 26 * stride] = hyp33_re
    scratch[tid + 27 * stride] = hyp11_im
    scratch[tid + 28 * stride] = hyp12_im
    scratch[tid + 29 * stride] = hyp13_im
    scratch[tid + 30 * stride] = hyp21_im
    scratch[tid + 31 * stride] = hyp22_im
    scratch[tid + 32 * stride] = hyp23_im
    scratch[tid + 33 * stride] = hyp31_im
    scratch[tid + 34 * stride] = hyp32_im
    scratch[tid + 35 * stride] = hyp33_im
    sync_threads()

    offset = stride >>> 1
    while offset > 0
        if tid <= offset
            for slot in 0:35
                scratch[tid + slot * stride] += scratch[tid + offset + slot * stride]
            end
        end
        sync_threads()
        offset >>>= 1
    end

    if threadIdx().x == 1
        _cuda_atomic_add!(dlp_re, t1 + (r1 - 1) * p1_dof_count, scratch[1 + 0 * stride])
        _cuda_atomic_add!(dlp_re, t1 + (r2 - 1) * p1_dof_count, scratch[1 + 1 * stride])
        _cuda_atomic_add!(dlp_re, t1 + (r3 - 1) * p1_dof_count, scratch[1 + 2 * stride])
        _cuda_atomic_add!(dlp_re, t2 + (r1 - 1) * p1_dof_count, scratch[1 + 3 * stride])
        _cuda_atomic_add!(dlp_re, t2 + (r2 - 1) * p1_dof_count, scratch[1 + 4 * stride])
        _cuda_atomic_add!(dlp_re, t2 + (r3 - 1) * p1_dof_count, scratch[1 + 5 * stride])
        _cuda_atomic_add!(dlp_re, t3 + (r1 - 1) * p1_dof_count, scratch[1 + 6 * stride])
        _cuda_atomic_add!(dlp_re, t3 + (r2 - 1) * p1_dof_count, scratch[1 + 7 * stride])
        _cuda_atomic_add!(dlp_re, t3 + (r3 - 1) * p1_dof_count, scratch[1 + 8 * stride])
        _cuda_atomic_add!(dlp_im, t1 + (r1 - 1) * p1_dof_count, scratch[1 + 9 * stride])
        _cuda_atomic_add!(dlp_im, t1 + (r2 - 1) * p1_dof_count, scratch[1 + 10 * stride])
        _cuda_atomic_add!(dlp_im, t1 + (r3 - 1) * p1_dof_count, scratch[1 + 11 * stride])
        _cuda_atomic_add!(dlp_im, t2 + (r1 - 1) * p1_dof_count, scratch[1 + 12 * stride])
        _cuda_atomic_add!(dlp_im, t2 + (r2 - 1) * p1_dof_count, scratch[1 + 13 * stride])
        _cuda_atomic_add!(dlp_im, t2 + (r3 - 1) * p1_dof_count, scratch[1 + 14 * stride])
        _cuda_atomic_add!(dlp_im, t3 + (r1 - 1) * p1_dof_count, scratch[1 + 15 * stride])
        _cuda_atomic_add!(dlp_im, t3 + (r2 - 1) * p1_dof_count, scratch[1 + 16 * stride])
        _cuda_atomic_add!(dlp_im, t3 + (r3 - 1) * p1_dof_count, scratch[1 + 17 * stride])
        _cuda_atomic_add!(hyp_re, t1 + (r1 - 1) * p1_dof_count, scratch[1 + 18 * stride])
        _cuda_atomic_add!(hyp_re, t1 + (r2 - 1) * p1_dof_count, scratch[1 + 19 * stride])
        _cuda_atomic_add!(hyp_re, t1 + (r3 - 1) * p1_dof_count, scratch[1 + 20 * stride])
        _cuda_atomic_add!(hyp_re, t2 + (r1 - 1) * p1_dof_count, scratch[1 + 21 * stride])
        _cuda_atomic_add!(hyp_re, t2 + (r2 - 1) * p1_dof_count, scratch[1 + 22 * stride])
        _cuda_atomic_add!(hyp_re, t2 + (r3 - 1) * p1_dof_count, scratch[1 + 23 * stride])
        _cuda_atomic_add!(hyp_re, t3 + (r1 - 1) * p1_dof_count, scratch[1 + 24 * stride])
        _cuda_atomic_add!(hyp_re, t3 + (r2 - 1) * p1_dof_count, scratch[1 + 25 * stride])
        _cuda_atomic_add!(hyp_re, t3 + (r3 - 1) * p1_dof_count, scratch[1 + 26 * stride])
        _cuda_atomic_add!(hyp_im, t1 + (r1 - 1) * p1_dof_count, scratch[1 + 27 * stride])
        _cuda_atomic_add!(hyp_im, t1 + (r2 - 1) * p1_dof_count, scratch[1 + 28 * stride])
        _cuda_atomic_add!(hyp_im, t1 + (r3 - 1) * p1_dof_count, scratch[1 + 29 * stride])
        _cuda_atomic_add!(hyp_im, t2 + (r1 - 1) * p1_dof_count, scratch[1 + 30 * stride])
        _cuda_atomic_add!(hyp_im, t2 + (r2 - 1) * p1_dof_count, scratch[1 + 31 * stride])
        _cuda_atomic_add!(hyp_im, t2 + (r3 - 1) * p1_dof_count, scratch[1 + 32 * stride])
        _cuda_atomic_add!(hyp_im, t3 + (r1 - 1) * p1_dof_count, scratch[1 + 33 * stride])
        _cuda_atomic_add!(hyp_im, t3 + (r2 - 1) * p1_dof_count, scratch[1 + 34 * stride])
        _cuda_atomic_add!(hyp_im, t3 + (r3 - 1) * p1_dof_count, scratch[1 + 35 * stride])
    end

    return nothing
end

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
)
    pair = blockIdx().x
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
    qpair = threadIdx().x
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

        qpair += blockDim().x
    end

    tid = threadIdx().x
    stride = blockDim().x
    scratch[tid + 0 * stride] = slp1_re
    scratch[tid + 1 * stride] = slp2_re
    scratch[tid + 2 * stride] = slp3_re
    scratch[tid + 3 * stride] = slp1_im
    scratch[tid + 4 * stride] = slp2_im
    scratch[tid + 5 * stride] = slp3_im
    scratch[tid + 6 * stride] = hyp11_re
    scratch[tid + 7 * stride] = hyp12_re
    scratch[tid + 8 * stride] = hyp13_re
    scratch[tid + 9 * stride] = hyp21_re
    scratch[tid + 10 * stride] = hyp22_re
    scratch[tid + 11 * stride] = hyp23_re
    scratch[tid + 12 * stride] = hyp31_re
    scratch[tid + 13 * stride] = hyp32_re
    scratch[tid + 14 * stride] = hyp33_re
    scratch[tid + 15 * stride] = hyp11_im
    scratch[tid + 16 * stride] = hyp12_im
    scratch[tid + 17 * stride] = hyp13_im
    scratch[tid + 18 * stride] = hyp21_im
    scratch[tid + 19 * stride] = hyp22_im
    scratch[tid + 20 * stride] = hyp23_im
    scratch[tid + 21 * stride] = hyp31_im
    scratch[tid + 22 * stride] = hyp32_im
    scratch[tid + 23 * stride] = hyp33_im
    sync_threads()

    offset = stride >>> 1
    while offset > 0
        if tid <= offset
            for slot in 0:23
                scratch[tid + slot * stride] += scratch[tid + offset + slot * stride]
            end
        end
        sync_threads()
        offset >>>= 1
    end

    if threadIdx().x == 1
        slp_col = trial_index
        _cuda_atomic_add!(slp_re, t1 + (slp_col - 1) * p1_dof_count, scratch[1 + 0 * stride])
        _cuda_atomic_add!(slp_re, t2 + (slp_col - 1) * p1_dof_count, scratch[1 + 1 * stride])
        _cuda_atomic_add!(slp_re, t3 + (slp_col - 1) * p1_dof_count, scratch[1 + 2 * stride])
        _cuda_atomic_add!(slp_im, t1 + (slp_col - 1) * p1_dof_count, scratch[1 + 3 * stride])
        _cuda_atomic_add!(slp_im, t2 + (slp_col - 1) * p1_dof_count, scratch[1 + 4 * stride])
        _cuda_atomic_add!(slp_im, t3 + (slp_col - 1) * p1_dof_count, scratch[1 + 5 * stride])

        _cuda_atomic_add!(hyp_re, t1 + (r1 - 1) * p1_dof_count, scratch[1 + 6 * stride])
        _cuda_atomic_add!(hyp_re, t1 + (r2 - 1) * p1_dof_count, scratch[1 + 7 * stride])
        _cuda_atomic_add!(hyp_re, t1 + (r3 - 1) * p1_dof_count, scratch[1 + 8 * stride])
        _cuda_atomic_add!(hyp_re, t2 + (r1 - 1) * p1_dof_count, scratch[1 + 9 * stride])
        _cuda_atomic_add!(hyp_re, t2 + (r2 - 1) * p1_dof_count, scratch[1 + 10 * stride])
        _cuda_atomic_add!(hyp_re, t2 + (r3 - 1) * p1_dof_count, scratch[1 + 11 * stride])
        _cuda_atomic_add!(hyp_re, t3 + (r1 - 1) * p1_dof_count, scratch[1 + 12 * stride])
        _cuda_atomic_add!(hyp_re, t3 + (r2 - 1) * p1_dof_count, scratch[1 + 13 * stride])
        _cuda_atomic_add!(hyp_re, t3 + (r3 - 1) * p1_dof_count, scratch[1 + 14 * stride])
        _cuda_atomic_add!(hyp_im, t1 + (r1 - 1) * p1_dof_count, scratch[1 + 15 * stride])
        _cuda_atomic_add!(hyp_im, t1 + (r2 - 1) * p1_dof_count, scratch[1 + 16 * stride])
        _cuda_atomic_add!(hyp_im, t1 + (r3 - 1) * p1_dof_count, scratch[1 + 17 * stride])
        _cuda_atomic_add!(hyp_im, t2 + (r1 - 1) * p1_dof_count, scratch[1 + 18 * stride])
        _cuda_atomic_add!(hyp_im, t2 + (r2 - 1) * p1_dof_count, scratch[1 + 19 * stride])
        _cuda_atomic_add!(hyp_im, t2 + (r3 - 1) * p1_dof_count, scratch[1 + 20 * stride])
        _cuda_atomic_add!(hyp_im, t3 + (r1 - 1) * p1_dof_count, scratch[1 + 21 * stride])
        _cuda_atomic_add!(hyp_im, t3 + (r2 - 1) * p1_dof_count, scratch[1 + 22 * stride])
        _cuda_atomic_add!(hyp_im, t3 + (r3 - 1) * p1_dof_count, scratch[1 + 23 * stride])
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
)
    pair = blockIdx().x
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
    qpair = threadIdx().x
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

        qpair += blockDim().x
    end

    tid = threadIdx().x
    stride = blockDim().x
    scratch[tid + 0 * stride] = dlp11_re
    scratch[tid + 1 * stride] = dlp12_re
    scratch[tid + 2 * stride] = dlp13_re
    scratch[tid + 3 * stride] = dlp21_re
    scratch[tid + 4 * stride] = dlp22_re
    scratch[tid + 5 * stride] = dlp23_re
    scratch[tid + 6 * stride] = dlp31_re
    scratch[tid + 7 * stride] = dlp32_re
    scratch[tid + 8 * stride] = dlp33_re
    scratch[tid + 9 * stride] = dlp11_im
    scratch[tid + 10 * stride] = dlp12_im
    scratch[tid + 11 * stride] = dlp13_im
    scratch[tid + 12 * stride] = dlp21_im
    scratch[tid + 13 * stride] = dlp22_im
    scratch[tid + 14 * stride] = dlp23_im
    scratch[tid + 15 * stride] = dlp31_im
    scratch[tid + 16 * stride] = dlp32_im
    scratch[tid + 17 * stride] = dlp33_im
    scratch[tid + 18 * stride] = adj1_re
    scratch[tid + 19 * stride] = adj2_re
    scratch[tid + 20 * stride] = adj3_re
    scratch[tid + 21 * stride] = adj1_im
    scratch[tid + 22 * stride] = adj2_im
    scratch[tid + 23 * stride] = adj3_im
    sync_threads()

    offset = stride >>> 1
    while offset > 0
        if tid <= offset
            for slot in 0:23
                scratch[tid + slot * stride] += scratch[tid + offset + slot * stride]
            end
        end
        sync_threads()
        offset >>>= 1
    end

    if threadIdx().x == 1
        _cuda_atomic_add!(dlp_re, t1 + (r1 - 1) * p1_dof_count, scratch[1 + 0 * stride])
        _cuda_atomic_add!(dlp_re, t1 + (r2 - 1) * p1_dof_count, scratch[1 + 1 * stride])
        _cuda_atomic_add!(dlp_re, t1 + (r3 - 1) * p1_dof_count, scratch[1 + 2 * stride])
        _cuda_atomic_add!(dlp_re, t2 + (r1 - 1) * p1_dof_count, scratch[1 + 3 * stride])
        _cuda_atomic_add!(dlp_re, t2 + (r2 - 1) * p1_dof_count, scratch[1 + 4 * stride])
        _cuda_atomic_add!(dlp_re, t2 + (r3 - 1) * p1_dof_count, scratch[1 + 5 * stride])
        _cuda_atomic_add!(dlp_re, t3 + (r1 - 1) * p1_dof_count, scratch[1 + 6 * stride])
        _cuda_atomic_add!(dlp_re, t3 + (r2 - 1) * p1_dof_count, scratch[1 + 7 * stride])
        _cuda_atomic_add!(dlp_re, t3 + (r3 - 1) * p1_dof_count, scratch[1 + 8 * stride])
        _cuda_atomic_add!(dlp_im, t1 + (r1 - 1) * p1_dof_count, scratch[1 + 9 * stride])
        _cuda_atomic_add!(dlp_im, t1 + (r2 - 1) * p1_dof_count, scratch[1 + 10 * stride])
        _cuda_atomic_add!(dlp_im, t1 + (r3 - 1) * p1_dof_count, scratch[1 + 11 * stride])
        _cuda_atomic_add!(dlp_im, t2 + (r1 - 1) * p1_dof_count, scratch[1 + 12 * stride])
        _cuda_atomic_add!(dlp_im, t2 + (r2 - 1) * p1_dof_count, scratch[1 + 13 * stride])
        _cuda_atomic_add!(dlp_im, t2 + (r3 - 1) * p1_dof_count, scratch[1 + 14 * stride])
        _cuda_atomic_add!(dlp_im, t3 + (r1 - 1) * p1_dof_count, scratch[1 + 15 * stride])
        _cuda_atomic_add!(dlp_im, t3 + (r2 - 1) * p1_dof_count, scratch[1 + 16 * stride])
        _cuda_atomic_add!(dlp_im, t3 + (r3 - 1) * p1_dof_count, scratch[1 + 17 * stride])

        adj_col = trial_index
        _cuda_atomic_add!(adj_re, t1 + (adj_col - 1) * p1_dof_count, scratch[1 + 18 * stride])
        _cuda_atomic_add!(adj_re, t2 + (adj_col - 1) * p1_dof_count, scratch[1 + 19 * stride])
        _cuda_atomic_add!(adj_re, t3 + (adj_col - 1) * p1_dof_count, scratch[1 + 20 * stride])
        _cuda_atomic_add!(adj_im, t1 + (adj_col - 1) * p1_dof_count, scratch[1 + 21 * stride])
        _cuda_atomic_add!(adj_im, t2 + (adj_col - 1) * p1_dof_count, scratch[1 + 22 * stride])
        _cuda_atomic_add!(adj_im, t3 + (adj_col - 1) * p1_dof_count, scratch[1 + 23 * stride])
    end

    return nothing
end

function _profile_regular_thread_sweep!(
    timing,
    ::Type{T},
    p1_dof_count::Int,
    dp0_dof_count::Int,
    d_face_vertices,
    d_normals,
    d_areas,
    d_faces,
    d_curls,
    d_test_indices,
    d_trial_indices,
    d_rule_points,
    d_rule_weights,
    k::T,
    face_count::Int,
    rule_count::Int,
    total_pairs::Int,
) where {T<:AbstractFloat}
    buffers = _cuda_regular_real_buffers(T, p1_dof_count, dp0_dof_count)
    try
        CUDA.synchronize()
        for threads in (32, 64, 128)
            _cuda_fill_regular_real_buffers!(buffers, zero(T))
            shmem = threads * 48 * sizeof(T)
            _cuda_timed_stage!(timing, "regular_operator_thread_sweep_$(threads)") do
                CUDA.@cuda threads=threads blocks=total_pairs shmem=shmem _cuda_regular_quadrature_kernel!(
                    buffers.slp_re,
                    buffers.slp_im,
                    buffers.dlp_re,
                    buffers.dlp_im,
                    buffers.adj_re,
                    buffers.adj_im,
                    buffers.hyp_re,
                    buffers.hyp_im,
                    d_face_vertices,
                    d_normals,
                    d_areas,
                    d_faces,
                    d_curls,
                    d_test_indices,
                    d_trial_indices,
                    d_rule_points,
                    d_rule_weights,
                    k,
                    p1_dof_count,
                    dp0_dof_count,
                    face_count,
                    rule_count,
                    total_pairs,
                )
                CUDA.synchronize()
                nothing
            end
        end
    finally
        _cuda_free_regular_real_buffers!(buffers)
    end
    return nothing
end

function _profile_regular_quadrature_probes!(
    timing,
    d_face_vertices,
    d_normals,
    d_areas,
    d_faces,
    d_curls,
    d_test_indices,
    d_trial_indices,
    d_rule_points,
    d_rule_weights,
    k::T,
    face_count::Int,
    rule_count::Int,
    total_pairs::Int,
    threads::Int,
) where {T<:AbstractFloat}
    sink = CUDA.zeros(T, 1)
    shmem = threads * sizeof(T)
    try
        probe_specs = (
            ("regular_operator_probe_green_kernel", Int32(1)),
            ("regular_operator_probe_all_terms_kernel", Int32(2)),
            ("regular_operator_probe_slp_kernel", Int32(3)),
            ("regular_operator_probe_adjoint_kernel", Int32(4)),
            ("regular_operator_probe_dlp_kernel", Int32(5)),
            ("regular_operator_probe_hypersingular_kernel", Int32(6)),
        )
        for (name, mode) in probe_specs
            fill!(sink, zero(T))
            CUDA.synchronize()
            _cuda_timed_stage!(timing, name) do
                CUDA.@cuda threads=threads blocks=total_pairs shmem=shmem _cuda_regular_quadrature_probe_kernel!(
                    sink,
                    d_face_vertices,
                    d_normals,
                    d_areas,
                    d_faces,
                    d_curls,
                    d_test_indices,
                    d_trial_indices,
                    d_rule_points,
                    d_rule_weights,
                    k,
                    face_count,
                    rule_count,
                    total_pairs,
                    mode,
                )
                CUDA.synchronize()
                nothing
            end
        end
        timing !== nothing && (timing["regular_operator_probe_reduced_accumulator_kernel"] = timing["regular_operator_probe_all_terms_kernel"])
    finally
        CUDA.unsafe_free!(sink)
    end
    return nothing
end

function _profile_regular_slp_adjoint_colored!(
    timing,
    ::Type{T},
    p1_dof_count::Int,
    dp0_dof_count::Int,
    d_face_vertices,
    d_normals,
    d_areas,
    d_faces,
    d_color_indices,
    d_color_offsets,
    d_trial_indices,
    d_rule_points,
    d_rule_weights,
    k::T,
    face_count::Int,
    rule_count::Int,
    color_count::Int,
    threads::Int,
    pair_limit::Int,
) where {T<:AbstractFloat}
    slp_re = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
    slp_im = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
    adj_re = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
    adj_im = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
    color_offsets = Array(d_color_offsets)
    full_pair_count = length(d_color_indices) * length(d_trial_indices)
    trial_count = length(d_trial_indices)
    if pair_limit > 0 && pair_limit < full_pair_count
        trial_count = max(1, min(trial_count, cld(pair_limit, length(d_color_indices))))
    end
    shmem = threads * 12 * sizeof(T)
    try
        _cuda_timed_stage!(timing, "regular_operator_probe_slp_adjoint_colored_kernel") do
            for color in 1:color_count
                first_index = Int(color_offsets[color])
                last_index = Int(color_offsets[color + 1]) - 1
                first_index <= last_index || continue
                test_count = last_index - first_index + 1
                total_pairs = test_count * trial_count
                CUDA.@cuda threads=threads blocks=total_pairs shmem=shmem _cuda_regular_quadrature_slp_adjoint_kernel!(
                    slp_re,
                    slp_im,
                    adj_re,
                    adj_im,
                    d_face_vertices,
                    d_normals,
                    d_areas,
                    d_faces,
                    d_color_indices,
                    first_index,
                    test_count,
                    d_trial_indices,
                    trial_count,
                    d_rule_points,
                    d_rule_weights,
                    k,
                    p1_dof_count,
                    face_count,
                    rule_count,
                    total_pairs,
                    false,
                )
            end
            CUDA.synchronize()
            nothing
        end
    finally
        CUDA.unsafe_free!(slp_re)
        CUDA.unsafe_free!(slp_im)
        CUDA.unsafe_free!(adj_re)
        CUDA.unsafe_free!(adj_im)
    end
    return nothing
end

function _profile_regular_split_atomic!(
    timing,
    ::Type{T},
    p1_dof_count::Int,
    dp0_dof_count::Int,
    d_face_vertices,
    d_normals,
    d_areas,
    d_faces,
    d_curls,
    d_test_indices,
    d_trial_indices,
    d_rule_points,
    d_rule_weights,
    k::T,
    face_count::Int,
    rule_count::Int,
    total_pairs::Int,
    threads::Int,
) where {T<:AbstractFloat}
    buffers = _cuda_regular_real_buffers(T, p1_dof_count, dp0_dof_count)
    try
        _cuda_fill_regular_real_buffers!(buffers, zero(T))
        slp_shmem = threads * 12 * sizeof(T)
        _cuda_timed_stage!(timing, "regular_operator_probe_split_slp_adjoint_atomic_kernel") do
            CUDA.@cuda threads=threads blocks=total_pairs shmem=slp_shmem _cuda_regular_quadrature_slp_adjoint_kernel!(
                buffers.slp_re,
                buffers.slp_im,
                buffers.adj_re,
                buffers.adj_im,
                d_face_vertices,
                d_normals,
                d_areas,
                d_faces,
                d_test_indices,
                1,
                length(d_test_indices),
                d_trial_indices,
                length(d_trial_indices),
                d_rule_points,
                d_rule_weights,
                k,
                p1_dof_count,
                face_count,
                rule_count,
                total_pairs,
                true,
            )
            CUDA.synchronize()
            nothing
        end

        _cuda_fill_regular_real_buffers!(buffers, zero(T))
        dlp_shmem = threads * 36 * sizeof(T)
        _cuda_timed_stage!(timing, "regular_operator_probe_split_dlp_hyp_atomic_kernel") do
            CUDA.@cuda threads=threads blocks=total_pairs shmem=dlp_shmem _cuda_regular_quadrature_dlp_hyp_kernel!(
                buffers.dlp_re,
                buffers.dlp_im,
                buffers.hyp_re,
                buffers.hyp_im,
                d_face_vertices,
                d_normals,
                d_areas,
                d_faces,
                d_curls,
                d_test_indices,
                d_trial_indices,
                d_rule_points,
                d_rule_weights,
                k,
                p1_dof_count,
                face_count,
                rule_count,
                total_pairs,
            )
            CUDA.synchronize()
            nothing
        end

        if timing !== nothing
            timing["regular_operator_probe_split_atomic_total"] =
                timing["regular_operator_probe_split_slp_adjoint_atomic_kernel"] +
                timing["regular_operator_probe_split_dlp_hyp_atomic_kernel"]
        end

        _cuda_fill_regular_real_buffers!(buffers, zero(T))
        balanced_shmem = threads * 24 * sizeof(T)
        _cuda_timed_stage!(timing, "regular_operator_probe_split_slp_hyp_atomic_kernel") do
            CUDA.@cuda threads=threads blocks=total_pairs shmem=balanced_shmem _cuda_regular_quadrature_slp_hyp_kernel!(
                buffers.slp_re,
                buffers.slp_im,
                buffers.hyp_re,
                buffers.hyp_im,
                d_face_vertices,
                d_normals,
                d_areas,
                d_faces,
                d_curls,
                d_test_indices,
                d_trial_indices,
                d_rule_points,
                d_rule_weights,
                k,
                p1_dof_count,
                face_count,
                rule_count,
                total_pairs,
            )
            CUDA.synchronize()
            nothing
        end

        _cuda_fill_regular_real_buffers!(buffers, zero(T))
        _cuda_timed_stage!(timing, "regular_operator_probe_split_dlp_adjoint_atomic_kernel") do
            CUDA.@cuda threads=threads blocks=total_pairs shmem=balanced_shmem _cuda_regular_quadrature_dlp_adjoint_kernel!(
                buffers.dlp_re,
                buffers.dlp_im,
                buffers.adj_re,
                buffers.adj_im,
                d_face_vertices,
                d_normals,
                d_areas,
                d_faces,
                d_test_indices,
                d_trial_indices,
                d_rule_points,
                d_rule_weights,
                k,
                p1_dof_count,
                face_count,
                rule_count,
                total_pairs,
            )
            CUDA.synchronize()
            nothing
        end

        if timing !== nothing
            timing["regular_operator_probe_split_balanced_atomic_total"] =
                timing["regular_operator_probe_split_slp_hyp_atomic_kernel"] +
                timing["regular_operator_probe_split_dlp_adjoint_atomic_kernel"]
        end
    finally
        _cuda_free_regular_real_buffers!(buffers)
    end
    return nothing
end

function _zero_extended_regular_profile_timings!(timing)
    timing === nothing && return nothing
    for name in (
        "regular_operator_thread_sweep_32",
        "regular_operator_thread_sweep_64",
        "regular_operator_thread_sweep_128",
        "regular_operator_probe_slp_kernel",
        "regular_operator_probe_adjoint_kernel",
        "regular_operator_probe_dlp_kernel",
        "regular_operator_probe_hypersingular_kernel",
        "regular_operator_probe_reduced_accumulator_kernel",
        "regular_operator_probe_slp_adjoint_colored_kernel",
        "regular_operator_probe_split_slp_adjoint_atomic_kernel",
        "regular_operator_probe_split_dlp_hyp_atomic_kernel",
        "regular_operator_probe_split_atomic_total",
        "regular_operator_probe_split_slp_hyp_atomic_kernel",
        "regular_operator_probe_split_dlp_adjoint_atomic_kernel",
        "regular_operator_probe_split_balanced_atomic_total",
    )
        timing[name] = 0.0
    end
    return nothing
end
