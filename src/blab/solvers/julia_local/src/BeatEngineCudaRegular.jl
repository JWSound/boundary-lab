function _cuda_regular_kernel!(
    slp_re,
    slp_im,
    dlp_re,
    dlp_im,
    adj_re,
    adj_im,
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
    dp0_dof_count,
    face_count,
    rule_count,
    total_pairs,
    skip_adjacent,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    pair = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    four_pi = typeof(k)(12.566370614359172)

    while pair <= total_pairs
        test_loop_index = ((pair - 1) % length(test_indices)) + 1
        trial_loop_index = ((pair - 1) ÷ length(test_indices)) + 1
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

        if !skip_adjacent || !adjacent
            tv1x = face_vertices[test_index]
            tv1y = face_vertices[test_index + face_count]
            tv1z = face_vertices[test_index + 2 * face_count]
            tv2x = face_vertices[test_index + 3 * face_count]
            tv2y = face_vertices[test_index + 4 * face_count]
            tv2z = face_vertices[test_index + 5 * face_count]
            tv3x = face_vertices[test_index + 6 * face_count]
            tv3y = face_vertices[test_index + 7 * face_count]
            tv3z = face_vertices[test_index + 8 * face_count]

            rv1x = trial_sign_x * face_vertices[trial_index]
            rv1y = trial_sign_y * face_vertices[trial_index + face_count]
            rv1z = trial_sign_z * face_vertices[trial_index + 2 * face_count]
            rv2x = trial_sign_x * face_vertices[trial_index + 3 * face_count]
            rv2y = trial_sign_y * face_vertices[trial_index + 4 * face_count]
            rv2z = trial_sign_z * face_vertices[trial_index + 5 * face_count]
            rv3x = trial_sign_x * face_vertices[trial_index + 6 * face_count]
            rv3y = trial_sign_y * face_vertices[trial_index + 7 * face_count]
            rv3z = trial_sign_z * face_vertices[trial_index + 8 * face_count]

            tnx = normals[test_index]
            tny = normals[test_index + face_count]
            tnz = normals[test_index + 2 * face_count]
            rnx = trial_sign_x * normals[trial_index]
            rny = trial_sign_y * normals[trial_index + face_count]
            rnz = trial_sign_z * normals[trial_index + 2 * face_count]
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

            rc11 = trial_curl_sign_x * curls[trial_index]
            rc12 = trial_curl_sign_y * curls[trial_index + face_count]
            rc13 = trial_curl_sign_z * curls[trial_index + 2 * face_count]
            rc21 = trial_curl_sign_x * curls[trial_index + 3 * face_count]
            rc22 = trial_curl_sign_y * curls[trial_index + 4 * face_count]
            rc23 = trial_curl_sign_z * curls[trial_index + 5 * face_count]
            rc31 = trial_curl_sign_x * curls[trial_index + 6 * face_count]
            rc32 = trial_curl_sign_y * curls[trial_index + 7 * face_count]
            rc33 = trial_curl_sign_z * curls[trial_index + 8 * face_count]

            jac_scale = typeof(k)(4) * areas[test_index] * areas[trial_index]

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

            dlp11_re = zero(k)
            dlp12_re = zero(k)
            dlp13_re = zero(k)
            dlp21_re = zero(k)
            dlp22_re = zero(k)
            dlp23_re = zero(k)
            dlp31_re = zero(k)
            dlp32_re = zero(k)
            dlp33_re = zero(k)
            dlp11_im = zero(k)
            dlp12_im = zero(k)
            dlp13_im = zero(k)
            dlp21_im = zero(k)
            dlp22_im = zero(k)
            dlp23_im = zero(k)
            dlp31_im = zero(k)
            dlp32_im = zero(k)
            dlp33_im = zero(k)

            hyp11_re = zero(k)
            hyp12_re = zero(k)
            hyp13_re = zero(k)
            hyp21_re = zero(k)
            hyp22_re = zero(k)
            hyp23_re = zero(k)
            hyp31_re = zero(k)
            hyp32_re = zero(k)
            hyp33_re = zero(k)
            hyp11_im = zero(k)
            hyp12_im = zero(k)
            hyp13_im = zero(k)
            hyp21_im = zero(k)
            hyp22_im = zero(k)
            hyp23_im = zero(k)
            hyp31_im = zero(k)
            hyp32_im = zero(k)
            hyp33_im = zero(k)

            for tq in 1:rule_count
                txi = rule_points[tq]
                teta = rule_points[tq + rule_count]
                tw = rule_weights[tq]
                tv1 = one(k) - txi - teta
                tv2 = txi
                tv3 = teta
                x = tv1 * tv1x + tv2 * tv2x + tv3 * tv3x
                y = tv1 * tv1y + tv2 * tv2y + tv3 * tv3y
                z = tv1 * tv1z + tv2 * tv2z + tv3 * tv3z

                for rq in 1:rule_count
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
                        test_projection = -(dx * tnx + dy * tny + dz * tnz) * inv_radius
                        factor_re = -inv_radius
                        factor_im = k
                        deriv_re = green_re * factor_re - green_im * factor_im
                        deriv_im = green_re * factor_im + green_im * factor_re
                        dlp_value_re = deriv_re * source_projection * weight
                        dlp_value_im = deriv_im * source_projection * weight
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
                end
            end

            slp_col = trial_index
            adj_col = trial_index
            dlp_col1 = r1
            dlp_col2 = r2
            dlp_col3 = r3
            row1 = t1
            row2 = t2
            row3 = t3

            _cuda_atomic_add!(slp_re, row1 + (slp_col - 1) * p1_dof_count, slp1_re)
            _cuda_atomic_add!(slp_re, row2 + (slp_col - 1) * p1_dof_count, slp2_re)
            _cuda_atomic_add!(slp_re, row3 + (slp_col - 1) * p1_dof_count, slp3_re)
            _cuda_atomic_add!(slp_im, row1 + (slp_col - 1) * p1_dof_count, slp1_im)
            _cuda_atomic_add!(slp_im, row2 + (slp_col - 1) * p1_dof_count, slp2_im)
            _cuda_atomic_add!(slp_im, row3 + (slp_col - 1) * p1_dof_count, slp3_im)

            _cuda_atomic_add!(adj_re, row1 + (adj_col - 1) * p1_dof_count, adj1_re)
            _cuda_atomic_add!(adj_re, row2 + (adj_col - 1) * p1_dof_count, adj2_re)
            _cuda_atomic_add!(adj_re, row3 + (adj_col - 1) * p1_dof_count, adj3_re)
            _cuda_atomic_add!(adj_im, row1 + (adj_col - 1) * p1_dof_count, adj1_im)
            _cuda_atomic_add!(adj_im, row2 + (adj_col - 1) * p1_dof_count, adj2_im)
            _cuda_atomic_add!(adj_im, row3 + (adj_col - 1) * p1_dof_count, adj3_im)

            _cuda_atomic_add!(dlp_re, row1 + (dlp_col1 - 1) * p1_dof_count, dlp11_re)
            _cuda_atomic_add!(dlp_re, row1 + (dlp_col2 - 1) * p1_dof_count, dlp12_re)
            _cuda_atomic_add!(dlp_re, row1 + (dlp_col3 - 1) * p1_dof_count, dlp13_re)
            _cuda_atomic_add!(dlp_re, row2 + (dlp_col1 - 1) * p1_dof_count, dlp21_re)
            _cuda_atomic_add!(dlp_re, row2 + (dlp_col2 - 1) * p1_dof_count, dlp22_re)
            _cuda_atomic_add!(dlp_re, row2 + (dlp_col3 - 1) * p1_dof_count, dlp23_re)
            _cuda_atomic_add!(dlp_re, row3 + (dlp_col1 - 1) * p1_dof_count, dlp31_re)
            _cuda_atomic_add!(dlp_re, row3 + (dlp_col2 - 1) * p1_dof_count, dlp32_re)
            _cuda_atomic_add!(dlp_re, row3 + (dlp_col3 - 1) * p1_dof_count, dlp33_re)
            _cuda_atomic_add!(dlp_im, row1 + (dlp_col1 - 1) * p1_dof_count, dlp11_im)
            _cuda_atomic_add!(dlp_im, row1 + (dlp_col2 - 1) * p1_dof_count, dlp12_im)
            _cuda_atomic_add!(dlp_im, row1 + (dlp_col3 - 1) * p1_dof_count, dlp13_im)
            _cuda_atomic_add!(dlp_im, row2 + (dlp_col1 - 1) * p1_dof_count, dlp21_im)
            _cuda_atomic_add!(dlp_im, row2 + (dlp_col2 - 1) * p1_dof_count, dlp22_im)
            _cuda_atomic_add!(dlp_im, row2 + (dlp_col3 - 1) * p1_dof_count, dlp23_im)
            _cuda_atomic_add!(dlp_im, row3 + (dlp_col1 - 1) * p1_dof_count, dlp31_im)
            _cuda_atomic_add!(dlp_im, row3 + (dlp_col2 - 1) * p1_dof_count, dlp32_im)
            _cuda_atomic_add!(dlp_im, row3 + (dlp_col3 - 1) * p1_dof_count, dlp33_im)

            _cuda_atomic_add!(hyp_re, row1 + (dlp_col1 - 1) * p1_dof_count, hyp11_re)
            _cuda_atomic_add!(hyp_re, row1 + (dlp_col2 - 1) * p1_dof_count, hyp12_re)
            _cuda_atomic_add!(hyp_re, row1 + (dlp_col3 - 1) * p1_dof_count, hyp13_re)
            _cuda_atomic_add!(hyp_re, row2 + (dlp_col1 - 1) * p1_dof_count, hyp21_re)
            _cuda_atomic_add!(hyp_re, row2 + (dlp_col2 - 1) * p1_dof_count, hyp22_re)
            _cuda_atomic_add!(hyp_re, row2 + (dlp_col3 - 1) * p1_dof_count, hyp23_re)
            _cuda_atomic_add!(hyp_re, row3 + (dlp_col1 - 1) * p1_dof_count, hyp31_re)
            _cuda_atomic_add!(hyp_re, row3 + (dlp_col2 - 1) * p1_dof_count, hyp32_re)
            _cuda_atomic_add!(hyp_re, row3 + (dlp_col3 - 1) * p1_dof_count, hyp33_re)
            _cuda_atomic_add!(hyp_im, row1 + (dlp_col1 - 1) * p1_dof_count, hyp11_im)
            _cuda_atomic_add!(hyp_im, row1 + (dlp_col2 - 1) * p1_dof_count, hyp12_im)
            _cuda_atomic_add!(hyp_im, row1 + (dlp_col3 - 1) * p1_dof_count, hyp13_im)
            _cuda_atomic_add!(hyp_im, row2 + (dlp_col1 - 1) * p1_dof_count, hyp21_im)
            _cuda_atomic_add!(hyp_im, row2 + (dlp_col2 - 1) * p1_dof_count, hyp22_im)
            _cuda_atomic_add!(hyp_im, row2 + (dlp_col3 - 1) * p1_dof_count, hyp23_im)
            _cuda_atomic_add!(hyp_im, row3 + (dlp_col1 - 1) * p1_dof_count, hyp31_im)
            _cuda_atomic_add!(hyp_im, row3 + (dlp_col2 - 1) * p1_dof_count, hyp32_im)
            _cuda_atomic_add!(hyp_im, row3 + (dlp_col3 - 1) * p1_dof_count, hyp33_im)
        end

        pair += stride
    end

    return nothing
end

function _cuda_regular_quadrature_kernel!(
    slp_re,
    slp_im,
    dlp_re,
    dlp_im,
    adj_re,
    adj_im,
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
    dp0_dof_count,
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
    accumulator_count = 48
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

    dlp11_re = zero(k)
    dlp12_re = zero(k)
    dlp13_re = zero(k)
    dlp21_re = zero(k)
    dlp22_re = zero(k)
    dlp23_re = zero(k)
    dlp31_re = zero(k)
    dlp32_re = zero(k)
    dlp33_re = zero(k)
    dlp11_im = zero(k)
    dlp12_im = zero(k)
    dlp13_im = zero(k)
    dlp21_im = zero(k)
    dlp22_im = zero(k)
    dlp23_im = zero(k)
    dlp31_im = zero(k)
    dlp32_im = zero(k)
    dlp33_im = zero(k)

    hyp11_re = zero(k)
    hyp12_re = zero(k)
    hyp13_re = zero(k)
    hyp21_re = zero(k)
    hyp22_re = zero(k)
    hyp23_re = zero(k)
    hyp31_re = zero(k)
    hyp32_re = zero(k)
    hyp33_re = zero(k)
    hyp11_im = zero(k)
    hyp12_im = zero(k)
    hyp13_im = zero(k)
    hyp21_im = zero(k)
    hyp22_im = zero(k)
    hyp23_im = zero(k)
    hyp31_im = zero(k)
    hyp32_im = zero(k)
    hyp33_im = zero(k)

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
            test_projection = -(dx * tnx + dy * tny + dz * tnz) * inv_radius
            factor_re = -inv_radius
            factor_im = k
            deriv_re = green_re * factor_re - green_im * factor_im
            deriv_im = green_re * factor_im + green_im * factor_re
            dlp_value_re = deriv_re * source_projection * weight
            dlp_value_im = deriv_im * source_projection * weight
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
    scratch[tid + 12 * stride] = dlp11_re
    scratch[tid + 13 * stride] = dlp12_re
    scratch[tid + 14 * stride] = dlp13_re
    scratch[tid + 15 * stride] = dlp21_re
    scratch[tid + 16 * stride] = dlp22_re
    scratch[tid + 17 * stride] = dlp23_re
    scratch[tid + 18 * stride] = dlp31_re
    scratch[tid + 19 * stride] = dlp32_re
    scratch[tid + 20 * stride] = dlp33_re
    scratch[tid + 21 * stride] = dlp11_im
    scratch[tid + 22 * stride] = dlp12_im
    scratch[tid + 23 * stride] = dlp13_im
    scratch[tid + 24 * stride] = dlp21_im
    scratch[tid + 25 * stride] = dlp22_im
    scratch[tid + 26 * stride] = dlp23_im
    scratch[tid + 27 * stride] = dlp31_im
    scratch[tid + 28 * stride] = dlp32_im
    scratch[tid + 29 * stride] = dlp33_im
    scratch[tid + 30 * stride] = hyp11_re
    scratch[tid + 31 * stride] = hyp12_re
    scratch[tid + 32 * stride] = hyp13_re
    scratch[tid + 33 * stride] = hyp21_re
    scratch[tid + 34 * stride] = hyp22_re
    scratch[tid + 35 * stride] = hyp23_re
    scratch[tid + 36 * stride] = hyp31_re
    scratch[tid + 37 * stride] = hyp32_re
    scratch[tid + 38 * stride] = hyp33_re
    scratch[tid + 39 * stride] = hyp11_im
    scratch[tid + 40 * stride] = hyp12_im
    scratch[tid + 41 * stride] = hyp13_im
    scratch[tid + 42 * stride] = hyp21_im
    scratch[tid + 43 * stride] = hyp22_im
    scratch[tid + 44 * stride] = hyp23_im
    scratch[tid + 45 * stride] = hyp31_im
    scratch[tid + 46 * stride] = hyp32_im
    scratch[tid + 47 * stride] = hyp33_im
    sync_threads()

    offset = stride >>> 1
    while offset > 0
        if tid <= offset
            for slot in 0:47
                scratch[tid + slot * stride] += scratch[tid + offset + slot * stride]
            end
        end
        sync_threads()
        offset >>>= 1
    end

    slp1_re = scratch[1 + 0 * stride]
    slp2_re = scratch[1 + 1 * stride]
    slp3_re = scratch[1 + 2 * stride]
    slp1_im = scratch[1 + 3 * stride]
    slp2_im = scratch[1 + 4 * stride]
    slp3_im = scratch[1 + 5 * stride]
    adj1_re = scratch[1 + 6 * stride]
    adj2_re = scratch[1 + 7 * stride]
    adj3_re = scratch[1 + 8 * stride]
    adj1_im = scratch[1 + 9 * stride]
    adj2_im = scratch[1 + 10 * stride]
    adj3_im = scratch[1 + 11 * stride]
    dlp11_re = scratch[1 + 12 * stride]
    dlp12_re = scratch[1 + 13 * stride]
    dlp13_re = scratch[1 + 14 * stride]
    dlp21_re = scratch[1 + 15 * stride]
    dlp22_re = scratch[1 + 16 * stride]
    dlp23_re = scratch[1 + 17 * stride]
    dlp31_re = scratch[1 + 18 * stride]
    dlp32_re = scratch[1 + 19 * stride]
    dlp33_re = scratch[1 + 20 * stride]
    dlp11_im = scratch[1 + 21 * stride]
    dlp12_im = scratch[1 + 22 * stride]
    dlp13_im = scratch[1 + 23 * stride]
    dlp21_im = scratch[1 + 24 * stride]
    dlp22_im = scratch[1 + 25 * stride]
    dlp23_im = scratch[1 + 26 * stride]
    dlp31_im = scratch[1 + 27 * stride]
    dlp32_im = scratch[1 + 28 * stride]
    dlp33_im = scratch[1 + 29 * stride]
    hyp11_re = scratch[1 + 30 * stride]
    hyp12_re = scratch[1 + 31 * stride]
    hyp13_re = scratch[1 + 32 * stride]
    hyp21_re = scratch[1 + 33 * stride]
    hyp22_re = scratch[1 + 34 * stride]
    hyp23_re = scratch[1 + 35 * stride]
    hyp31_re = scratch[1 + 36 * stride]
    hyp32_re = scratch[1 + 37 * stride]
    hyp33_re = scratch[1 + 38 * stride]
    hyp11_im = scratch[1 + 39 * stride]
    hyp12_im = scratch[1 + 40 * stride]
    hyp13_im = scratch[1 + 41 * stride]
    hyp21_im = scratch[1 + 42 * stride]
    hyp22_im = scratch[1 + 43 * stride]
    hyp23_im = scratch[1 + 44 * stride]
    hyp31_im = scratch[1 + 45 * stride]
    hyp32_im = scratch[1 + 46 * stride]
    hyp33_im = scratch[1 + 47 * stride]

    if threadIdx().x == 1
        row1 = t1
        row2 = t2
        row3 = t3
        slp_col = trial_index
        adj_col = trial_index

        _cuda_atomic_add!(slp_re, row1 + (slp_col - 1) * p1_dof_count, slp1_re)
        _cuda_atomic_add!(slp_re, row2 + (slp_col - 1) * p1_dof_count, slp2_re)
        _cuda_atomic_add!(slp_re, row3 + (slp_col - 1) * p1_dof_count, slp3_re)
        _cuda_atomic_add!(slp_im, row1 + (slp_col - 1) * p1_dof_count, slp1_im)
        _cuda_atomic_add!(slp_im, row2 + (slp_col - 1) * p1_dof_count, slp2_im)
        _cuda_atomic_add!(slp_im, row3 + (slp_col - 1) * p1_dof_count, slp3_im)
        _cuda_atomic_add!(adj_re, row1 + (adj_col - 1) * p1_dof_count, adj1_re)
        _cuda_atomic_add!(adj_re, row2 + (adj_col - 1) * p1_dof_count, adj2_re)
        _cuda_atomic_add!(adj_re, row3 + (adj_col - 1) * p1_dof_count, adj3_re)
        _cuda_atomic_add!(adj_im, row1 + (adj_col - 1) * p1_dof_count, adj1_im)
        _cuda_atomic_add!(adj_im, row2 + (adj_col - 1) * p1_dof_count, adj2_im)
        _cuda_atomic_add!(adj_im, row3 + (adj_col - 1) * p1_dof_count, adj3_im)

        _cuda_atomic_add!(dlp_re, row1 + (r1 - 1) * p1_dof_count, dlp11_re)
        _cuda_atomic_add!(dlp_re, row1 + (r2 - 1) * p1_dof_count, dlp12_re)
        _cuda_atomic_add!(dlp_re, row1 + (r3 - 1) * p1_dof_count, dlp13_re)
        _cuda_atomic_add!(dlp_re, row2 + (r1 - 1) * p1_dof_count, dlp21_re)
        _cuda_atomic_add!(dlp_re, row2 + (r2 - 1) * p1_dof_count, dlp22_re)
        _cuda_atomic_add!(dlp_re, row2 + (r3 - 1) * p1_dof_count, dlp23_re)
        _cuda_atomic_add!(dlp_re, row3 + (r1 - 1) * p1_dof_count, dlp31_re)
        _cuda_atomic_add!(dlp_re, row3 + (r2 - 1) * p1_dof_count, dlp32_re)
        _cuda_atomic_add!(dlp_re, row3 + (r3 - 1) * p1_dof_count, dlp33_re)
        _cuda_atomic_add!(dlp_im, row1 + (r1 - 1) * p1_dof_count, dlp11_im)
        _cuda_atomic_add!(dlp_im, row1 + (r2 - 1) * p1_dof_count, dlp12_im)
        _cuda_atomic_add!(dlp_im, row1 + (r3 - 1) * p1_dof_count, dlp13_im)
        _cuda_atomic_add!(dlp_im, row2 + (r1 - 1) * p1_dof_count, dlp21_im)
        _cuda_atomic_add!(dlp_im, row2 + (r2 - 1) * p1_dof_count, dlp22_im)
        _cuda_atomic_add!(dlp_im, row2 + (r3 - 1) * p1_dof_count, dlp23_im)
        _cuda_atomic_add!(dlp_im, row3 + (r1 - 1) * p1_dof_count, dlp31_im)
        _cuda_atomic_add!(dlp_im, row3 + (r2 - 1) * p1_dof_count, dlp32_im)
        _cuda_atomic_add!(dlp_im, row3 + (r3 - 1) * p1_dof_count, dlp33_im)

        _cuda_atomic_add!(hyp_re, row1 + (r1 - 1) * p1_dof_count, hyp11_re)
        _cuda_atomic_add!(hyp_re, row1 + (r2 - 1) * p1_dof_count, hyp12_re)
        _cuda_atomic_add!(hyp_re, row1 + (r3 - 1) * p1_dof_count, hyp13_re)
        _cuda_atomic_add!(hyp_re, row2 + (r1 - 1) * p1_dof_count, hyp21_re)
        _cuda_atomic_add!(hyp_re, row2 + (r2 - 1) * p1_dof_count, hyp22_re)
        _cuda_atomic_add!(hyp_re, row2 + (r3 - 1) * p1_dof_count, hyp23_re)
        _cuda_atomic_add!(hyp_re, row3 + (r1 - 1) * p1_dof_count, hyp31_re)
        _cuda_atomic_add!(hyp_re, row3 + (r2 - 1) * p1_dof_count, hyp32_re)
        _cuda_atomic_add!(hyp_re, row3 + (r3 - 1) * p1_dof_count, hyp33_re)
        _cuda_atomic_add!(hyp_im, row1 + (r1 - 1) * p1_dof_count, hyp11_im)
        _cuda_atomic_add!(hyp_im, row1 + (r2 - 1) * p1_dof_count, hyp12_im)
        _cuda_atomic_add!(hyp_im, row1 + (r3 - 1) * p1_dof_count, hyp13_im)
        _cuda_atomic_add!(hyp_im, row2 + (r1 - 1) * p1_dof_count, hyp21_im)
        _cuda_atomic_add!(hyp_im, row2 + (r2 - 1) * p1_dof_count, hyp22_im)
        _cuda_atomic_add!(hyp_im, row2 + (r3 - 1) * p1_dof_count, hyp23_im)
        _cuda_atomic_add!(hyp_im, row3 + (r1 - 1) * p1_dof_count, hyp31_im)
        _cuda_atomic_add!(hyp_im, row3 + (r2 - 1) * p1_dof_count, hyp32_im)
        _cuda_atomic_add!(hyp_im, row3 + (r3 - 1) * p1_dof_count, hyp33_im)
    end

    return nothing
end

function _cuda_geometry_arrays(mesh::BoundaryMesh{T}) where {T}
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

        for i in 1:3
            face_vertices[element_index, 3 * (i - 1) + 1] = vertices[i][1]
            face_vertices[element_index, 3 * (i - 1) + 2] = vertices[i][2]
            face_vertices[element_index, 3 * (i - 1) + 3] = vertices[i][3]
            normals[element_index, i] = normal[i]
            faces[element_index, i] = Int32(face[i])
            curls[element_index, 3 * (i - 1) + 1] = element_curls[i][1]
            curls[element_index, 3 * (i - 1) + 2] = element_curls[i][2]
            curls[element_index, 3 * (i - 1) + 3] = element_curls[i][3]
        end
    end

    return face_vertices, normals, areas, faces, curls
end

function _cuda_rule_arrays(rule::TriangleRule{T}) where {T}
    rule_count = length(rule.points)
    points = Matrix{T}(undef, rule_count, 2)
    weights = Vector{T}(undef, rule_count)
    for i in 1:rule_count
        points[i, 1] = rule.points[i][1]
        points[i, 2] = rule.points[i][2]
        weights[i] = rule.weights[i]
    end
    return points, weights
end

function _regular_face_color_arrays(mesh::BoundaryMesh, indices::Vector{Int})
    vertex_colors = [Int[] for _ in eachindex(mesh.vertices)]
    color_groups = Vector{Int}[]
    forbidden = Set{Int}()

    for element_index in indices
        empty!(forbidden)
        for vertex in mesh.faces[element_index]
            union!(forbidden, vertex_colors[vertex])
        end

        color = 1
        while color in forbidden
            color += 1
        end
        while length(color_groups) < color
            push!(color_groups, Int[])
        end

        push!(color_groups[color], element_index)
        for vertex in mesh.faces[element_index]
            push!(vertex_colors[vertex], color)
        end
    end

    offsets = Vector{Int32}(undef, length(color_groups) + 1)
    flat_indices = Int32[]
    offsets[1] = Int32(1)
    for (color_index, group) in enumerate(color_groups)
        append!(flat_indices, Int32.(group))
        offsets[color_index + 1] = Int32(length(flat_indices) + 1)
    end

    return flat_indices, offsets, length(color_groups)
end

function build_cuda_regular_assembly_cache(
    mesh::BoundaryMesh{T},
    rule::TriangleRule{T};
    element_indices=eachindex(mesh.faces),
) where {T<:AbstractFloat}
    CUDA.functional() || error("CUDA regular-pair assembly cache requested, but CUDA.functional() is false.")
    indices = collect(element_indices)
    face_vertices, normals, areas, faces, curls = _cuda_geometry_arrays(mesh)
    rule_points, rule_weights = _cuda_rule_arrays(rule)
    color_indices, color_offsets, color_count = _regular_face_color_arrays(mesh, indices)

    return CudaRegularAssemblyCache{T}(
        CuArray(face_vertices),
        CuArray(normals),
        CuArray(areas),
        CuArray(faces),
        CuArray(curls),
        CuArray(rule_points),
        CuArray(rule_weights),
        CuArray(Int32.(indices)),
        CuArray(Int32.(indices)),
        CuArray(color_indices),
        CuArray(color_offsets),
        indices,
        length(mesh.faces),
        length(rule.points),
        color_count,
    )
end
