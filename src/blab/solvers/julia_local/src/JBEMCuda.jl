import Pkg

required_packages = ["CUDA"]
for pkg in required_packages
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

using CUDA 

struct CudaRegularAssemblyCache{T}
    face_vertices
    normals
    areas
    faces
    curls
    rule_points
    rule_weights
    test_indices
    trial_indices
    color_indices
    color_offsets
    element_indices::Vector{Int}
    face_count::Int
    rule_count::Int
    color_count::Int
end

struct CudaFieldEvaluationCache{T}
    source_points
    source_normals
    source_weights
    source_faces
    source_elements
    basis_values
    source_count::Int
end

struct CudaSingularCorrectionCache{T}
    test_indices
    trial_indices
    rule_indices
    jac_scales
    normal_products
    p1_rows
    p1_cols
    dp0_cols
    rule_offsets
    rule_test_points
    rule_trial_points
    rule_weights
    pair_count::Int
end

struct CudaImageSingularCorrectionCache{T}
    test_indices
    trial_indices
    rule_indices
    jac_scales
    normal_products
    p1_rows
    p1_cols
    dp0_cols
    rule_offsets
    rule_test_points
    rule_trial_points
    rule_weights
    transform_signs
    curl_signs
    pair_count::Int
end

@inline function _cuda_atomic_add!(array, index, value)
    CUDA.@atomic array[index] += value
    return nothing
end

function _cuda_block_sum(value, scratch)
    tid = threadIdx().x
    scratch[tid] = value
    sync_threads()

    offset = blockDim().x >>> 1
    while offset > 0
        if tid <= offset
            scratch[tid] += scratch[tid + offset]
        end
        sync_threads()
        offset >>>= 1
    end

    return scratch[1]
end

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

function _cuda_field_arrays(cache::FieldEvaluationCache{T}) where {T}
    source_count = length(cache.source_points)
    source_points = Matrix{T}(undef, source_count, 3)
    source_normals = Matrix{T}(undef, source_count, 3)
    basis_values = Matrix{T}(undef, source_count, 3)
    source_weights = Vector{T}(undef, source_count)
    source_faces = Matrix{Int32}(undef, source_count, 3)
    source_elements = Vector{Int32}(undef, source_count)

    for source_index in 1:source_count
        point = cache.source_points[source_index]
        normal = cache.source_normals[source_index]
        basis = cache.basis_values[source_index]
        face = cache.source_faces[source_index]
        source_points[source_index, 1] = point[1]
        source_points[source_index, 2] = point[2]
        source_points[source_index, 3] = point[3]
        source_normals[source_index, 1] = normal[1]
        source_normals[source_index, 2] = normal[2]
        source_normals[source_index, 3] = normal[3]
        basis_values[source_index, 1] = basis[1]
        basis_values[source_index, 2] = basis[2]
        basis_values[source_index, 3] = basis[3]
        source_weights[source_index] = cache.source_weights[source_index]
        source_faces[source_index, 1] = Int32(face[1])
        source_faces[source_index, 2] = Int32(face[2])
        source_faces[source_index, 3] = Int32(face[3])
        source_elements[source_index] = Int32(cache.source_elements[source_index])
    end

    return source_points, source_normals, source_weights, source_faces, source_elements, basis_values
end

function build_cuda_field_evaluation_cache(cache::FieldEvaluationCache{T}) where {T<:AbstractFloat}
    CUDA.functional() || error("CUDA field-evaluation cache requested, but CUDA.functional() is false.")
    source_points, source_normals, source_weights, source_faces, source_elements, basis_values = _cuda_field_arrays(cache)
    return CudaFieldEvaluationCache{T}(
        CuArray(source_points),
        CuArray(source_normals),
        CuArray(source_weights),
        CuArray(source_faces),
        CuArray(source_elements),
        CuArray(basis_values),
        length(source_weights),
    )
end

function build_cuda_field_evaluation_cache(mesh::BoundaryMesh{T}, rule::TriangleRule{T}; symmetry_mode::Symbol=:off) where {T<:AbstractFloat}
    return build_cuda_field_evaluation_cache(build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode))
end

function _cuda_eval_point_arrays(eval_points, ::Type{T}) where {T}
    point_count = length(eval_points)
    points = Matrix{T}(undef, point_count, 3)
    for point_index in 1:point_count
        point = eval_points[point_index]
        points[point_index, 1] = T(point[1])
        points[point_index, 2] = T(point[2])
        points[point_index, 3] = T(point[3])
    end
    return points
end

function _cuda_weighted_field_sources_kernel!(
    pressure_re,
    pressure_im,
    neumann_re,
    neumann_im,
    pressure,
    q_neumann,
    source_weights,
    source_faces,
    source_elements,
    basis_values,
    source_count,
)
    source_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    while source_index <= source_count
        face1 = source_faces[source_index]
        face2 = source_faces[source_index + source_count]
        face3 = source_faces[source_index + 2 * source_count]
        basis1 = basis_values[source_index]
        basis2 = basis_values[source_index + source_count]
        basis3 = basis_values[source_index + 2 * source_count]
        weight = source_weights[source_index]

        p = (basis1 * pressure[face1] + basis2 * pressure[face2] + basis3 * pressure[face3]) * weight
        q = q_neumann[source_elements[source_index]] * weight
        pressure_re[source_index] = real(p)
        pressure_im[source_index] = imag(p)
        neumann_re[source_index] = real(q)
        neumann_im[source_index] = imag(q)

        source_index += stride
    end

    return nothing
end

function _cuda_field_eval_kernel!(
    pot_re,
    pot_im,
    eval_points,
    source_points,
    source_normals,
    pressure_re,
    pressure_im,
    neumann_re,
    neumann_im,
    k,
    source_count,
    point_count,
)
    point_index = blockIdx().x
    point_index > point_count && return nothing

    tid = threadIdx().x
    threads = blockDim().x
    T = typeof(k)
    four_pi = T(12.566370614359172)
    scratch = CUDA.@cuDynamicSharedMem(T, 2 * threads)
    local_re = zero(T)
    local_im = zero(T)

    x1 = eval_points[point_index]
    x2 = eval_points[point_index + point_count]
    x3 = eval_points[point_index + 2 * point_count]

    source_index = tid
    while source_index <= source_count
        y1 = source_points[source_index]
        y2 = source_points[source_index + source_count]
        y3 = source_points[source_index + 2 * source_count]
        r1 = y1 - x1
        r2 = y2 - x2
        r3 = y3 - x3
        radius2 = r1 * r1 + r2 * r2 + r3 * r3

        if radius2 > zero(T)
            radius = sqrt(radius2)
            phase = k * radius
            green_scale = inv(four_pi * radius)
            green_re = cos(phase) * green_scale
            green_im = sin(phase) * green_scale
            grad_scale_re = -inv(radius)
            grad_scale_im = k
            normal = (
                r1 * source_normals[source_index] +
                r2 * source_normals[source_index + source_count] +
                r3 * source_normals[source_index + 2 * source_count]
            ) / radius
            double_re = (green_re * grad_scale_re - green_im * grad_scale_im) * normal
            double_im = (green_re * grad_scale_im + green_im * grad_scale_re) * normal

            p_re = pressure_re[source_index]
            p_im = pressure_im[source_index]
            q_re = neumann_re[source_index]
            q_im = neumann_im[source_index]

            local_re += double_re * p_re - double_im * p_im - (green_re * q_re - green_im * q_im)
            local_im += double_re * p_im + double_im * p_re - (green_re * q_im + green_im * q_re)
        end

        source_index += threads
    end

    scratch[tid] = local_re
    scratch[tid + threads] = local_im
    sync_threads()

    offset = threads >>> 1
    while offset > 0
        if tid <= offset
            scratch[tid] += scratch[tid + offset]
            scratch[tid + threads] += scratch[tid + threads + offset]
        end
        sync_threads()
        offset >>>= 1
    end

    if tid == 1
        pot_re[point_index] = scratch[1]
        pot_im[point_index] = scratch[threads + 1]
    end

    return nothing
end

function _cuda_singular_scatter_kernel!(
    slp_re,
    slp_im,
    adj_re,
    adj_im,
    dlp_re,
    dlp_im,
    hyp_re,
    hyp_im,
    p1_rows,
    p1_cols,
    dp0_cols,
    slp_values,
    adjoint_values,
    dlp_values,
    hypersingular_values,
    p1_dof_count,
    pair_count,
)
    pair_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    while pair_index <= pair_count
        dp0_col = dp0_cols[pair_index]

        for i in 1:3
            row = p1_rows[pair_index + (i - 1) * pair_count]
            slp_value = slp_values[pair_index + (i - 1) * pair_count]
            adj_value = adjoint_values[pair_index + (i - 1) * pair_count]
            slp_index = row + (dp0_col - 1) * p1_dof_count
            _cuda_atomic_add!(slp_re, slp_index, real(slp_value))
            _cuda_atomic_add!(slp_im, slp_index, imag(slp_value))
            _cuda_atomic_add!(adj_re, slp_index, real(adj_value))
            _cuda_atomic_add!(adj_im, slp_index, imag(adj_value))
        end

        value_index = 1
        for j in 1:3
            col = p1_cols[pair_index + (j - 1) * pair_count]
            for i in 1:3
                row = p1_rows[pair_index + (i - 1) * pair_count]
                dense_index = row + (col - 1) * p1_dof_count
                dlp_value = dlp_values[pair_index + (value_index - 1) * pair_count]
                hyp_value = hypersingular_values[pair_index + (value_index - 1) * pair_count]
                _cuda_atomic_add!(dlp_re, dense_index, real(dlp_value))
                _cuda_atomic_add!(dlp_im, dense_index, imag(dlp_value))
                _cuda_atomic_add!(hyp_re, dense_index, real(hyp_value))
                _cuda_atomic_add!(hyp_im, dense_index, imag(hyp_value))
                value_index += 1
            end
        end

        pair_index += stride
    end

    return nothing
end

function _singular_cache_cuda_arrays(cache::SingularCorrectionCache{T}, p1_space::P1Space, dp0_space::DP0Space) where {T}
    pair_count = cache.pair_count
    test_indices = Vector{Int32}(undef, pair_count)
    trial_indices = Vector{Int32}(undef, pair_count)
    rule_indices = Vector{Int32}(undef, pair_count)
    jac_scales = Vector{T}(undef, pair_count)
    normal_products = Vector{T}(undef, pair_count)
    p1_rows = Matrix{Int32}(undef, pair_count, 3)
    p1_cols = Matrix{Int32}(undef, pair_count, 3)
    dp0_cols = Vector{Int32}(undef, pair_count)

    for (pair_index, pair) in enumerate(cache.pairs)
        test_indices[pair_index] = Int32(pair.test_index)
        trial_indices[pair_index] = Int32(pair.trial_index)
        rule_indices[pair_index] = Int32(pair.rule_index)
        jac_scales[pair_index] = pair.jac_scale
        normal_products[pair_index] = pair.normal_product
        test_dofs = p1_space.local_to_global[pair.test_index]
        trial_dofs = p1_space.local_to_global[pair.trial_index]
        dp0_cols[pair_index] = Int32(dp0_space.local_to_global[pair.trial_index])
        for i in 1:3
            p1_rows[pair_index, i] = Int32(test_dofs[i])
            p1_cols[pair_index, i] = Int32(trial_dofs[i])
        end
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

    return (
        test_indices=test_indices,
        trial_indices=trial_indices,
        rule_indices=rule_indices,
        jac_scales=jac_scales,
        normal_products=normal_products,
        p1_rows=p1_rows,
        p1_cols=p1_cols,
        dp0_cols=dp0_cols,
        rule_offsets=rule_offsets,
        rule_test_points=rule_test_points,
        rule_trial_points=rule_trial_points,
        rule_weights=rule_weights,
    )
end

function build_cuda_singular_correction_cache(cache::SingularCorrectionCache{T}, p1_space::P1Space, dp0_space::DP0Space) where {T<:AbstractFloat}
    arrays = _singular_cache_cuda_arrays(cache, p1_space, dp0_space)
    return CudaSingularCorrectionCache{T}(
        CuArray(arrays.test_indices),
        CuArray(arrays.trial_indices),
        CuArray(arrays.rule_indices),
        CuArray(arrays.jac_scales),
        CuArray(arrays.normal_products),
        CuArray(arrays.p1_rows),
        CuArray(arrays.p1_cols),
        CuArray(arrays.dp0_cols),
        CuArray(arrays.rule_offsets),
        CuArray(arrays.rule_test_points),
        CuArray(arrays.rule_trial_points),
        CuArray(arrays.rule_weights),
        cache.pair_count,
    )
end

function image_singular_cache_arrays(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    singular_order::Int,
    element_indices,
    symmetry_mode::Symbol;
    tolerance::T=T(1e-8),
) where {T<:AbstractFloat}
    transforms = symmetry_image_transforms(symmetry_mode)
    base_rules = Dict(
        :coincident => duffy_rule(T, singular_order, :coincident),
        :edge_adjacent => duffy_rule(T, singular_order, :edge_adjacent),
        :vertex_adjacent => duffy_rule(T, singular_order, :vertex_adjacent),
    )
    rules = DuffyRule{T}[]
    rule_indices_by_key = Dict{NTuple{5,Int},Int}()

    pairs = NamedTuple[]
    for transform in transforms
        for (test_index, trial_index) in image_singular_candidates(mesh, element_indices, transform; tolerance=tolerance)
            test_vertices = mesh.face_vertices[test_index]
            trial_vertices = reflect_vertices(transform, mesh.face_vertices[trial_index])
            info = geometric_adjacency_info(test_vertices, trial_vertices; tolerance=tolerance)
            info.kind == :regular && continue
            rule_index = rule_for_singular_orientation!(rules, rule_indices_by_key, base_rules, info)
            test_normal = mesh.normals[test_index]
            trial_normal = reflect_normal(transform, mesh.normals[trial_index])
            push!(pairs, (
                test_index=test_index,
                trial_index=trial_index,
                rule_index=rule_index,
                jac_scale=(T(2.0) * mesh.areas[test_index]) * (T(2.0) * mesh.areas[trial_index]),
                normal_product=dot(test_normal, trial_normal),
                signs=transform.signs,
                curl_signs=SVector{3,Int}(
                    transform.determinant * transform.signs[1],
                    transform.determinant * transform.signs[2],
                    transform.determinant * transform.signs[3],
                ),
            ))
        end
    end

    pair_count = length(pairs)
    test_indices = Vector{Int32}(undef, pair_count)
    trial_indices = Vector{Int32}(undef, pair_count)
    rule_indices = Vector{Int32}(undef, pair_count)
    jac_scales = Vector{T}(undef, pair_count)
    normal_products = Vector{T}(undef, pair_count)
    p1_rows = Matrix{Int32}(undef, pair_count, 3)
    p1_cols = Matrix{Int32}(undef, pair_count, 3)
    dp0_cols = Vector{Int32}(undef, pair_count)
    transform_signs = Matrix{T}(undef, pair_count, 3)
    curl_signs = Matrix{T}(undef, pair_count, 3)

    for (pair_index, pair) in enumerate(pairs)
        test_indices[pair_index] = Int32(pair.test_index)
        trial_indices[pair_index] = Int32(pair.trial_index)
        rule_indices[pair_index] = Int32(pair.rule_index)
        jac_scales[pair_index] = pair.jac_scale
        normal_products[pair_index] = pair.normal_product
        test_dofs = p1_space.local_to_global[pair.test_index]
        trial_dofs = p1_space.local_to_global[pair.trial_index]
        dp0_cols[pair_index] = Int32(dp0_space.local_to_global[pair.trial_index])
        for i in 1:3
            p1_rows[pair_index, i] = Int32(test_dofs[i])
            p1_cols[pair_index, i] = Int32(trial_dofs[i])
            transform_signs[pair_index, i] = T(pair.signs[i])
            curl_signs[pair_index, i] = T(pair.curl_signs[i])
        end
    end

    total_rule_points = sum(length(rule.weights) for rule in rules)
    rule_offsets = Vector{Int32}(undef, length(rules) + 1)
    rule_test_points = Matrix{T}(undef, total_rule_points, 2)
    rule_trial_points = Matrix{T}(undef, total_rule_points, 2)
    rule_weights = Vector{T}(undef, total_rule_points)
    offset = 1
    for (rule_index, rule) in enumerate(rules)
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

    return (
        pair_count=pair_count,
        test_indices=test_indices,
        trial_indices=trial_indices,
        rule_indices=rule_indices,
        jac_scales=jac_scales,
        normal_products=normal_products,
        p1_rows=p1_rows,
        p1_cols=p1_cols,
        dp0_cols=dp0_cols,
        rule_offsets=rule_offsets,
        rule_test_points=rule_test_points,
        rule_trial_points=rule_trial_points,
        rule_weights=rule_weights,
        transform_signs=transform_signs,
        curl_signs=curl_signs,
    )
end

function build_cuda_image_singular_correction_cache(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    singular_order::Int,
    element_indices,
    symmetry_mode::Symbol,
) where {T<:AbstractFloat}
    arrays = image_singular_cache_arrays(mesh, p1_space, dp0_space, singular_order, element_indices, symmetry_mode)
    return CudaImageSingularCorrectionCache{T}(
        CuArray(arrays.test_indices),
        CuArray(arrays.trial_indices),
        CuArray(arrays.rule_indices),
        CuArray(arrays.jac_scales),
        CuArray(arrays.normal_products),
        CuArray(arrays.p1_rows),
        CuArray(arrays.p1_cols),
        CuArray(arrays.dp0_cols),
        CuArray(arrays.rule_offsets),
        CuArray(arrays.rule_test_points),
        CuArray(arrays.rule_trial_points),
        CuArray(arrays.rule_weights),
        CuArray(arrays.transform_signs),
        CuArray(arrays.curl_signs),
        arrays.pair_count,
    )
end

function _free_cuda_image_singular_correction_cache!(cache::CudaImageSingularCorrectionCache)
    CUDA.unsafe_free!(cache.test_indices)
    CUDA.unsafe_free!(cache.trial_indices)
    CUDA.unsafe_free!(cache.rule_indices)
    CUDA.unsafe_free!(cache.jac_scales)
    CUDA.unsafe_free!(cache.normal_products)
    CUDA.unsafe_free!(cache.p1_rows)
    CUDA.unsafe_free!(cache.p1_cols)
    CUDA.unsafe_free!(cache.dp0_cols)
    CUDA.unsafe_free!(cache.rule_offsets)
    CUDA.unsafe_free!(cache.rule_test_points)
    CUDA.unsafe_free!(cache.rule_trial_points)
    CUDA.unsafe_free!(cache.rule_weights)
    CUDA.unsafe_free!(cache.transform_signs)
    CUDA.unsafe_free!(cache.curl_signs)
    return nothing
end

function _cuda_duffy_blocks_kernel!(
    slp_values,
    adjoint_values,
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
)
    pair_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    T = typeof(k)
    four_pi = T(12.566370614359172)

    while pair_index <= pair_count
        test_index = test_indices[pair_index]
        trial_index = trial_indices[pair_index]
        rule_index = rule_indices[pair_index]
        q_start = rule_offsets[rule_index]
        q_stop = rule_offsets[rule_index + 1] - 1
        jac_scale = jac_scales[pair_index]
        normal_product = normal_products[pair_index]

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

        slp1_re = zero(T); slp1_im = zero(T)
        slp2_re = zero(T); slp2_im = zero(T)
        slp3_re = zero(T); slp3_im = zero(T)
        adj1_re = zero(T); adj1_im = zero(T)
        adj2_re = zero(T); adj2_im = zero(T)
        adj3_re = zero(T); adj3_im = zero(T)

        dlp_re_11 = zero(T); dlp_im_11 = zero(T)
        dlp_re_21 = zero(T); dlp_im_21 = zero(T)
        dlp_re_31 = zero(T); dlp_im_31 = zero(T)
        dlp_re_12 = zero(T); dlp_im_12 = zero(T)
        dlp_re_22 = zero(T); dlp_im_22 = zero(T)
        dlp_re_32 = zero(T); dlp_im_32 = zero(T)
        dlp_re_13 = zero(T); dlp_im_13 = zero(T)
        dlp_re_23 = zero(T); dlp_im_23 = zero(T)
        dlp_re_33 = zero(T); dlp_im_33 = zero(T)

        hyp_re_11 = zero(T); hyp_im_11 = zero(T)
        hyp_re_21 = zero(T); hyp_im_21 = zero(T)
        hyp_re_31 = zero(T); hyp_im_31 = zero(T)
        hyp_re_12 = zero(T); hyp_im_12 = zero(T)
        hyp_re_22 = zero(T); hyp_im_22 = zero(T)
        hyp_re_32 = zero(T); hyp_im_32 = zero(T)
        hyp_re_13 = zero(T); hyp_im_13 = zero(T)
        hyp_re_23 = zero(T); hyp_im_23 = zero(T)
        hyp_re_33 = zero(T); hyp_im_33 = zero(T)

        for q in q_start:q_stop
            tx = rule_test_points[q]
            ty = rule_test_points[q + (length(rule_weights))]
            rx = rule_trial_points[q]
            ry = rule_trial_points[q + (length(rule_weights))]
            tb1 = T(1.0) - tx - ty
            tb2 = tx
            tb3 = ty
            rb1 = T(1.0) - rx - ry
            rb2 = rx
            rb3 = ry

            x1 = tb1 * tv1x + tb2 * tv2x + tb3 * tv3x
            x2 = tb1 * tv1y + tb2 * tv2y + tb3 * tv3y
            x3 = tb1 * tv1z + tb2 * tv2z + tb3 * tv3z
            y1 = rb1 * rv1x + rb2 * rv2x + rb3 * rv3x
            y2 = rb1 * rv1y + rb2 * rv2y + rb3 * rv3y
            y3 = rb1 * rv1z + rb2 * rv2z + rb3 * rv3z

            r1 = y1 - x1
            r2 = y2 - x2
            r3 = y3 - x3
            radius2 = r1 * r1 + r2 * r2 + r3 * r3

            if radius2 > zero(T)
                radius = sqrt(radius2)
                phase = k * radius
                green_scale = inv(four_pi * radius)
                single_re = cos(phase) * green_scale
                single_im = sin(phase) * green_scale
                grad_scale_re = -inv(radius)
                grad_scale_im = k
                grad_re = single_re * grad_scale_re - single_im * grad_scale_im
                grad_im = single_re * grad_scale_im + single_im * grad_scale_re
                inv_radius = inv(radius)
                trial_dot = (r1 * rnx + r2 * rny + r3 * rnz) * inv_radius
                test_dot = -(r1 * tnx + r2 * tny + r3 * tnz) * inv_radius
                double_re = grad_re * trial_dot
                double_im = grad_im * trial_dot
                adj_re_value = grad_re * test_dot
                adj_im_value = grad_im * test_dot
                weight = rule_weights[q] * jac_scale
                single_re *= weight
                single_im *= weight
                double_re *= weight
                double_im *= weight
                adj_re_value *= weight
                adj_im_value *= weight

                k2_basis_normal = k * k * normal_product
                c11 = tc11 * rc11 + tc12 * rc12 + tc13 * rc13 - k2_basis_normal * tb1 * rb1
                c21 = tc21 * rc11 + tc22 * rc12 + tc23 * rc13 - k2_basis_normal * tb2 * rb1
                c31 = tc31 * rc11 + tc32 * rc12 + tc33 * rc13 - k2_basis_normal * tb3 * rb1
                c12 = tc11 * rc21 + tc12 * rc22 + tc13 * rc23 - k2_basis_normal * tb1 * rb2
                c22 = tc21 * rc21 + tc22 * rc22 + tc23 * rc23 - k2_basis_normal * tb2 * rb2
                c32 = tc31 * rc21 + tc32 * rc22 + tc33 * rc23 - k2_basis_normal * tb3 * rb2
                c13 = tc11 * rc31 + tc12 * rc32 + tc13 * rc33 - k2_basis_normal * tb1 * rb3
                c23 = tc21 * rc31 + tc22 * rc32 + tc23 * rc33 - k2_basis_normal * tb2 * rb3
                c33 = tc31 * rc31 + tc32 * rc32 + tc33 * rc33 - k2_basis_normal * tb3 * rb3

                slp1_re += tb1 * single_re; slp1_im += tb1 * single_im
                slp2_re += tb2 * single_re; slp2_im += tb2 * single_im
                slp3_re += tb3 * single_re; slp3_im += tb3 * single_im
                adj1_re += tb1 * adj_re_value; adj1_im += tb1 * adj_im_value
                adj2_re += tb2 * adj_re_value; adj2_im += tb2 * adj_im_value
                adj3_re += tb3 * adj_re_value; adj3_im += tb3 * adj_im_value

                dlp_re_11 += tb1 * rb1 * double_re; dlp_im_11 += tb1 * rb1 * double_im
                dlp_re_21 += tb2 * rb1 * double_re; dlp_im_21 += tb2 * rb1 * double_im
                dlp_re_31 += tb3 * rb1 * double_re; dlp_im_31 += tb3 * rb1 * double_im
                dlp_re_12 += tb1 * rb2 * double_re; dlp_im_12 += tb1 * rb2 * double_im
                dlp_re_22 += tb2 * rb2 * double_re; dlp_im_22 += tb2 * rb2 * double_im
                dlp_re_32 += tb3 * rb2 * double_re; dlp_im_32 += tb3 * rb2 * double_im
                dlp_re_13 += tb1 * rb3 * double_re; dlp_im_13 += tb1 * rb3 * double_im
                dlp_re_23 += tb2 * rb3 * double_re; dlp_im_23 += tb2 * rb3 * double_im
                dlp_re_33 += tb3 * rb3 * double_re; dlp_im_33 += tb3 * rb3 * double_im

                hyp_re_11 += c11 * single_re; hyp_im_11 += c11 * single_im
                hyp_re_21 += c21 * single_re; hyp_im_21 += c21 * single_im
                hyp_re_31 += c31 * single_re; hyp_im_31 += c31 * single_im
                hyp_re_12 += c12 * single_re; hyp_im_12 += c12 * single_im
                hyp_re_22 += c22 * single_re; hyp_im_22 += c22 * single_im
                hyp_re_32 += c32 * single_re; hyp_im_32 += c32 * single_im
                hyp_re_13 += c13 * single_re; hyp_im_13 += c13 * single_im
                hyp_re_23 += c23 * single_re; hyp_im_23 += c23 * single_im
                hyp_re_33 += c33 * single_re; hyp_im_33 += c33 * single_im
            end
        end

        slp_values[pair_index] = Complex{T}(slp1_re, slp1_im)
        slp_values[pair_index + pair_count] = Complex{T}(slp2_re, slp2_im)
        slp_values[pair_index + 2 * pair_count] = Complex{T}(slp3_re, slp3_im)
        adjoint_values[pair_index] = Complex{T}(adj1_re, adj1_im)
        adjoint_values[pair_index + pair_count] = Complex{T}(adj2_re, adj2_im)
        adjoint_values[pair_index + 2 * pair_count] = Complex{T}(adj3_re, adj3_im)

        dlp_values[pair_index] = Complex{T}(dlp_re_11, dlp_im_11)
        dlp_values[pair_index + pair_count] = Complex{T}(dlp_re_21, dlp_im_21)
        dlp_values[pair_index + 2 * pair_count] = Complex{T}(dlp_re_31, dlp_im_31)
        dlp_values[pair_index + 3 * pair_count] = Complex{T}(dlp_re_12, dlp_im_12)
        dlp_values[pair_index + 4 * pair_count] = Complex{T}(dlp_re_22, dlp_im_22)
        dlp_values[pair_index + 5 * pair_count] = Complex{T}(dlp_re_32, dlp_im_32)
        dlp_values[pair_index + 6 * pair_count] = Complex{T}(dlp_re_13, dlp_im_13)
        dlp_values[pair_index + 7 * pair_count] = Complex{T}(dlp_re_23, dlp_im_23)
        dlp_values[pair_index + 8 * pair_count] = Complex{T}(dlp_re_33, dlp_im_33)

        hypersingular_values[pair_index] = Complex{T}(hyp_re_11, hyp_im_11)
        hypersingular_values[pair_index + pair_count] = Complex{T}(hyp_re_21, hyp_im_21)
        hypersingular_values[pair_index + 2 * pair_count] = Complex{T}(hyp_re_31, hyp_im_31)
        hypersingular_values[pair_index + 3 * pair_count] = Complex{T}(hyp_re_12, hyp_im_12)
        hypersingular_values[pair_index + 4 * pair_count] = Complex{T}(hyp_re_22, hyp_im_22)
        hypersingular_values[pair_index + 5 * pair_count] = Complex{T}(hyp_re_32, hyp_im_32)
        hypersingular_values[pair_index + 6 * pair_count] = Complex{T}(hyp_re_13, hyp_im_13)
        hypersingular_values[pair_index + 7 * pair_count] = Complex{T}(hyp_re_23, hyp_im_23)
        hypersingular_values[pair_index + 8 * pair_count] = Complex{T}(hyp_re_33, hyp_im_33)

        pair_index += stride
    end

    return nothing
end

function _cuda_image_singular_delta_blocks_kernel!(
    slp_values,
    adjoint_values,
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
    regular_rule_points,
    regular_rule_weights,
    transform_signs,
    curl_signs,
    face_vertices,
    normals,
    curls,
    k,
    face_count,
    regular_rule_count,
    pair_count,
)
    pair_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    T = typeof(k)
    four_pi = T(12.566370614359172)

    while pair_index <= pair_count
        test_index = test_indices[pair_index]
        trial_index = trial_indices[pair_index]
        rule_index = rule_indices[pair_index]
        q_start = rule_offsets[rule_index]
        q_stop = rule_offsets[rule_index + 1] - 1
        jac_scale = jac_scales[pair_index]
        normal_product = normal_products[pair_index]

        sx = transform_signs[pair_index]
        sy = transform_signs[pair_index + pair_count]
        sz = transform_signs[pair_index + 2 * pair_count]
        csx = curl_signs[pair_index]
        csy = curl_signs[pair_index + pair_count]
        csz = curl_signs[pair_index + 2 * pair_count]

        tv1x = face_vertices[test_index]
        tv1y = face_vertices[test_index + face_count]
        tv1z = face_vertices[test_index + 2 * face_count]
        tv2x = face_vertices[test_index + 3 * face_count]
        tv2y = face_vertices[test_index + 4 * face_count]
        tv2z = face_vertices[test_index + 5 * face_count]
        tv3x = face_vertices[test_index + 6 * face_count]
        tv3y = face_vertices[test_index + 7 * face_count]
        tv3z = face_vertices[test_index + 8 * face_count]

        rv1x = sx * face_vertices[trial_index]
        rv1y = sy * face_vertices[trial_index + face_count]
        rv1z = sz * face_vertices[trial_index + 2 * face_count]
        rv2x = sx * face_vertices[trial_index + 3 * face_count]
        rv2y = sy * face_vertices[trial_index + 4 * face_count]
        rv2z = sz * face_vertices[trial_index + 5 * face_count]
        rv3x = sx * face_vertices[trial_index + 6 * face_count]
        rv3y = sy * face_vertices[trial_index + 7 * face_count]
        rv3z = sz * face_vertices[trial_index + 8 * face_count]

        tnx = normals[test_index]
        tny = normals[test_index + face_count]
        tnz = normals[test_index + 2 * face_count]
        rnx = sx * normals[trial_index]
        rny = sy * normals[trial_index + face_count]
        rnz = sz * normals[trial_index + 2 * face_count]

        tc11 = curls[test_index]
        tc12 = curls[test_index + face_count]
        tc13 = curls[test_index + 2 * face_count]
        tc21 = curls[test_index + 3 * face_count]
        tc22 = curls[test_index + 4 * face_count]
        tc23 = curls[test_index + 5 * face_count]
        tc31 = curls[test_index + 6 * face_count]
        tc32 = curls[test_index + 7 * face_count]
        tc33 = curls[test_index + 8 * face_count]

        rc11 = csx * curls[trial_index]
        rc12 = csy * curls[trial_index + face_count]
        rc13 = csz * curls[trial_index + 2 * face_count]
        rc21 = csx * curls[trial_index + 3 * face_count]
        rc22 = csy * curls[trial_index + 4 * face_count]
        rc23 = csz * curls[trial_index + 5 * face_count]
        rc31 = csx * curls[trial_index + 6 * face_count]
        rc32 = csy * curls[trial_index + 7 * face_count]
        rc33 = csz * curls[trial_index + 8 * face_count]

        slp1_re = zero(T); slp1_im = zero(T)
        slp2_re = zero(T); slp2_im = zero(T)
        slp3_re = zero(T); slp3_im = zero(T)
        adj1_re = zero(T); adj1_im = zero(T)
        adj2_re = zero(T); adj2_im = zero(T)
        adj3_re = zero(T); adj3_im = zero(T)
        dlp_re_11 = zero(T); dlp_im_11 = zero(T)
        dlp_re_21 = zero(T); dlp_im_21 = zero(T)
        dlp_re_31 = zero(T); dlp_im_31 = zero(T)
        dlp_re_12 = zero(T); dlp_im_12 = zero(T)
        dlp_re_22 = zero(T); dlp_im_22 = zero(T)
        dlp_re_32 = zero(T); dlp_im_32 = zero(T)
        dlp_re_13 = zero(T); dlp_im_13 = zero(T)
        dlp_re_23 = zero(T); dlp_im_23 = zero(T)
        dlp_re_33 = zero(T); dlp_im_33 = zero(T)
        hyp_re_11 = zero(T); hyp_im_11 = zero(T)
        hyp_re_21 = zero(T); hyp_im_21 = zero(T)
        hyp_re_31 = zero(T); hyp_im_31 = zero(T)
        hyp_re_12 = zero(T); hyp_im_12 = zero(T)
        hyp_re_22 = zero(T); hyp_im_22 = zero(T)
        hyp_re_32 = zero(T); hyp_im_32 = zero(T)
        hyp_re_13 = zero(T); hyp_im_13 = zero(T)
        hyp_re_23 = zero(T); hyp_im_23 = zero(T)
        hyp_re_33 = zero(T); hyp_im_33 = zero(T)

        for q in q_start:q_stop
            tx = rule_test_points[q]
            ty = rule_test_points[q + length(rule_weights)]
            rx = rule_trial_points[q]
            ry = rule_trial_points[q + length(rule_weights)]
            tb1 = T(1.0) - tx - ty
            tb2 = tx
            tb3 = ty
            rb1 = T(1.0) - rx - ry
            rb2 = rx
            rb3 = ry
            x1 = tb1 * tv1x + tb2 * tv2x + tb3 * tv3x
            x2 = tb1 * tv1y + tb2 * tv2y + tb3 * tv3y
            x3 = tb1 * tv1z + tb2 * tv2z + tb3 * tv3z
            y1 = rb1 * rv1x + rb2 * rv2x + rb3 * rv3x
            y2 = rb1 * rv1y + rb2 * rv2y + rb3 * rv3y
            y3 = rb1 * rv1z + rb2 * rv2z + rb3 * rv3z
            r1 = y1 - x1
            r2 = y2 - x2
            r3 = y3 - x3
            radius2 = r1 * r1 + r2 * r2 + r3 * r3
            if radius2 > zero(T)
                radius = sqrt(radius2)
                phase = k * radius
                single_re = cos(phase) / (four_pi * radius)
                single_im = sin(phase) / (four_pi * radius)
                grad_scale_re = -inv(radius)
                grad_scale_im = k
                grad_re = single_re * grad_scale_re - single_im * grad_scale_im
                grad_im = single_re * grad_scale_im + single_im * grad_scale_re
                inv_radius = inv(radius)
                trial_dot = (r1 * rnx + r2 * rny + r3 * rnz) * inv_radius
                test_dot = -(r1 * tnx + r2 * tny + r3 * tnz) * inv_radius
                double_re = grad_re * trial_dot
                double_im = grad_im * trial_dot
                adj_re_value = grad_re * test_dot
                adj_im_value = grad_im * test_dot
                weight = rule_weights[q] * jac_scale
                single_re *= weight; single_im *= weight
                double_re *= weight; double_im *= weight
                adj_re_value *= weight; adj_im_value *= weight
                k2_basis_normal = k * k * normal_product
                c11 = tc11 * rc11 + tc12 * rc12 + tc13 * rc13 - k2_basis_normal * tb1 * rb1
                c21 = tc21 * rc11 + tc22 * rc12 + tc23 * rc13 - k2_basis_normal * tb2 * rb1
                c31 = tc31 * rc11 + tc32 * rc12 + tc33 * rc13 - k2_basis_normal * tb3 * rb1
                c12 = tc11 * rc21 + tc12 * rc22 + tc13 * rc23 - k2_basis_normal * tb1 * rb2
                c22 = tc21 * rc21 + tc22 * rc22 + tc23 * rc23 - k2_basis_normal * tb2 * rb2
                c32 = tc31 * rc21 + tc32 * rc22 + tc33 * rc23 - k2_basis_normal * tb3 * rb2
                c13 = tc11 * rc31 + tc12 * rc32 + tc13 * rc33 - k2_basis_normal * tb1 * rb3
                c23 = tc21 * rc31 + tc22 * rc32 + tc23 * rc33 - k2_basis_normal * tb2 * rb3
                c33 = tc31 * rc31 + tc32 * rc32 + tc33 * rc33 - k2_basis_normal * tb3 * rb3
                slp1_re += tb1 * single_re; slp1_im += tb1 * single_im
                slp2_re += tb2 * single_re; slp2_im += tb2 * single_im
                slp3_re += tb3 * single_re; slp3_im += tb3 * single_im
                adj1_re += tb1 * adj_re_value; adj1_im += tb1 * adj_im_value
                adj2_re += tb2 * adj_re_value; adj2_im += tb2 * adj_im_value
                adj3_re += tb3 * adj_re_value; adj3_im += tb3 * adj_im_value
                dlp_re_11 += tb1 * rb1 * double_re; dlp_im_11 += tb1 * rb1 * double_im
                dlp_re_21 += tb2 * rb1 * double_re; dlp_im_21 += tb2 * rb1 * double_im
                dlp_re_31 += tb3 * rb1 * double_re; dlp_im_31 += tb3 * rb1 * double_im
                dlp_re_12 += tb1 * rb2 * double_re; dlp_im_12 += tb1 * rb2 * double_im
                dlp_re_22 += tb2 * rb2 * double_re; dlp_im_22 += tb2 * rb2 * double_im
                dlp_re_32 += tb3 * rb2 * double_re; dlp_im_32 += tb3 * rb2 * double_im
                dlp_re_13 += tb1 * rb3 * double_re; dlp_im_13 += tb1 * rb3 * double_im
                dlp_re_23 += tb2 * rb3 * double_re; dlp_im_23 += tb2 * rb3 * double_im
                dlp_re_33 += tb3 * rb3 * double_re; dlp_im_33 += tb3 * rb3 * double_im
                hyp_re_11 += c11 * single_re; hyp_im_11 += c11 * single_im
                hyp_re_21 += c21 * single_re; hyp_im_21 += c21 * single_im
                hyp_re_31 += c31 * single_re; hyp_im_31 += c31 * single_im
                hyp_re_12 += c12 * single_re; hyp_im_12 += c12 * single_im
                hyp_re_22 += c22 * single_re; hyp_im_22 += c22 * single_im
                hyp_re_32 += c32 * single_re; hyp_im_32 += c32 * single_im
                hyp_re_13 += c13 * single_re; hyp_im_13 += c13 * single_im
                hyp_re_23 += c23 * single_re; hyp_im_23 += c23 * single_im
                hyp_re_33 += c33 * single_re; hyp_im_33 += c33 * single_im
            end
        end

        for tq in 1:regular_rule_count
            tx = regular_rule_points[tq]
            ty = regular_rule_points[tq + regular_rule_count]
            tw = regular_rule_weights[tq]
            tb1 = T(1.0) - tx - ty
            tb2 = tx
            tb3 = ty
            x1 = tb1 * tv1x + tb2 * tv2x + tb3 * tv3x
            x2 = tb1 * tv1y + tb2 * tv2y + tb3 * tv3y
            x3 = tb1 * tv1z + tb2 * tv2z + tb3 * tv3z
            for rq in 1:regular_rule_count
                rx = regular_rule_points[rq]
                ry = regular_rule_points[rq + regular_rule_count]
                rw = regular_rule_weights[rq]
                rb1 = T(1.0) - rx - ry
                rb2 = rx
                rb3 = ry
                y1 = rb1 * rv1x + rb2 * rv2x + rb3 * rv3x
                y2 = rb1 * rv1y + rb2 * rv2y + rb3 * rv3y
                y3 = rb1 * rv1z + rb2 * rv2z + rb3 * rv3z
                r1 = y1 - x1
                r2 = y2 - x2
                r3 = y3 - x3
                radius2 = r1 * r1 + r2 * r2 + r3 * r3
                if radius2 > zero(T)
                    radius = sqrt(radius2)
                    phase = k * radius
                    single_re = cos(phase) / (four_pi * radius)
                    single_im = sin(phase) / (four_pi * radius)
                    grad_scale_re = -inv(radius)
                    grad_scale_im = k
                    grad_re = single_re * grad_scale_re - single_im * grad_scale_im
                    grad_im = single_re * grad_scale_im + single_im * grad_scale_re
                    inv_radius = inv(radius)
                    trial_dot = (r1 * rnx + r2 * rny + r3 * rnz) * inv_radius
                    test_dot = -(r1 * tnx + r2 * tny + r3 * tnz) * inv_radius
                    double_re = grad_re * trial_dot
                    double_im = grad_im * trial_dot
                    adj_re_value = grad_re * test_dot
                    adj_im_value = grad_im * test_dot
                    weight = tw * rw * jac_scale
                    single_re *= weight; single_im *= weight
                    double_re *= weight; double_im *= weight
                    adj_re_value *= weight; adj_im_value *= weight
                    k2_basis_normal = k * k * normal_product
                    c11 = tc11 * rc11 + tc12 * rc12 + tc13 * rc13 - k2_basis_normal * tb1 * rb1
                    c21 = tc21 * rc11 + tc22 * rc12 + tc23 * rc13 - k2_basis_normal * tb2 * rb1
                    c31 = tc31 * rc11 + tc32 * rc12 + tc33 * rc13 - k2_basis_normal * tb3 * rb1
                    c12 = tc11 * rc21 + tc12 * rc22 + tc13 * rc23 - k2_basis_normal * tb1 * rb2
                    c22 = tc21 * rc21 + tc22 * rc22 + tc23 * rc23 - k2_basis_normal * tb2 * rb2
                    c32 = tc31 * rc21 + tc32 * rc22 + tc33 * rc23 - k2_basis_normal * tb3 * rb2
                    c13 = tc11 * rc31 + tc12 * rc32 + tc13 * rc33 - k2_basis_normal * tb1 * rb3
                    c23 = tc21 * rc31 + tc22 * rc32 + tc23 * rc33 - k2_basis_normal * tb2 * rb3
                    c33 = tc31 * rc31 + tc32 * rc32 + tc33 * rc33 - k2_basis_normal * tb3 * rb3
                    slp1_re -= tb1 * single_re; slp1_im -= tb1 * single_im
                    slp2_re -= tb2 * single_re; slp2_im -= tb2 * single_im
                    slp3_re -= tb3 * single_re; slp3_im -= tb3 * single_im
                    adj1_re -= tb1 * adj_re_value; adj1_im -= tb1 * adj_im_value
                    adj2_re -= tb2 * adj_re_value; adj2_im -= tb2 * adj_im_value
                    adj3_re -= tb3 * adj_re_value; adj3_im -= tb3 * adj_im_value
                    dlp_re_11 -= tb1 * rb1 * double_re; dlp_im_11 -= tb1 * rb1 * double_im
                    dlp_re_21 -= tb2 * rb1 * double_re; dlp_im_21 -= tb2 * rb1 * double_im
                    dlp_re_31 -= tb3 * rb1 * double_re; dlp_im_31 -= tb3 * rb1 * double_im
                    dlp_re_12 -= tb1 * rb2 * double_re; dlp_im_12 -= tb1 * rb2 * double_im
                    dlp_re_22 -= tb2 * rb2 * double_re; dlp_im_22 -= tb2 * rb2 * double_im
                    dlp_re_32 -= tb3 * rb2 * double_re; dlp_im_32 -= tb3 * rb2 * double_im
                    dlp_re_13 -= tb1 * rb3 * double_re; dlp_im_13 -= tb1 * rb3 * double_im
                    dlp_re_23 -= tb2 * rb3 * double_re; dlp_im_23 -= tb2 * rb3 * double_im
                    dlp_re_33 -= tb3 * rb3 * double_re; dlp_im_33 -= tb3 * rb3 * double_im
                    hyp_re_11 -= c11 * single_re; hyp_im_11 -= c11 * single_im
                    hyp_re_21 -= c21 * single_re; hyp_im_21 -= c21 * single_im
                    hyp_re_31 -= c31 * single_re; hyp_im_31 -= c31 * single_im
                    hyp_re_12 -= c12 * single_re; hyp_im_12 -= c12 * single_im
                    hyp_re_22 -= c22 * single_re; hyp_im_22 -= c22 * single_im
                    hyp_re_32 -= c32 * single_re; hyp_im_32 -= c32 * single_im
                    hyp_re_13 -= c13 * single_re; hyp_im_13 -= c13 * single_im
                    hyp_re_23 -= c23 * single_re; hyp_im_23 -= c23 * single_im
                    hyp_re_33 -= c33 * single_re; hyp_im_33 -= c33 * single_im
                end
            end
        end

        slp_values[pair_index] = Complex{T}(slp1_re, slp1_im)
        slp_values[pair_index + pair_count] = Complex{T}(slp2_re, slp2_im)
        slp_values[pair_index + 2 * pair_count] = Complex{T}(slp3_re, slp3_im)
        adjoint_values[pair_index] = Complex{T}(adj1_re, adj1_im)
        adjoint_values[pair_index + pair_count] = Complex{T}(adj2_re, adj2_im)
        adjoint_values[pair_index + 2 * pair_count] = Complex{T}(adj3_re, adj3_im)
        dlp_values[pair_index] = Complex{T}(dlp_re_11, dlp_im_11)
        dlp_values[pair_index + pair_count] = Complex{T}(dlp_re_21, dlp_im_21)
        dlp_values[pair_index + 2 * pair_count] = Complex{T}(dlp_re_31, dlp_im_31)
        dlp_values[pair_index + 3 * pair_count] = Complex{T}(dlp_re_12, dlp_im_12)
        dlp_values[pair_index + 4 * pair_count] = Complex{T}(dlp_re_22, dlp_im_22)
        dlp_values[pair_index + 5 * pair_count] = Complex{T}(dlp_re_32, dlp_im_32)
        dlp_values[pair_index + 6 * pair_count] = Complex{T}(dlp_re_13, dlp_im_13)
        dlp_values[pair_index + 7 * pair_count] = Complex{T}(dlp_re_23, dlp_im_23)
        dlp_values[pair_index + 8 * pair_count] = Complex{T}(dlp_re_33, dlp_im_33)
        hypersingular_values[pair_index] = Complex{T}(hyp_re_11, hyp_im_11)
        hypersingular_values[pair_index + pair_count] = Complex{T}(hyp_re_21, hyp_im_21)
        hypersingular_values[pair_index + 2 * pair_count] = Complex{T}(hyp_re_31, hyp_im_31)
        hypersingular_values[pair_index + 3 * pair_count] = Complex{T}(hyp_re_12, hyp_im_12)
        hypersingular_values[pair_index + 4 * pair_count] = Complex{T}(hyp_re_22, hyp_im_22)
        hypersingular_values[pair_index + 5 * pair_count] = Complex{T}(hyp_re_32, hyp_im_32)
        hypersingular_values[pair_index + 6 * pair_count] = Complex{T}(hyp_re_13, hyp_im_13)
        hypersingular_values[pair_index + 7 * pair_count] = Complex{T}(hyp_re_23, hyp_im_23)
        hypersingular_values[pair_index + 8 * pair_count] = Complex{T}(hyp_re_33, hyp_im_33)

        pair_index += stride
    end
    return nothing
end

function add_singular_corrections_cuda_compact!(
    operators,
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    singular_order::Int,
    element_indices,
    singular_cache=nothing;
    cuda_singular_cache=nothing,
    cuda_regular_cache=nothing,
    timing=nothing,
) where {T<:AbstractFloat}
    CUDA.functional() || error("CUDA singular correction scatter requested, but CUDA.functional() is false.")
    cache = singular_cache === nothing ? build_singular_correction_cache(mesh, singular_order, element_indices) : singular_cache

    cuda_cache = cuda_singular_cache
    if cuda_cache === nothing
        cuda_cache = _cuda_timed_stage!(timing, "singular_correction_cuda_cache_build") do
            build_cuda_singular_correction_cache(cache, p1_space, dp0_space)
        end
    else
        timing !== nothing && (timing["singular_correction_cuda_cache_build"] = 0.0)
    end

    face_vertices = normals = curls = nothing
    owns_geometry = cuda_regular_cache === nothing
    if owns_geometry
        _cuda_timed_stage!(timing, "singular_correction_geometry_transfer") do
            host_face_vertices, host_normals, _, _, host_curls = _cuda_geometry_arrays(mesh)
            face_vertices = CuArray(host_face_vertices)
            normals = CuArray(host_normals)
            curls = CuArray(host_curls)
            CUDA.synchronize()
            nothing
        end
    else
        timing !== nothing && (timing["singular_correction_geometry_transfer"] = 0.0)
        face_vertices = cuda_regular_cache.face_vertices
        normals = cuda_regular_cache.normals
        curls = cuda_regular_cache.curls
    end

    p1_dof_count = p1_space.global_dof_count
    dp0_dof_count = dp0_space.global_dof_count
    d_slp_values = d_adjoint_values = d_dlp_values = d_hyp_values = nothing
    _cuda_timed_stage!(timing, "singular_correction_block_alloc") do
        d_slp_values = CUDA.zeros(Complex{T}, cache.pair_count, 3)
        d_adjoint_values = CUDA.zeros(Complex{T}, cache.pair_count, 3)
        d_dlp_values = CUDA.zeros(Complex{T}, cache.pair_count, 9)
        d_hyp_values = CUDA.zeros(Complex{T}, cache.pair_count, 9)
        CUDA.synchronize()
        nothing
    end

    _cuda_timed_stage!(timing, "singular_correction_block_compute") do
        threads = 128
        blocks_per_grid = min(cld(cache.pair_count, threads), 65_535)
        CUDA.@cuda threads=threads blocks=blocks_per_grid _cuda_duffy_blocks_kernel!(
            d_slp_values,
            d_adjoint_values,
            d_dlp_values,
            d_hyp_values,
            cuda_cache.test_indices,
            cuda_cache.trial_indices,
            cuda_cache.rule_indices,
            cuda_cache.jac_scales,
            cuda_cache.normal_products,
            cuda_cache.rule_offsets,
            cuda_cache.rule_test_points,
            cuda_cache.rule_trial_points,
            cuda_cache.rule_weights,
            face_vertices,
            normals,
            curls,
            k,
            length(mesh.faces),
            cache.pair_count,
        )
        CUDA.synchronize()
        nothing
    end

    timing !== nothing && (timing["singular_correction_compact_transfer"] = 0.0)

    slp_re = slp_im = adj_re = adj_im = dlp_re = dlp_im = hyp_re = hyp_im = nothing
    _cuda_timed_stage!(timing, "singular_correction_gpu_alloc") do
        slp_re = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        slp_im = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        adj_re = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        adj_im = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        dlp_re = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        dlp_im = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        hyp_re = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        hyp_im = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        CUDA.synchronize()
        nothing
    end

    _cuda_timed_stage!(timing, "singular_correction_gpu_scatter") do
        threads = 128
        blocks_per_grid = min(cld(cache.pair_count, threads), 65_535)
        CUDA.@cuda threads=threads blocks=blocks_per_grid _cuda_singular_scatter_kernel!(
            slp_re,
            slp_im,
            adj_re,
            adj_im,
            dlp_re,
            dlp_im,
            hyp_re,
            hyp_im,
            cuda_cache.p1_rows,
            cuda_cache.p1_cols,
            cuda_cache.dp0_cols,
            d_slp_values,
            d_adjoint_values,
            d_dlp_values,
            d_hyp_values,
            p1_dof_count,
            cache.pair_count,
        )
        CUDA.synchronize()
        nothing
    end

    d_single = d_double = d_adjoint = d_hypersingular = nothing
    _cuda_timed_stage!(timing, "singular_correction_gpu_add") do
        d_single = _complex_gpu_matrix(slp_re, slp_im)
        d_adjoint = _complex_gpu_matrix(adj_re, adj_im)
        d_double = _complex_gpu_matrix(dlp_re, dlp_im)
        d_hypersingular = _complex_gpu_matrix(hyp_re, hyp_im)
        operators.single_layer .+= d_single
        operators.adjoint_double_layer .+= d_adjoint
        operators.double_layer .+= d_double
        operators.hypersingular .+= d_hypersingular
        CUDA.synchronize()
        nothing
    end

    if owns_geometry
        CUDA.unsafe_free!(face_vertices)
        CUDA.unsafe_free!(normals)
        CUDA.unsafe_free!(curls)
    end
    CUDA.unsafe_free!(d_slp_values)
    CUDA.unsafe_free!(d_adjoint_values)
    CUDA.unsafe_free!(d_dlp_values)
    CUDA.unsafe_free!(d_hyp_values)
    CUDA.unsafe_free!(slp_re)
    CUDA.unsafe_free!(slp_im)
    CUDA.unsafe_free!(adj_re)
    CUDA.unsafe_free!(adj_im)
    CUDA.unsafe_free!(dlp_re)
    CUDA.unsafe_free!(dlp_im)
    CUDA.unsafe_free!(hyp_re)
    CUDA.unsafe_free!(hyp_im)
    CUDA.unsafe_free!(d_single)
    CUDA.unsafe_free!(d_adjoint)
    CUDA.unsafe_free!(d_double)
    CUDA.unsafe_free!(d_hypersingular)
    return cache.pair_count
end

function add_image_singular_corrections_cuda_compact!(
    operators,
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    regular_rule::TriangleRule{T},
    singular_order::Int,
    element_indices,
    symmetry_mode::Symbol;
    cuda_regular_cache=nothing,
    timing=nothing,
) where {T<:AbstractFloat}
    CUDA.functional() || error("CUDA image singular correction scatter requested, but CUDA.functional() is false.")
    normalized_symmetry_mode(symmetry_mode) == :off && return 0

    cuda_cache = _cuda_timed_stage!(timing, "image_singular_correction_cuda_cache_build") do
        build_cuda_image_singular_correction_cache(mesh, p1_space, dp0_space, singular_order, element_indices, symmetry_mode)
    end
    pair_count = cuda_cache.pair_count
    if pair_count == 0
        _free_cuda_image_singular_correction_cache!(cuda_cache)
        return 0
    end

    face_vertices = normals = curls = regular_rule_points = regular_rule_weights = nothing
    owns_geometry = cuda_regular_cache === nothing
    owns_rule = cuda_regular_cache === nothing
    if owns_geometry
        _cuda_timed_stage!(timing, "image_singular_correction_geometry_transfer") do
            host_face_vertices, host_normals, _, _, host_curls = _cuda_geometry_arrays(mesh)
            face_vertices = CuArray(host_face_vertices)
            normals = CuArray(host_normals)
            curls = CuArray(host_curls)
            CUDA.synchronize()
            nothing
        end
        _cuda_timed_stage!(timing, "image_singular_correction_rule_transfer") do
            host_rule_points, host_rule_weights = _cuda_rule_arrays(regular_rule)
            regular_rule_points = CuArray(host_rule_points)
            regular_rule_weights = CuArray(host_rule_weights)
            CUDA.synchronize()
            nothing
        end
    else
        timing !== nothing && (timing["image_singular_correction_geometry_transfer"] = 0.0)
        timing !== nothing && (timing["image_singular_correction_rule_transfer"] = 0.0)
        face_vertices = cuda_regular_cache.face_vertices
        normals = cuda_regular_cache.normals
        curls = cuda_regular_cache.curls
        regular_rule_points = cuda_regular_cache.rule_points
        regular_rule_weights = cuda_regular_cache.rule_weights
    end

    p1_dof_count = p1_space.global_dof_count
    dp0_dof_count = dp0_space.global_dof_count
    d_slp_values = d_adjoint_values = d_dlp_values = d_hyp_values = nothing
    _cuda_timed_stage!(timing, "image_singular_correction_block_alloc") do
        d_slp_values = CUDA.zeros(Complex{T}, pair_count, 3)
        d_adjoint_values = CUDA.zeros(Complex{T}, pair_count, 3)
        d_dlp_values = CUDA.zeros(Complex{T}, pair_count, 9)
        d_hyp_values = CUDA.zeros(Complex{T}, pair_count, 9)
        CUDA.synchronize()
        nothing
    end

    _cuda_timed_stage!(timing, "image_singular_correction_block_compute") do
        threads = 128
        blocks_per_grid = min(cld(pair_count, threads), 65_535)
        CUDA.@cuda threads=threads blocks=blocks_per_grid _cuda_image_singular_delta_blocks_kernel!(
            d_slp_values,
            d_adjoint_values,
            d_dlp_values,
            d_hyp_values,
            cuda_cache.test_indices,
            cuda_cache.trial_indices,
            cuda_cache.rule_indices,
            cuda_cache.jac_scales,
            cuda_cache.normal_products,
            cuda_cache.rule_offsets,
            cuda_cache.rule_test_points,
            cuda_cache.rule_trial_points,
            cuda_cache.rule_weights,
            regular_rule_points,
            regular_rule_weights,
            cuda_cache.transform_signs,
            cuda_cache.curl_signs,
            face_vertices,
            normals,
            curls,
            k,
            length(mesh.faces),
            length(regular_rule.weights),
            pair_count,
        )
        CUDA.synchronize()
        nothing
    end

    slp_re = slp_im = adj_re = adj_im = dlp_re = dlp_im = hyp_re = hyp_im = nothing
    _cuda_timed_stage!(timing, "image_singular_correction_gpu_alloc") do
        slp_re = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        slp_im = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        adj_re = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        adj_im = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        dlp_re = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        dlp_im = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        hyp_re = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        hyp_im = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        CUDA.synchronize()
        nothing
    end

    _cuda_timed_stage!(timing, "image_singular_correction_gpu_scatter") do
        threads = 128
        blocks_per_grid = min(cld(pair_count, threads), 65_535)
        CUDA.@cuda threads=threads blocks=blocks_per_grid _cuda_singular_scatter_kernel!(
            slp_re,
            slp_im,
            adj_re,
            adj_im,
            dlp_re,
            dlp_im,
            hyp_re,
            hyp_im,
            cuda_cache.p1_rows,
            cuda_cache.p1_cols,
            cuda_cache.dp0_cols,
            d_slp_values,
            d_adjoint_values,
            d_dlp_values,
            d_hyp_values,
            p1_dof_count,
            pair_count,
        )
        CUDA.synchronize()
        nothing
    end

    if get(operators, :on_gpu, false)
        d_single = d_double = d_adjoint = d_hypersingular = nothing
        _cuda_timed_stage!(timing, "image_singular_correction_gpu_add") do
            d_single = _complex_gpu_matrix(slp_re, slp_im)
            d_adjoint = _complex_gpu_matrix(adj_re, adj_im)
            d_double = _complex_gpu_matrix(dlp_re, dlp_im)
            d_hypersingular = _complex_gpu_matrix(hyp_re, hyp_im)
            operators.single_layer .+= d_single
            operators.adjoint_double_layer .+= d_adjoint
            operators.double_layer .+= d_double
            operators.hypersingular .+= d_hypersingular
            CUDA.synchronize()
            nothing
        end
        CUDA.unsafe_free!(d_single)
        CUDA.unsafe_free!(d_adjoint)
        CUDA.unsafe_free!(d_double)
        CUDA.unsafe_free!(d_hypersingular)
    else
        _cuda_timed_stage!(timing, "image_singular_correction_cpu_add") do
            operators.single_layer .+= _complex_cpu_matrix(slp_re, slp_im, T)
            operators.adjoint_double_layer .+= _complex_cpu_matrix(adj_re, adj_im, T)
            operators.double_layer .+= _complex_cpu_matrix(dlp_re, dlp_im, T)
            operators.hypersingular .+= _complex_cpu_matrix(hyp_re, hyp_im, T)
            nothing
        end
    end

    if owns_geometry
        CUDA.unsafe_free!(face_vertices)
        CUDA.unsafe_free!(normals)
        CUDA.unsafe_free!(curls)
    end
    if owns_rule
        CUDA.unsafe_free!(regular_rule_points)
        CUDA.unsafe_free!(regular_rule_weights)
    end
    CUDA.unsafe_free!(d_slp_values)
    CUDA.unsafe_free!(d_adjoint_values)
    CUDA.unsafe_free!(d_dlp_values)
    CUDA.unsafe_free!(d_hyp_values)
    CUDA.unsafe_free!(slp_re)
    CUDA.unsafe_free!(slp_im)
    CUDA.unsafe_free!(adj_re)
    CUDA.unsafe_free!(adj_im)
    CUDA.unsafe_free!(dlp_re)
    CUDA.unsafe_free!(dlp_im)
    CUDA.unsafe_free!(hyp_re)
    CUDA.unsafe_free!(hyp_im)
    _free_cuda_image_singular_correction_cache!(cuda_cache)
    return pair_count
end

function evaluate_galerkin_field_cuda(
    eval_points,
    mesh::BoundaryMesh{T},
    pressure,
    q_neumann,
    k::T,
    cache::CudaFieldEvaluationCache{T};
    return_gpu::Bool=false,
) where {T<:AbstractFloat}
    point_count = length(eval_points)
    point_count == 0 && return return_gpu ? CuArray(Complex{T}[]) : Complex{T}[]
    CUDA.functional() || error("CUDA field evaluation requested, but CUDA.functional() is false.")

    d_eval_points = CuArray(_cuda_eval_point_arrays(eval_points, T))
    pressure_on_gpu = pressure isa CuArray
    neumann_on_gpu = q_neumann isa CuArray
    d_pressure = pressure_on_gpu ? pressure : CuArray(pressure)
    d_q_neumann = neumann_on_gpu ? q_neumann : CuArray(q_neumann)
    d_pressure_re = CUDA.zeros(T, cache.source_count)
    d_pressure_im = CUDA.zeros(T, cache.source_count)
    d_neumann_re = CUDA.zeros(T, cache.source_count)
    d_neumann_im = CUDA.zeros(T, cache.source_count)
    d_pot_re = CUDA.zeros(T, point_count)
    d_pot_im = CUDA.zeros(T, point_count)

    threads = 256
    source_blocks = min(cld(cache.source_count, threads), 65_535)
    CUDA.@cuda threads=threads blocks=source_blocks _cuda_weighted_field_sources_kernel!(
        d_pressure_re,
        d_pressure_im,
        d_neumann_re,
        d_neumann_im,
        d_pressure,
        d_q_neumann,
        cache.source_weights,
        cache.source_faces,
        cache.source_elements,
        cache.basis_values,
        cache.source_count,
    )

    eval_threads = 256
    CUDA.@cuda threads=eval_threads blocks=point_count shmem=2 * eval_threads * sizeof(T) _cuda_field_eval_kernel!(
        d_pot_re,
        d_pot_im,
        d_eval_points,
        cache.source_points,
        cache.source_normals,
        d_pressure_re,
        d_pressure_im,
        d_neumann_re,
        d_neumann_im,
        k,
        cache.source_count,
        point_count,
    )
    CUDA.synchronize()

    d_pot = complex.(d_pot_re, d_pot_im)
    result = return_gpu ? d_pot : Complex{T}.(Array(d_pot))

    CUDA.unsafe_free!(d_eval_points)
    pressure_on_gpu || CUDA.unsafe_free!(d_pressure)
    neumann_on_gpu || CUDA.unsafe_free!(d_q_neumann)
    CUDA.unsafe_free!(d_pressure_re)
    CUDA.unsafe_free!(d_pressure_im)
    CUDA.unsafe_free!(d_neumann_re)
    CUDA.unsafe_free!(d_neumann_im)
    CUDA.unsafe_free!(d_pot_re)
    CUDA.unsafe_free!(d_pot_im)
    return_gpu || CUDA.unsafe_free!(d_pot)

    return result
end

function release_operator_storage!(operators::NamedTuple)
    get(operators, :on_gpu, false) || return nothing
    CUDA.unsafe_free!(operators.single_layer)
    CUDA.unsafe_free!(operators.double_layer)
    CUDA.unsafe_free!(operators.adjoint_double_layer)
    CUDA.unsafe_free!(operators.hypersingular)
    return nothing
end

function _complex_gpu_matrix(real_part, imag_part)
    return complex.(real_part, imag_part)
end

function _complex_cpu_matrix(real_part, imag_part, ::Type{T}) where {T}
    return Complex{T}.(Array(real_part), Array(imag_part))
end

function _apply_p1_row_weights!(matrix, weights)
    matrix .*= reshape(weights, :, 1)
    return nothing
end

function _apply_operator_p1_row_weights!(operators, mesh::BoundaryMesh{T}, symmetry_mode) where {T<:AbstractFloat}
    weights = p1_symmetry_orbit_weights(mesh, symmetry_mode)
    if get(operators, :on_gpu, false)
        d_weights = CuArray(Complex{T}.(weights))
        _apply_p1_row_weights!(operators.single_layer, d_weights)
        _apply_p1_row_weights!(operators.double_layer, d_weights)
        _apply_p1_row_weights!(operators.adjoint_double_layer, d_weights)
        _apply_p1_row_weights!(operators.hypersingular, d_weights)
        CUDA.synchronize()
        CUDA.unsafe_free!(d_weights)
    else
        complex_weights = Complex{T}.(weights)
        _apply_p1_row_weights!(operators.single_layer, complex_weights)
        _apply_p1_row_weights!(operators.double_layer, complex_weights)
        _apply_p1_row_weights!(operators.adjoint_double_layer, complex_weights)
        _apply_p1_row_weights!(operators.hypersingular, complex_weights)
    end
    return nothing
end

function _cuda_timed_stage!(timing, name::String, thunk)
    value = nothing
    elapsed = @elapsed value = thunk()
    timing !== nothing && (timing[name] = elapsed)
    return value
end

_cuda_timed_stage!(thunk, timing, name::String) = _cuda_timed_stage!(timing, name, thunk)

_regular_quadrature_threads(rule_count::Int) = 32

function _cuda_regular_real_buffers(::Type{T}, p1_dof_count::Int, dp0_dof_count::Int) where {T<:AbstractFloat}
    return (
        slp_re=CUDA.zeros(T, p1_dof_count, dp0_dof_count),
        slp_im=CUDA.zeros(T, p1_dof_count, dp0_dof_count),
        adj_re=CUDA.zeros(T, p1_dof_count, dp0_dof_count),
        adj_im=CUDA.zeros(T, p1_dof_count, dp0_dof_count),
        dlp_re=CUDA.zeros(T, p1_dof_count, p1_dof_count),
        dlp_im=CUDA.zeros(T, p1_dof_count, p1_dof_count),
        hyp_re=CUDA.zeros(T, p1_dof_count, p1_dof_count),
        hyp_im=CUDA.zeros(T, p1_dof_count, p1_dof_count),
    )
end

function _cuda_fill_regular_real_buffers!(buffers, value)
    fill!(buffers.slp_re, value)
    fill!(buffers.slp_im, value)
    fill!(buffers.adj_re, value)
    fill!(buffers.adj_im, value)
    fill!(buffers.dlp_re, value)
    fill!(buffers.dlp_im, value)
    fill!(buffers.hyp_re, value)
    fill!(buffers.hyp_im, value)
    CUDA.synchronize()
    return nothing
end

function _cuda_free_regular_real_buffers!(buffers)
    CUDA.unsafe_free!(buffers.slp_re)
    CUDA.unsafe_free!(buffers.slp_im)
    CUDA.unsafe_free!(buffers.adj_re)
    CUDA.unsafe_free!(buffers.adj_im)
    CUDA.unsafe_free!(buffers.dlp_re)
    CUDA.unsafe_free!(buffers.dlp_im)
    CUDA.unsafe_free!(buffers.hyp_re)
    CUDA.unsafe_free!(buffers.hyp_im)
    return nothing
end

function _launch_regular_split_atomic_kernel!(
    slp_re,
    slp_im,
    dlp_re,
    dlp_im,
    adj_re,
    adj_im,
    hyp_re,
    hyp_im,
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
    p1_dof_count::Int,
    face_count::Int,
    rule_count::Int,
    total_pairs::Int,
    threads::Int,
) where {T<:AbstractFloat}
    slp_shmem = threads * 12 * sizeof(T)
    CUDA.@cuda threads=threads blocks=total_pairs shmem=slp_shmem _cuda_regular_quadrature_slp_adjoint_kernel!(
        slp_re,
        slp_im,
        adj_re,
        adj_im,
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

    dlp_shmem = threads * 36 * sizeof(T)
    CUDA.@cuda threads=threads blocks=total_pairs shmem=dlp_shmem _cuda_regular_quadrature_dlp_hyp_kernel!(
        dlp_re,
        dlp_im,
        hyp_re,
        hyp_im,
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
    return nothing
end

function _launch_regular_symmetry_image_kernel!(
    slp_re,
    slp_im,
    dlp_re,
    dlp_im,
    adj_re,
    adj_im,
    hyp_re,
    hyp_im,
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
    p1_dof_count::Int,
    dp0_dof_count::Int,
    face_count::Int,
    rule_count::Int,
    total_pairs::Int,
    threads::Int,
    transform::SymmetryTransform,
) where {T<:AbstractFloat}
    trial_sign_x = T(transform.signs[1])
    trial_sign_y = T(transform.signs[2])
    trial_sign_z = T(transform.signs[3])
    trial_curl_sign_x = T(transform.determinant * transform.signs[1])
    trial_curl_sign_y = T(transform.determinant * transform.signs[2])
    trial_curl_sign_z = T(transform.determinant * transform.signs[3])
    blocks = min(cld(total_pairs, threads), 65_535)
    CUDA.@cuda threads=threads blocks=blocks _cuda_regular_kernel!(
        slp_re,
        slp_im,
        dlp_re,
        dlp_im,
        adj_re,
        adj_im,
        hyp_re,
        hyp_im,
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
        false,
        trial_sign_x,
        trial_sign_y,
        trial_sign_z,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
    )
    return nothing
end

include(joinpath(@__DIR__, "JBEMCudaProfiling.jl"))

function assemble_regular_galerkin_operators_cuda_regular(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    rule::TriangleRule{T};
    skip_singular::Bool=true,
    singular_order::Int=2,
    element_indices=eachindex(mesh.faces),
    cache=nothing,
    return_gpu::Bool=false,
    parallel_quadrature::Bool=true,
    timing=nothing,
    singular_cache=nothing,
    cuda_singular_cache=nothing,
    profile_regular_kernel::Bool=false,
    regular_probe_pair_limit::Int=1_000_000,
    regular_assembly_mode::Symbol=:fused,
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    CUDA.functional() || error("CUDA regular-pair assembly requested, but CUDA.functional() is false.")
    regular_assembly_mode in (:fused, :split_atomic) || error("Unknown regular CUDA assembly mode: $(regular_assembly_mode)")
    regular_assembly_mode == :split_atomic && !parallel_quadrature && error("regular_assembly_mode=:split_atomic requires parallel_quadrature=true")

    indices = cache === nothing ? collect(element_indices) : cache.element_indices
    face_count = cache === nothing ? length(mesh.faces) : cache.face_count
    p1_dof_count = p1_space.global_dof_count
    dp0_dof_count = dp0_space.global_dof_count
    rule_count = cache === nothing ? length(rule.points) : cache.rule_count
    total_pairs = length(indices) * length(indices)
    symmetry_images = symmetry_image_transforms(symmetry_mode)
    kernel_mode = regular_assembly_mode == :split_atomic ? "split_atomic" : (parallel_quadrature ? "parallel_quadrature" : "serial_pair")
    kernel_threads = parallel_quadrature ? _regular_quadrature_threads(rule_count) : 128
    kernel_blocks = parallel_quadrature ? total_pairs : min(cld(total_pairs, kernel_threads), 65_535)
    kernel_shmem = if regular_assembly_mode == :split_atomic
        kernel_threads * 36 * sizeof(T)
    elseif parallel_quadrature
        kernel_threads * 48 * sizeof(T)
    else
        0
    end
    probe_pair_count = regular_probe_pair_limit <= 0 ? total_pairs : min(total_pairs, regular_probe_pair_limit)

    if cache === nothing
        face_vertices, normals, areas, faces, curls = _cuda_geometry_arrays(mesh)
        rule_points, rule_weights = _cuda_rule_arrays(rule)
        color_indices, color_offsets, color_count = _regular_face_color_arrays(mesh, indices)

        _cuda_timed_stage!(timing, "regular_operator_geometry_transfer") do
            d_face_vertices = CuArray(face_vertices)
            d_normals = CuArray(normals)
            d_areas = CuArray(areas)
            d_faces = CuArray(faces)
            d_curls = CuArray(curls)
            d_rule_points = CuArray(rule_points)
            d_rule_weights = CuArray(rule_weights)
            d_test_indices = CuArray(Int32.(indices))
            d_trial_indices = CuArray(Int32.(indices))
            d_color_indices = CuArray(color_indices)
            d_color_offsets = CuArray(color_offsets)
            CUDA.synchronize()
            nothing
        end
    else
        timing !== nothing && (timing["regular_operator_geometry_transfer"] = 0.0)
        d_face_vertices = cache.face_vertices
        d_normals = cache.normals
        d_areas = cache.areas
        d_faces = cache.faces
        d_curls = cache.curls
        d_rule_points = cache.rule_points
        d_rule_weights = cache.rule_weights
        d_test_indices = cache.test_indices
        d_trial_indices = cache.trial_indices
        d_color_indices = cache.color_indices
        d_color_offsets = cache.color_offsets
        color_count = cache.color_count
    end

    slp_re = slp_im = adj_re = adj_im = dlp_re = dlp_im = hyp_re = hyp_im = nothing
    _cuda_timed_stage!(timing, "regular_operator_gpu_alloc") do
        slp_re = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        slp_im = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        adj_re = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        adj_im = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        dlp_re = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        dlp_im = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        hyp_re = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        hyp_im = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        CUDA.synchronize()
        nothing
    end

    if profile_regular_kernel && parallel_quadrature
        _profile_regular_thread_sweep!(
            timing,
            T,
            p1_dof_count,
            dp0_dof_count,
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
        )
        _profile_regular_quadrature_probes!(
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
            k,
            face_count,
            rule_count,
            probe_pair_count,
            kernel_threads,
        )
        _profile_regular_slp_adjoint_colored!(
            timing,
            T,
            p1_dof_count,
            dp0_dof_count,
            d_face_vertices,
            d_normals,
            d_areas,
            d_faces,
            d_color_indices,
            d_color_offsets,
            d_trial_indices,
            d_rule_points,
            d_rule_weights,
            k,
            face_count,
            rule_count,
            color_count,
            kernel_threads,
            regular_probe_pair_limit,
        )
        _profile_regular_split_atomic!(
            timing,
            T,
            p1_dof_count,
            dp0_dof_count,
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
            probe_pair_count,
            kernel_threads,
        )
    else
        timing !== nothing && begin
            timing["regular_operator_probe_green_kernel"] = 0.0
            timing["regular_operator_probe_all_terms_kernel"] = 0.0
        end
        _zero_extended_regular_profile_timings!(timing)
    end

    _cuda_timed_stage!(timing, "regular_operator_kernel") do
        if regular_assembly_mode == :split_atomic
            _launch_regular_split_atomic_kernel!(
                slp_re,
                slp_im,
                dlp_re,
                dlp_im,
                adj_re,
                adj_im,
                hyp_re,
                hyp_im,
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
                kernel_threads,
            )
        elseif parallel_quadrature
            CUDA.@cuda threads=kernel_threads blocks=total_pairs shmem=kernel_shmem _cuda_regular_quadrature_kernel!(
                slp_re,
                slp_im,
                dlp_re,
                dlp_im,
                adj_re,
                adj_im,
                hyp_re,
                hyp_im,
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
        else
            CUDA.@cuda threads=kernel_threads blocks=kernel_blocks _cuda_regular_kernel!(
                slp_re,
                slp_im,
                dlp_re,
                dlp_im,
                adj_re,
                adj_im,
                hyp_re,
                hyp_im,
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
                true,
                one(T),
                one(T),
                one(T),
                one(T),
                one(T),
                one(T),
            )
        end
        for transform in symmetry_images
            _launch_regular_symmetry_image_kernel!(
                slp_re,
                slp_im,
                dlp_re,
                dlp_im,
                adj_re,
                adj_im,
                hyp_re,
                hyp_im,
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
                128,
                transform,
            )
        end
        CUDA.synchronize()
        nothing
    end

    correction_cache = singular_cache
    adjacent_pairs = _cuda_timed_stage!(timing, "regular_operator_count_transfer") do
        if correction_cache === nothing
            count_adjacent_pairs(mesh, indices)
        else
            correction_cache.pair_count
        end
    end
    regular_pairs = total_pairs - adjacent_pairs + length(symmetry_images) * total_pairs

    single_layer = double_layer = adjoint_double_layer = hypersingular = nothing
    if return_gpu
        _cuda_timed_stage!(timing, "regular_operator_complex_materialize") do
            single_layer = _complex_gpu_matrix(slp_re, slp_im)
            double_layer = _complex_gpu_matrix(dlp_re, dlp_im)
            adjoint_double_layer = _complex_gpu_matrix(adj_re, adj_im)
            hypersingular = _complex_gpu_matrix(hyp_re, hyp_im)
            CUDA.synchronize()
            nothing
        end
        timing !== nothing && (timing["regular_operator_cpu_transfer"] = 0.0)
    else
        timing !== nothing && (timing["regular_operator_complex_materialize"] = 0.0)
        _cuda_timed_stage!(timing, "regular_operator_cpu_transfer") do
            single_layer = _complex_cpu_matrix(slp_re, slp_im, T)
            double_layer = _complex_cpu_matrix(dlp_re, dlp_im, T)
            adjoint_double_layer = _complex_cpu_matrix(adj_re, adj_im, T)
            hypersingular = _complex_cpu_matrix(hyp_re, hyp_im, T)
            nothing
        end
    end

    if skip_singular
        singular_pairs = 0
        skipped_pairs = adjacent_pairs
    else
        if return_gpu
            correction_cache === nothing && (correction_cache = build_singular_correction_cache(mesh, singular_order, indices))
            singular_pairs = add_singular_corrections_cuda_compact!(
                (
                    single_layer=single_layer,
                    double_layer=double_layer,
                    adjoint_double_layer=adjoint_double_layer,
                    hypersingular=hypersingular,
                ),
                mesh,
                p1_space,
                dp0_space,
                k,
                singular_order,
                indices,
                correction_cache,
                cuda_singular_cache=cuda_singular_cache,
                cuda_regular_cache=cache,
                timing=timing,
            )
        else
            correction_cache === nothing && (correction_cache = build_singular_correction_cache(mesh, singular_order, indices))
            singular_pairs = assemble_singular_galerkin_corrections!(
                single_layer,
                double_layer,
                adjoint_double_layer,
                hypersingular,
                mesh,
                p1_space,
                dp0_space,
                k,
                singular_order,
                indices,
                correction_cache,
            )
        end
        skipped_pairs = 0
    end

    image_singular_pairs = _cuda_timed_stage!(timing, "regular_operator_image_singular_corrections") do
        if skip_singular || isempty(symmetry_images)
            0
        else
            add_image_singular_corrections_cuda_compact!(
                (
                    single_layer=single_layer,
                    double_layer=double_layer,
                    adjoint_double_layer=adjoint_double_layer,
                    hypersingular=hypersingular,
                    on_gpu=return_gpu,
                ),
                mesh,
                p1_space,
                dp0_space,
                k,
                rule,
                singular_order,
                indices,
                symmetry_mode;
                cuda_regular_cache=cache,
                timing=timing,
            )
        end
    end

    _cuda_timed_stage!(timing, "regular_operator_symmetry_row_weights") do
        _apply_operator_p1_row_weights!(
            (
                single_layer=single_layer,
                double_layer=double_layer,
                adjoint_double_layer=adjoint_double_layer,
                hypersingular=hypersingular,
                on_gpu=return_gpu,
            ),
            mesh,
            symmetry_mode,
        )
    end

    return (
        single_layer=single_layer,
        double_layer=double_layer,
        adjoint_double_layer=adjoint_double_layer,
        hypersingular=hypersingular,
        regular_pairs=regular_pairs,
        singular_pairs=singular_pairs,
        skipped_pairs=skipped_pairs,
        image_singular_pairs=image_singular_pairs,
        on_gpu=return_gpu,
        regular_kernel_threads=kernel_threads,
        regular_kernel_blocks=kernel_blocks,
        regular_kernel_shared_memory_bytes=kernel_shmem,
        regular_kernel_qpair_count=rule_count * rule_count,
        regular_kernel_total_pairs=total_pairs,
        regular_probe_pair_count=probe_pair_count,
        regular_kernel_mode=kernel_mode,
        regular_assembly_mode=regular_assembly_mode,
        regular_color_count=color_count,
    )
end
