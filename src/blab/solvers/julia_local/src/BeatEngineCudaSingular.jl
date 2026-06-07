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
