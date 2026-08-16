@inline function _rocm_global_linear_index()
    return (AMDGPU.workgroupIdx().x - 1) * AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x
end

@inline function _rocm_faces_are_adjacent(faces, test_index, trial_index, face_count)
    t1 = faces[test_index]
    t2 = faces[test_index + face_count]
    t3 = faces[test_index + 2 * face_count]
    r1 = faces[trial_index]
    r2 = faces[trial_index + face_count]
    r3 = faces[trial_index + 2 * face_count]
    return t1 == r1 || t1 == r2 || t1 == r3 ||
        t2 == r1 || t2 == r2 || t2 == r3 ||
        t3 == r1 || t3 == r2 || t3 == r3
end

@inline function _rocm_basis_value(local_index, basis1, basis2, basis3)
    return local_index == 1 ? basis1 : local_index == 2 ? basis2 : basis3
end

@inline function _rocm_find_pair(pair_offsets, trial_indices, test_index, trial_index)
    pair_position = pair_offsets[test_index]
    pair_stop = pair_offsets[test_index + 1] - 1
    while pair_position <= pair_stop
        trial_indices[pair_position] == trial_index && return pair_position
        pair_position += 1
    end
    return zero(pair_position)
end

@inline function _rocm_face_point(face_vertices, element_index, face_count, basis1, basis2, basis3)
    x = basis1 * face_vertices[element_index] +
        basis2 * face_vertices[element_index + 3 * face_count] +
        basis3 * face_vertices[element_index + 6 * face_count]
    y = basis1 * face_vertices[element_index + face_count] +
        basis2 * face_vertices[element_index + 4 * face_count] +
        basis3 * face_vertices[element_index + 7 * face_count]
    z = basis1 * face_vertices[element_index + 2 * face_count] +
        basis2 * face_vertices[element_index + 5 * face_count] +
        basis3 * face_vertices[element_index + 8 * face_count]
    return x, y, z
end

function _rocm_regular_slp_adjoint_entries_kernel!(
    single_layer,
    adjoint_double_layer,
    face_vertices,
    normals,
    areas,
    faces,
    rule_points,
    rule_weights,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    dp0_elements,
    k,
    p1_dof_count,
    dp0_dof_count,
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
    total_entries = p1_dof_count * dp0_dof_count
    linear_index > total_entries && return nothing

    row = ((linear_index - 1) % p1_dof_count) + 1
    column = div(linear_index - 1, p1_dof_count) + 1
    trial_index = dp0_elements[column]
    if trial_index == 0
        single_layer[linear_index] = zero(eltype(single_layer))
        adjoint_double_layer[linear_index] = zero(eltype(adjoint_double_layer))
        return nothing
    end

    four_pi = typeof(k)(12.566370614359172)
    slp_re = zero(k)
    slp_im = zero(k)
    adj_re = zero(k)
    adj_im = zero(k)
    trial_nx = trial_sign_x * normals[trial_index]
    trial_ny = trial_sign_y * normals[trial_index + face_count]
    trial_nz = trial_sign_z * normals[trial_index + 2 * face_count]
    trial_area = areas[trial_index]

    incident_position = vertex_offsets[row]
    incident_stop = vertex_offsets[row + 1] - 1
    while incident_position <= incident_stop
        test_index = incident_elements[incident_position]
        local_row = incident_local_indices[incident_position]
        skip_pair = skip_mode == 0 ?
            _rocm_faces_are_adjacent(faces, test_index, trial_index, face_count) :
            skip_mode == 1 && _rocm_find_pair(pair_offsets, singular_trial_indices, test_index, trial_index) != 0
        if !skip_pair
            test_nx = normals[test_index]
            test_ny = normals[test_index + face_count]
            test_nz = normals[test_index + 2 * face_count]
            jac_scale = typeof(k)(4) * areas[test_index] * trial_area

            test_q = 1
            while test_q <= rule_count
                test_xi = rule_points[test_q]
                test_eta = rule_points[test_q + rule_count]
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
                test_weight = rule_weights[test_q]

                trial_q = 1
                while trial_q <= rule_count
                    trial_xi = rule_points[trial_q]
                    trial_eta = rule_points[trial_q + rule_count]
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
                        weight = test_weight * rule_weights[trial_q] * jac_scale * test_basis
                        slp_re += green_re * weight
                        slp_im += green_im * weight

                        test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
                        grad_re = -green_re * inv_radius - green_im * k
                        grad_im = green_re * k - green_im * inv_radius
                        adj_re += grad_re * test_dot * weight
                        adj_im += grad_im * test_dot * weight
                    end
                    trial_q += 1
                end
                test_q += 1
            end
        end
        incident_position += 1
    end

    single_layer[linear_index] += Complex(slp_re, slp_im)
    adjoint_double_layer[linear_index] += Complex(adj_re, adj_im)
    return nothing
end

function _rocm_regular_dlp_hyp_entries_kernel!(
    double_layer,
    hypersingular,
    face_vertices,
    normals,
    areas,
    faces,
    curls,
    rule_points,
    rule_weights,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
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
    total_entries = p1_dof_count * p1_dof_count
    linear_index > total_entries && return nothing

    row = ((linear_index - 1) % p1_dof_count) + 1
    column = div(linear_index - 1, p1_dof_count) + 1
    four_pi = typeof(k)(12.566370614359172)
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
        test_nx = normals[test_index]
        test_ny = normals[test_index + face_count]
        test_nz = normals[test_index + 2 * face_count]
        test_curl_offset = 3 * (local_row - 1)
        test_curl_x = curls[test_index + test_curl_offset * face_count]
        test_curl_y = curls[test_index + (test_curl_offset + 1) * face_count]
        test_curl_z = curls[test_index + (test_curl_offset + 2) * face_count]

        trial_position = vertex_offsets[column]
        trial_stop = vertex_offsets[column + 1] - 1
        while trial_position <= trial_stop
            trial_index = incident_elements[trial_position]
            local_column = incident_local_indices[trial_position]
            skip_pair = skip_mode == 0 ?
                _rocm_faces_are_adjacent(faces, test_index, trial_index, face_count) :
                skip_mode == 1 && _rocm_find_pair(pair_offsets, singular_trial_indices, test_index, trial_index) != 0
            if !skip_pair
                trial_nx = trial_sign_x * normals[trial_index]
                trial_ny = trial_sign_y * normals[trial_index + face_count]
                trial_nz = trial_sign_z * normals[trial_index + 2 * face_count]
                normal_product = test_nx * trial_nx + test_ny * trial_ny + test_nz * trial_nz
                trial_curl_offset = 3 * (local_column - 1)
                curl_product =
                    test_curl_x * trial_curl_sign_x * curls[trial_index + trial_curl_offset * face_count] +
                    test_curl_y * trial_curl_sign_y * curls[trial_index + (trial_curl_offset + 1) * face_count] +
                    test_curl_z * trial_curl_sign_z * curls[trial_index + (trial_curl_offset + 2) * face_count]
                jac_scale = typeof(k)(4) * areas[test_index] * areas[trial_index]

                test_q = 1
                while test_q <= rule_count
                    test_xi = rule_points[test_q]
                    test_eta = rule_points[test_q + rule_count]
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
                    test_weight = rule_weights[test_q]

                    trial_q = 1
                    while trial_q <= rule_count
                        trial_xi = rule_points[trial_q]
                        trial_eta = rule_points[trial_q + rule_count]
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
                            basis_product = test_basis * trial_basis
                            weight = test_weight * rule_weights[trial_q] * jac_scale
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
                        trial_q += 1
                    end
                    test_q += 1
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

function _launch_rocm_regular_entry_kernels!(operators, cache::RocmRegularAssemblyCache, k)
    groupsize = _rocm_kernel_groupsize()
    slp_entries = cache.p1_dof_count * cache.dp0_dof_count
    p1_entries = cache.p1_dof_count * cache.p1_dof_count
    AMDGPU.@roc groupsize=groupsize gridsize=cld(slp_entries, groupsize) _rocm_regular_slp_adjoint_entries_kernel!(
        operators.single_layer,
        operators.adjoint_double_layer,
        cache.face_vertices,
        cache.normals,
        cache.areas,
        cache.faces,
        cache.rule_points,
        cache.rule_weights,
        cache.vertex_offsets,
        cache.incident_elements,
        cache.incident_local_indices,
        cache.dp0_elements,
        k,
        cache.p1_dof_count,
        cache.dp0_dof_count,
        cache.face_count,
        cache.rule_count,
        cache.vertex_offsets,
        cache.incident_elements,
        Int32(0),
        one(k),
        one(k),
        one(k),
    )
    AMDGPU.@roc groupsize=groupsize gridsize=cld(p1_entries, groupsize) _rocm_regular_dlp_hyp_entries_kernel!(
        operators.double_layer,
        operators.hypersingular,
        cache.face_vertices,
        cache.normals,
        cache.areas,
        cache.faces,
        cache.curls,
        cache.rule_points,
        cache.rule_weights,
        cache.vertex_offsets,
        cache.incident_elements,
        cache.incident_local_indices,
        k,
        cache.p1_dof_count,
        cache.face_count,
        cache.rule_count,
        cache.vertex_offsets,
        cache.incident_elements,
        Int32(0),
        one(k),
        one(k),
        one(k),
        one(k),
        one(k),
        one(k),
    )
    return nothing
end

function _launch_rocm_symmetry_regular_entry_kernels!(
    operators,
    cache::RocmRegularAssemblyCache,
    image_cache::RocmSingularCorrectionCache,
    transform::SymmetryTransform,
    k;
    skip_image_singular::Bool,
)
    groupsize = _rocm_kernel_groupsize()
    slp_entries = cache.p1_dof_count * cache.dp0_dof_count
    p1_entries = cache.p1_dof_count * cache.p1_dof_count
    sx = typeof(k)(transform.signs[1])
    sy = typeof(k)(transform.signs[2])
    sz = typeof(k)(transform.signs[3])
    csx = typeof(k)(transform.determinant * transform.signs[1])
    csy = typeof(k)(transform.determinant * transform.signs[2])
    csz = typeof(k)(transform.determinant * transform.signs[3])
    skip_mode = skip_image_singular ? Int32(1) : Int32(2)
    AMDGPU.@roc groupsize=groupsize gridsize=cld(slp_entries, groupsize) _rocm_regular_slp_adjoint_entries_kernel!(
        operators.single_layer,
        operators.adjoint_double_layer,
        cache.face_vertices,
        cache.normals,
        cache.areas,
        cache.faces,
        cache.rule_points,
        cache.rule_weights,
        cache.vertex_offsets,
        cache.incident_elements,
        cache.incident_local_indices,
        cache.dp0_elements,
        k,
        cache.p1_dof_count,
        cache.dp0_dof_count,
        cache.face_count,
        cache.rule_count,
        image_cache.pair_offsets,
        image_cache.trial_indices,
        skip_mode,
        sx,
        sy,
        sz,
    )
    AMDGPU.@roc groupsize=groupsize gridsize=cld(p1_entries, groupsize) _rocm_regular_dlp_hyp_entries_kernel!(
        operators.double_layer,
        operators.hypersingular,
        cache.face_vertices,
        cache.normals,
        cache.areas,
        cache.faces,
        cache.curls,
        cache.rule_points,
        cache.rule_weights,
        cache.vertex_offsets,
        cache.incident_elements,
        cache.incident_local_indices,
        k,
        cache.p1_dof_count,
        cache.face_count,
        cache.rule_count,
        image_cache.pair_offsets,
        image_cache.trial_indices,
        skip_mode,
        sx,
        sy,
        sz,
        csx,
        csy,
        csz,
    )
    return nothing
end
