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

_regular_quadrature_threads(rule_count::Int) = 16

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

function _launch_regular_split_balanced_atomic_kernel!(
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
    shmem = threads * 24 * sizeof(T)
    CUDA.@cuda threads=threads blocks=total_pairs shmem=shmem _cuda_regular_quadrature_slp_hyp_kernel!(
        slp_re,
        slp_im,
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

    CUDA.@cuda threads=threads blocks=total_pairs shmem=shmem _cuda_regular_quadrature_dlp_adjoint_kernel!(
        dlp_re,
        dlp_im,
        adj_re,
        adj_im,
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
