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

struct CudaSparseScatterCache{R,C,V}
    rows::R
    columns::C
    values::V
end

function build_cuda_sparse_scatter_cache(matrix::SparseMatrixCSC)
    CUDA.functional() || error("CUDA sparse scatter cache requested, but CUDA.functional() is false.")
    rows, columns, values = findnz(matrix)
    return CudaSparseScatterCache(
        CuArray(Int32.(rows)),
        CuArray(Int32.(columns)),
        CuArray(values),
    )
end

function _cuda_sparse_scatter_kernel!(
    destination,
    rows,
    columns,
    values,
    row_offset::Int32,
    column_offset::Int32,
    alpha,
    add::Bool,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= length(values)
        row = rows[index] + row_offset
        column = columns[index] + column_offset
        value = alpha * values[index]
        if add
            destination[row, column] += value
        else
            destination[row, column] = value
        end
    end
    return
end

function scatter_cuda_sparse_to_dense!(
    destination,
    cache::CudaSparseScatterCache;
    row_offset::Integer=0,
    column_offset::Integer=0,
    alpha=one(eltype(destination)),
    add::Bool=false,
)
    isempty(cache.values) && return destination
    threads = 256
    blocks = cld(length(cache.values), threads)
    CUDA.@cuda threads=threads blocks=blocks _cuda_sparse_scatter_kernel!(
        destination,
        cache.rows,
        cache.columns,
        cache.values,
        Int32(row_offset),
        Int32(column_offset),
        convert(eltype(destination), alpha),
        add,
    )
    return destination
end

function release_cuda_sparse_scatter_cache!(cache::CudaSparseScatterCache)
    CUDA.unsafe_free!(cache.rows)
    CUDA.unsafe_free!(cache.columns)
    CUDA.unsafe_free!(cache.values)
    return nothing
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

function _launch_regular_serial_pair_batched_kernel!(
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
) where {T<:AbstractFloat}
    threads = 128
    blocks = cld(total_pairs, threads)
    # The q4 split kernels naturally compile to about 100 registers/thread. On CC 7.5,
    # capping at 80 admits six resident blocks (75% theoretical occupancy); the
    # resulting spills remain L2-resident and benchmark faster than the 72/84/88
    # register variants on sample_detailed.msh.
    CUDA.@cuda threads=threads blocks=blocks maxregs=80 _cuda_regular_quadrature_slp_hyp_kernel!(
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

    CUDA.@cuda threads=threads blocks=blocks maxregs=80 _cuda_regular_quadrature_dlp_adjoint_kernel!(
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
        false,
        false,
        slp_re,
        slp_im,
        adj_re,
        adj_im,
        slp_re,
    )
    return nothing
end
