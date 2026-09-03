function _cuda_bm_identity_kernel!(
    matrix_re,
    matrix_im,
    rhs_re,
    rhs_im,
    areas,
    faces,
    q_neumann,
    inverse_k,
    p1_dof_count,
    face_count,
    rhs_only,
)
    face_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    face_index > face_count && return nothing
    row1 = faces[face_index]
    row2 = faces[face_index + face_count]
    row3 = faces[face_index + 2 * face_count]
    area = areas[face_index]
    diagonal = area / typeof(area)(6)
    off_diagonal = area / typeof(area)(12)

    if !rhs_only
        for (row, col, value) in (
            (row1, row1, diagonal), (row1, row2, off_diagonal), (row1, row3, off_diagonal),
            (row2, row1, off_diagonal), (row2, row2, diagonal), (row2, row3, off_diagonal),
            (row3, row1, off_diagonal), (row3, row2, off_diagonal), (row3, row3, diagonal),
        )
            _cuda_atomic_add!(matrix_re, row + (col - 1) * p1_dof_count, typeof(area)(0.5) * value)
        end
    end

    # -0.5 * (i/k) * M_P1,DP0 * q; every local P1/DP0 entry is area/3.
    rhs_scale = area * inverse_k / typeof(area)(6)
    q = q_neumann[face_index]
    for row in (row1, row2, row3)
        _cuda_atomic_add!(rhs_re, row, rhs_scale * imag(q))
        _cuda_atomic_add!(rhs_im, row, -rhs_scale * real(q))
    end
    return nothing
end

function _cuda_bm_scale_rhs_kernel!(rhs_re, rhs_im, row_weights, dof_count)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index > dof_count && return nothing
    weight = row_weights[index]
    rhs_re[index] *= weight
    rhs_im[index] *= weight
    return nothing
end

function _cuda_bm_scale_rows_kernel!(matrix_re, matrix_im, rhs_re, rhs_im, row_weights, dof_count)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = dof_count * dof_count
    index > total && return nothing
    row = ((index - 1) % dof_count) + 1
    weight = row_weights[row]
    matrix_re[index] *= weight
    matrix_im[index] *= weight
    if index <= dof_count
        rhs_re[index] *= weight
        rhs_im[index] *= weight
    end
    return nothing
end

function _cuda_bm_correction_scatter_kernel!(
    matrix_re,
    matrix_im,
    rhs_re,
    rhs_im,
    q_neumann,
    p1_rows,
    p1_cols,
    dp0_cols,
    slp_values,
    adjoint_values,
    dlp_values,
    hypersingular_values,
    inverse_k,
    p1_dof_count,
    pair_count,
    rhs_only,
)
    pair_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while pair_index <= pair_count
        q = q_neumann[dp0_cols[pair_index]]
        for i in 1:3
            row = p1_rows[pair_index + (i - 1) * pair_count]
            slp = slp_values[pair_index + (i - 1) * pair_count]
            adjoint = adjoint_values[pair_index + (i - 1) * pair_count]
            _cuda_bm_add_rhs!(
                rhs_re,
                rhs_im,
                row,
                real(slp),
                imag(slp),
                real(adjoint),
                imag(adjoint),
                q,
                inverse_k,
            )
        end

        if !rhs_only
            value_index = 1
            for j in 1:3
                col = p1_cols[pair_index + (j - 1) * pair_count]
                for i in 1:3
                    row = p1_rows[pair_index + (i - 1) * pair_count]
                    dlp = dlp_values[pair_index + (value_index - 1) * pair_count]
                    hypersingular = hypersingular_values[pair_index + (value_index - 1) * pair_count]
                    _cuda_bm_add_lhs!(
                        matrix_re,
                        matrix_im,
                        row + (col - 1) * p1_dof_count,
                        real(dlp),
                        imag(dlp),
                        real(hypersingular),
                        imag(hypersingular),
                        inverse_k,
                    )
                    value_index += 1
                end
            end
        end
        pair_index += stride
    end
    return nothing
end

function _launch_cuda_bm_regular_transform!(
    matrix_re,
    matrix_im,
    rhs_re,
    rhs_im,
    q_neumann,
    cache::CudaRegularAssemblyCache{T},
    k::T,
    transform::SymmetryTransform;
    skip_adjacent::Bool,
    rhs_only::Bool=false,
) where {T<:AbstractFloat}
    total_pairs = length(cache.element_indices) * length(cache.element_indices)
    threads = 128
    blocks = min(cld(total_pairs, threads), 65_535)
    placeholder = CUDA.zeros(T, 1)
    try
        signs = T.(transform.signs)
        curl_signs = T.(transform.determinant .* transform.signs)
        CUDA.@cuda threads=threads blocks=blocks _cuda_regular_kernel!(
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            cache.face_vertices,
            cache.normals,
            cache.areas,
            cache.faces,
            cache.curls,
            cache.test_indices,
            cache.trial_indices,
            cache.rule_points,
            cache.rule_weights,
            k,
            size(matrix_re, 1),
            length(q_neumann),
            cache.face_count,
            cache.rule_count,
            total_pairs,
            skip_adjacent,
            signs[1],
            signs[2],
            signs[3],
            curl_signs[1],
            curl_signs[2],
            curl_signs[3],
            true,
            rhs_only,
            matrix_re,
            matrix_im,
            rhs_re,
            rhs_im,
            q_neumann,
        )
        CUDA.synchronize()
    finally
        CUDA.unsafe_free!(placeholder)
    end
    return total_pairs
end

function _cuda_bm_block_arrays(::Type{T}, pair_count::Int) where {T<:AbstractFloat}
    return (
        slp=CUDA.zeros(Complex{T}, pair_count, 3),
        adjoint=CUDA.zeros(Complex{T}, pair_count, 3),
        dlp=CUDA.zeros(Complex{T}, pair_count, 9),
        hypersingular=CUDA.zeros(Complex{T}, pair_count, 9),
    )
end

function _release_cuda_bm_block_arrays!(blocks)
    CUDA.unsafe_free!(blocks.slp)
    CUDA.unsafe_free!(blocks.adjoint)
    CUDA.unsafe_free!(blocks.dlp)
    CUDA.unsafe_free!(blocks.hypersingular)
    return nothing
end

function _scatter_cuda_bm_blocks!(matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, blocks, cache, k; rhs_only::Bool=false)
    cache.pair_count == 0 && return 0
    threads = 128
    blocks_per_grid = min(cld(cache.pair_count, threads), 65_535)
    CUDA.@cuda threads=threads blocks=blocks_per_grid _cuda_bm_correction_scatter_kernel!(
        matrix_re,
        matrix_im,
        rhs_re,
        rhs_im,
        q_neumann,
        cache.p1_rows,
        cache.p1_cols,
        cache.dp0_cols,
        blocks.slp,
        blocks.adjoint,
        blocks.dlp,
        blocks.hypersingular,
        inv(k),
        size(matrix_re, 1),
        cache.pair_count,
        rhs_only,
    )
    CUDA.synchronize()
    return cache.pair_count
end

function add_cuda_bm_singular_corrections!(
    matrix_re,
    matrix_im,
    rhs_re,
    rhs_im,
    q_neumann,
    mesh::BoundaryMesh{T},
    k::T,
    host_cache,
    cuda_cache,
    regular_cache;
    timing=nothing,
    rhs_only::Bool=false,
) where {T<:AbstractFloat}
    host_cache.pair_count == 0 && return 0
    blocks = _cuda_bm_block_arrays(T, host_cache.pair_count)
    try
        _cuda_timed_stage!(timing, "direct_system_singular_compute") do
            threads = 128
            block_count = min(cld(host_cache.pair_count, threads), 65_535)
            CUDA.@cuda threads=threads blocks=block_count _cuda_duffy_blocks_kernel!(
                blocks.slp,
                blocks.adjoint,
                blocks.dlp,
                blocks.hypersingular,
                cuda_cache.test_indices,
                cuda_cache.trial_indices,
                cuda_cache.rule_indices,
                cuda_cache.jac_scales,
                cuda_cache.normal_products,
                cuda_cache.rule_offsets,
                cuda_cache.rule_test_points,
                cuda_cache.rule_trial_points,
                cuda_cache.rule_weights,
                regular_cache.face_vertices,
                regular_cache.normals,
                regular_cache.curls,
                k,
                length(mesh.faces),
                host_cache.pair_count,
            )
            CUDA.synchronize()
        end
        return _cuda_timed_stage!(timing, "direct_system_singular_scatter") do
            _scatter_cuda_bm_blocks!(matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, blocks, cuda_cache, k; rhs_only=rhs_only)
        end
    finally
        _release_cuda_bm_block_arrays!(blocks)
    end
end

function add_cuda_bm_image_corrections!(
    matrix_re,
    matrix_im,
    rhs_re,
    rhs_im,
    q_neumann,
    mesh::BoundaryMesh{T},
    k::T,
    regular_rule::TriangleRule{T},
    cuda_cache,
    regular_cache;
    timing=nothing,
    timing_prefix="direct_system_image",
    rhs_only::Bool=false,
) where {T<:AbstractFloat}
    cuda_cache === nothing && return 0
    cuda_cache.pair_count == 0 && return 0
    blocks = _cuda_bm_block_arrays(T, cuda_cache.pair_count)
    try
        _cuda_timed_stage!(timing, "$(timing_prefix)_compute") do
            threads = 128
            block_count = min(cld(cuda_cache.pair_count, threads), 65_535)
            CUDA.@cuda threads=threads blocks=block_count _cuda_image_singular_delta_blocks_kernel!(
                blocks.slp,
                blocks.adjoint,
                blocks.dlp,
                blocks.hypersingular,
                cuda_cache.test_indices,
                cuda_cache.trial_indices,
                cuda_cache.rule_indices,
                cuda_cache.jac_scales,
                cuda_cache.normal_products,
                cuda_cache.rule_offsets,
                cuda_cache.rule_test_points,
                cuda_cache.rule_trial_points,
                cuda_cache.rule_weights,
                regular_cache.rule_points,
                regular_cache.rule_weights,
                cuda_cache.transform_signs,
                cuda_cache.curl_signs,
                regular_cache.face_vertices,
                regular_cache.normals,
                regular_cache.curls,
                k,
                length(mesh.faces),
                length(regular_rule.weights),
                cuda_cache.pair_count,
            )
            CUDA.synchronize()
        end
        return _cuda_timed_stage!(timing, "$(timing_prefix)_scatter") do
            _scatter_cuda_bm_blocks!(matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, blocks, cuda_cache, k; rhs_only=rhs_only)
        end
    finally
        _release_cuda_bm_block_arrays!(blocks)
    end
end

function assemble_burton_miller_neumann_system_cuda(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    q_neumann::CuArray,
    k::T,
    rule::TriangleRule{T};
    device_cache,
    singular_cache,
    device_singular_cache,
    device_image_singular_cache=nothing,
    near_correction_cache=nothing,
    device_near_correction_cache=nothing,
    image_near_correction_cache=nothing,
    device_image_near_correction_cache=nothing,
    symmetry_mode::Symbol=:off,
    timing=nothing,
) where {T<:AbstractFloat}
    CUDA.functional() || error("Direct Burton-Miller CUDA assembly requested, but CUDA.functional() is false.")
    length(q_neumann) == dp0_space.global_dof_count || error("Direct Burton-Miller Neumann vector size mismatch.")
    device_cache === nothing && error("Direct Burton-Miller CUDA assembly requires a regular device cache.")
    singular_cache === nothing && error("Direct Burton-Miller CUDA assembly requires a singular correction cache.")
    device_singular_cache === nothing && error(
        "Direct Burton-Miller CUDA assembly requires a singular device cache.",
    )
    if !isempty(symmetry_image_transforms(symmetry_mode)) && device_image_singular_cache === nothing
        error("Direct Burton-Miller symmetry assembly requires an image-singular device cache.")
    end
    if near_correction_cache !== nothing && near_correction_cache.pair_count > 0 &&
       device_near_correction_cache === nothing
        error("Direct Burton-Miller near correction requires a matching device cache.")
    end
    if image_near_correction_cache !== nothing && image_near_correction_cache.pair_count > 0 &&
       device_image_near_correction_cache === nothing
        error("Direct Burton-Miller image-near correction requires a matching device cache.")
    end
    p1_count = p1_space.global_dof_count
    # Store real and imaginary lanes interleaved so the assembled storage can
    # be reinterpreted as Complex{T} without allocating a second dense matrix.
    # This removes the 3x dense-matrix peak (real + imaginary + complex) that
    # otherwise exhausts an 11 GiB GPU for eight moderate speaker boundaries.
    matrix_storage = CUDA.zeros(T, 2, p1_count, p1_count)
    matrix_re = view(matrix_storage, 1, :, :)
    matrix_im = view(matrix_storage, 2, :, :)
    rhs_re = CUDA.zeros(T, p1_count)
    rhs_im = CUDA.zeros(T, p1_count)
    matrix = rhs = nothing
    succeeded = false
    try
        identity_transform = symmetry_transforms(:off; include_identity=true)[1]
        _cuda_timed_stage!(timing, "direct_system_regular") do
            _launch_cuda_bm_regular_transform!(
                matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, device_cache, k, identity_transform;
                skip_adjacent=true,
            )
        end
        for transform in symmetry_image_transforms(symmetry_mode)
            _cuda_timed_stage!(timing, "direct_system_regular_image") do
                _launch_cuda_bm_regular_transform!(
                    matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, device_cache, k, transform;
                    skip_adjacent=false,
                )
            end
        end

        singular_pairs = add_cuda_bm_singular_corrections!(
            matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
            mesh, k, singular_cache, device_singular_cache, device_cache;
            timing=timing,
        )
        image_singular_pairs = add_cuda_bm_image_corrections!(
            matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
            mesh, k, rule, device_image_singular_cache, device_cache;
            timing=timing,
        )
        near_pair_count = 0
        if near_correction_cache !== nothing && near_correction_cache.pair_count > 0
            near_pair_count += add_cuda_bm_image_corrections!(
                matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
                mesh, k, rule, device_near_correction_cache, device_cache;
                timing=timing,
                timing_prefix="direct_system_near",
            )
        end
        if image_near_correction_cache !== nothing && image_near_correction_cache.pair_count > 0
            near_pair_count += add_cuda_bm_image_corrections!(
                matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
                mesh, k, rule, device_image_near_correction_cache, device_cache;
                timing=timing,
                timing_prefix="direct_system_ground_near",
            )
        end

        _cuda_timed_stage!(timing, "direct_system_identity") do
            threads = 256
            blocks = cld(length(mesh.faces), threads)
            CUDA.@cuda threads=threads blocks=blocks _cuda_bm_identity_kernel!(
                matrix_re,
                matrix_im,
                rhs_re,
                rhs_im,
                device_cache.areas,
                device_cache.faces,
                q_neumann,
                inv(k),
                p1_count,
                length(mesh.faces),
                false,
            )
            CUDA.synchronize()
        end

        _cuda_timed_stage!(timing, "direct_system_row_weights") do
            row_weights = CuArray(p1_symmetry_orbit_weights(mesh, symmetry_mode))
            try
                threads = 256
                blocks = cld(p1_count * p1_count, threads)
                CUDA.@cuda threads=threads blocks=blocks _cuda_bm_scale_rows_kernel!(
                    matrix_re, matrix_im, rhs_re, rhs_im, row_weights, p1_count,
                )
                CUDA.synchronize()
            finally
                CUDA.unsafe_free!(row_weights)
            end
        end

        _cuda_timed_stage!(timing, "direct_system_complex_materialize") do
            matrix = reshape(reinterpret(Complex{T}, matrix_storage), p1_count, p1_count)
            rhs = complex.(rhs_re, rhs_im)
            CUDA.synchronize()
        end
        succeeded = true
        return (
            matrix=matrix,
            rhs=rhs,
            regular_pairs=length(device_cache.element_indices)^2 * symmetry_reduction_factor(symmetry_mode) - singular_cache.pair_count,
            singular_pairs=singular_pairs,
            image_singular_pairs=image_singular_pairs,
            near_pair_count=near_pair_count,
            on_gpu=true,
            assembly_mode=:direct_burton_miller,
        )
    finally
        CUDA.unsafe_free!(rhs_re)
        CUDA.unsafe_free!(rhs_im)
        if !succeeded
            CUDA.unsafe_free!(matrix_storage)
            rhs === nothing || CUDA.unsafe_free!(rhs)
        end
    end
end

function assemble_burton_miller_rhs_cuda(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    q_neumann::CuArray,
    k::T,
    rule::TriangleRule{T};
    device_cache,
    singular_cache,
    device_singular_cache,
    device_image_singular_cache=nothing,
    near_correction_cache=nothing,
    device_near_correction_cache=nothing,
    image_near_correction_cache=nothing,
    device_image_near_correction_cache=nothing,
    symmetry_mode::Symbol=:off,
    timing=nothing,
) where {T<:AbstractFloat}
    CUDA.functional() || error("Burton-Miller CUDA RHS assembly requested, but CUDA.functional() is false.")
    length(q_neumann) == dp0_space.global_dof_count || error("Burton-Miller Neumann vector size mismatch.")
    device_cache === nothing && error("Burton-Miller CUDA RHS assembly requires a regular device cache.")
    singular_cache === nothing && error("Burton-Miller CUDA RHS assembly requires a singular correction cache.")
    device_singular_cache === nothing && error("Burton-Miller CUDA RHS assembly requires a singular device cache.")
    if !isempty(symmetry_image_transforms(symmetry_mode)) && device_image_singular_cache === nothing
        error("Burton-Miller symmetry RHS assembly requires an image-singular device cache.")
    end
    if near_correction_cache !== nothing && near_correction_cache.pair_count > 0 &&
       device_near_correction_cache === nothing
        error("Burton-Miller near RHS correction requires a matching device cache.")
    end
    if image_near_correction_cache !== nothing && image_near_correction_cache.pair_count > 0 &&
       device_image_near_correction_cache === nothing
        error("Burton-Miller image-near RHS correction requires a matching device cache.")
    end

    p1_count = p1_space.global_dof_count
    matrix_re = CUDA.zeros(T, 1)
    matrix_im = CUDA.zeros(T, 1)
    rhs_re = CUDA.zeros(T, p1_count)
    rhs_im = CUDA.zeros(T, p1_count)
    rhs = nothing
    succeeded = false
    try
        identity_transform = symmetry_transforms(:off; include_identity=true)[1]
        _cuda_timed_stage!(timing, "rhs_regular") do
            _launch_cuda_bm_regular_transform!(
                matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, device_cache, k, identity_transform;
                skip_adjacent=true,
                rhs_only=true,
            )
        end
        for transform in symmetry_image_transforms(symmetry_mode)
            _cuda_timed_stage!(timing, "rhs_regular_image") do
                _launch_cuda_bm_regular_transform!(
                    matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, device_cache, k, transform;
                    skip_adjacent=false,
                    rhs_only=true,
                )
            end
        end

        add_cuda_bm_singular_corrections!(
            matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
            mesh, k, singular_cache, device_singular_cache, device_cache;
            timing=timing,
            rhs_only=true,
        )
        add_cuda_bm_image_corrections!(
            matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
            mesh, k, rule, device_image_singular_cache, device_cache;
            timing=timing,
            rhs_only=true,
        )
        if near_correction_cache !== nothing && near_correction_cache.pair_count > 0
            add_cuda_bm_image_corrections!(
                matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
                mesh, k, rule, device_near_correction_cache, device_cache;
                timing=timing,
                timing_prefix="rhs_near",
                rhs_only=true,
            )
        end
        if image_near_correction_cache !== nothing && image_near_correction_cache.pair_count > 0
            add_cuda_bm_image_corrections!(
                matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
                mesh, k, rule, device_image_near_correction_cache, device_cache;
                timing=timing,
                timing_prefix="rhs_ground_near",
                rhs_only=true,
            )
        end

        _cuda_timed_stage!(timing, "rhs_identity") do
            threads = 256
            blocks = cld(length(mesh.faces), threads)
            CUDA.@cuda threads=threads blocks=blocks _cuda_bm_identity_kernel!(
                matrix_re,
                matrix_im,
                rhs_re,
                rhs_im,
                device_cache.areas,
                device_cache.faces,
                q_neumann,
                inv(k),
                p1_count,
                length(mesh.faces),
                true,
            )
            CUDA.synchronize()
        end

        _cuda_timed_stage!(timing, "rhs_row_weights") do
            row_weights = CuArray(p1_symmetry_orbit_weights(mesh, symmetry_mode))
            try
                threads = 256
                blocks = cld(p1_count, threads)
                CUDA.@cuda threads=threads blocks=blocks _cuda_bm_scale_rhs_kernel!(
                    rhs_re, rhs_im, row_weights, p1_count,
                )
                CUDA.synchronize()
            finally
                CUDA.unsafe_free!(row_weights)
            end
        end

        _cuda_timed_stage!(timing, "rhs_complex_materialize") do
            rhs = complex.(rhs_re, rhs_im)
            CUDA.synchronize()
        end
        succeeded = true
        return rhs
    finally
        CUDA.unsafe_free!(matrix_re)
        CUDA.unsafe_free!(matrix_im)
        CUDA.unsafe_free!(rhs_re)
        CUDA.unsafe_free!(rhs_im)
        (!succeeded && rhs !== nothing) && CUDA.unsafe_free!(rhs)
    end
end

function solve_burton_miller_system_cuda!(system; return_gpu::Bool=false)
    get(system, :on_gpu, false) || error("Direct Burton-Miller CUDA solve requires a GPU-resident system.")
    factorization = pressure = nothing
    try
        factorization = lu!(system.matrix)
        pressure = factorization \ system.rhs
        return return_gpu ? pressure : Array(pressure)
    finally
        if factorization === nothing
            CUDA.unsafe_free!(system.matrix)
        else
            CUDA.unsafe_free!(factorization.factors)
            CUDA.unsafe_free!(factorization.ipiv)
        end
        CUDA.unsafe_free!(system.rhs)
        (!return_gpu && pressure !== nothing) && CUDA.unsafe_free!(pressure)
    end
end

function release_burton_miller_system_cuda!(system)
    CUDA.unsafe_free!(system.matrix)
    CUDA.unsafe_free!(system.rhs)
    return nothing
end
