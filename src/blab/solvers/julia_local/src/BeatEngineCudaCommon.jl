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
