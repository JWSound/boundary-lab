function _rocm_not_implemented(feature::AbstractString)
    error("BEAT Engine ROCm $(feature) is not implemented yet.")
end

struct RocmRegularAssemblyCache{T,C}
    host_cache::C
    face_vertices
    normals
    areas
    faces
    curls
    rule_points
    rule_weights
    vertex_offsets
    incident_elements
    incident_local_indices
    dp0_elements
    element_indices::Vector{Int}
    face_count::Int
    p1_dof_count::Int
    dp0_dof_count::Int
    rule_count::Int
end

struct RocmFieldEvaluationCache{C}
    host_cache::C
end

struct RocmSingularCorrectionCache{T}
    pair_offsets
    trial_indices
    rule_indices
    jac_scales
    normal_products
    rule_offsets
    rule_test_points
    rule_trial_points
    rule_weights
    pair_count::Int
end

function _require_rocm!(; rocsolver::Bool=false)
    AMDGPU.functional() || error("ROCm solve requested, but AMDGPU.functional() is false.")
    AMDGPU.functional(:rocblas) || error("ROCm solve requested, but rocBLAS is not functional.")
    if rocsolver
        AMDGPU.functional(:rocsolver) || error("ROCm solve requested, but rocSOLVER is not functional.")
    end
    return nothing
end
