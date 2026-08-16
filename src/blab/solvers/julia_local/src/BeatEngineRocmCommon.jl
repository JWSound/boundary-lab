function _rocm_not_implemented(feature::AbstractString)
    error("BEAT Engine ROCm $(feature) is not implemented yet.")
end

struct RocmRegularAssemblyCache{C}
    host_cache::C
end

struct RocmFieldEvaluationCache{C}
    host_cache::C
end

function _require_rocm!(; rocsolver::Bool=false)
    AMDGPU.functional() || error("ROCm solve requested, but AMDGPU.functional() is false.")
    AMDGPU.functional(:rocblas) || error("ROCm solve requested, but rocBLAS is not functional.")
    if rocsolver
        AMDGPU.functional(:rocsolver) || error("ROCm solve requested, but rocSOLVER is not functional.")
    end
    return nothing
end
