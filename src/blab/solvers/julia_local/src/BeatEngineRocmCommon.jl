struct RocmSingularCorrectionCache{T}
    pair_offsets
    test_indices
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
    p1_dofs
    element_dp0_dofs
    color_elements
    color_offsets::Vector{Int}
    element_indices::Vector{Int}
    face_count::Int
    p1_dof_count::Int
    dp0_dof_count::Int
    rule_count::Int
    symmetry_mode::Symbol
    image_transforms::Vector{SymmetryTransform}
    image_singular_caches::Vector{RocmSingularCorrectionCache{T}}
    image_singular_pair_count::Int
end

struct RocmFieldEvaluationCache{T}
    source_points
    source_normals
    source_weights
    source_faces
    source_elements
    basis_values
    source_count::Int
end

function _require_rocm!(; rocsolver::Bool=false)
    AMDGPU.functional() || error("ROCm solve requested, but AMDGPU.functional() is false.")
    AMDGPU.functional(:rocblas) || error("ROCm solve requested, but rocBLAS is not functional.")
    if rocsolver
        AMDGPU.functional(:rocsolver) || error("ROCm solve requested, but rocSOLVER is not functional.")
    end
    return nothing
end

function _rocm_kernel_groupsize()
    groupsize = parse(Int, get(ENV, "BLAB_ROCM_KERNEL_GROUPSIZE", "64"))
    groupsize in (32, 64, 128, 256) ||
        error("BLAB_ROCM_KERNEL_GROUPSIZE must be 32, 64, 128, or 256; got $(groupsize).")
    return groupsize
end

function _normalized_rocm_regular_kernel_mode(value=nothing)
    value === nothing && (value = get(ENV, "BLAB_ROCM_REGULAR_KERNEL_MODE", "pair_owned"))
    mode = Symbol(lowercase(strip(String(value))))
    aliases = Dict(
        :pair => :pair_owned,
        :pair_owned => :pair_owned,
        :colored => :pair_owned,
        :colored_pair_owned => :pair_owned,
        :entry => :entry_owned,
        :entry_owned => :entry_owned,
    )
    normalized = get(aliases, mode, nothing)
    normalized === nothing && error(
        "BLAB_ROCM_REGULAR_KERNEL_MODE must be pair_owned or entry_owned; got $(value).",
    )
    return normalized
end
