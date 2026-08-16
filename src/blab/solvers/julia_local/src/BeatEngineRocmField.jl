function build_rocm_field_evaluation_cache(cache::FieldEvaluationCache{T}) where {T<:AbstractFloat}
    _require_rocm!()
    return RocmFieldEvaluationCache(cache)
end

function build_rocm_field_evaluation_cache(
    mesh::BoundaryMesh{T},
    rule::TriangleRule{T};
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    normalized_symmetry_mode(symmetry_mode) == :off ||
        error("The initial BEAT Engine ROCm backend supports only symmetry_mode=:off.")
    return build_rocm_field_evaluation_cache(build_field_evaluation_cache(mesh, rule; symmetry_mode=:off))
end

function evaluate_galerkin_field_rocm(
    eval_points,
    mesh::BoundaryMesh{T},
    pressure,
    q_neumann,
    k::T,
    cache::RocmFieldEvaluationCache,
) where {T<:AbstractFloat}
    # The first correctness milestone keeps field reconstruction on the trusted CPU
    # path. Native ROCm field kernels are a separate performance milestone.
    host_pressure = pressure isa Array ? pressure : Array(pressure)
    host_neumann = q_neumann isa Array ? q_neumann : Array(q_neumann)
    return evaluate_galerkin_field_cpu(eval_points, mesh, host_pressure, host_neumann, k, cache.host_cache)
end
