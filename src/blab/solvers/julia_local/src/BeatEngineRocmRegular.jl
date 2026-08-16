function build_rocm_regular_assembly_cache(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    rule::TriangleRule{T};
    singular_order::Int=2,
    element_indices=eachindex(mesh.faces),
    threaded::Bool=true,
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    normalized_symmetry_mode(symmetry_mode) == :off ||
        error("The initial BEAT Engine ROCm backend supports only symmetry_mode=:off.")
    return RocmRegularAssemblyCache(
        build_beat_cpu_assembly_cache(
            mesh,
            p1_space,
            dp0_space,
            rule;
            singular_order=singular_order,
            element_indices=element_indices,
            threaded=threaded,
            symmetry_mode=:off,
        ),
    )
end
