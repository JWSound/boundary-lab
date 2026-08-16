function assemble_regular_galerkin_operators_rocm_regular(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    rule::TriangleRule{T};
    skip_singular::Bool=true,
    singular_order::Int=2,
    element_indices=eachindex(mesh.faces),
    cache=nothing,
    return_device::Bool=true,
    accelerator_quadrature::Bool=true,
    timing=nothing,
    singular_cache=nothing,
    rocm_singular_cache=nothing,
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    _require_rocm!()
    return_device || error("ROCm host-staged assembly requires return_device=true.")
    accelerator_quadrature || error("ROCm host-staged assembly requires accelerator_quadrature=true.")
    normalized_symmetry_mode(symmetry_mode) == :off ||
        error("The initial BEAT Engine ROCm backend supports only symmetry_mode=:off.")
    rocm_singular_cache === nothing ||
        error("Native ROCm singular-correction caches are not supported by the host-staged backend.")

    host_cache = cache === nothing ? nothing : cache.host_cache
    host_operators = nothing
    host_elapsed = @elapsed begin
        host_operators = assemble_regular_galerkin_operators_cpu(
            mesh,
            p1_space,
            dp0_space,
            k,
            rule;
            skip_singular=skip_singular,
            singular_order=singular_order,
            element_indices=element_indices,
            threaded=true,
            singular_cache=singular_cache,
            cpu_cache=host_cache,
            symmetry_mode=:off,
        )
    end
    timing !== nothing && (timing["rocm_host_assembly"] = host_elapsed)

    device_operators = nothing
    transfer_elapsed = @elapsed begin
        device_operators = (
            single_layer=AMDGPU.ROCArray(host_operators.single_layer),
            double_layer=AMDGPU.ROCArray(host_operators.double_layer),
            adjoint_double_layer=AMDGPU.ROCArray(host_operators.adjoint_double_layer),
            hypersingular=AMDGPU.ROCArray(host_operators.hypersingular),
        )
        AMDGPU.synchronize()
    end
    timing !== nothing && (timing["rocm_operator_upload"] = transfer_elapsed)

    return merge(
        host_operators,
        device_operators,
        (
            on_gpu=true,
            gpu_backend=:rocm,
            host_staged_assembly=true,
            regular_kernel_mode="rocm_host_staged_cpu_assembly",
            regular_assembly_mode=:rocm_host_staged_cpu_assembly,
        ),
    )
end
