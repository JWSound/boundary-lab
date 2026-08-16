function _normalized_rocm_assembly_mode(value)
    value === nothing && (value = get(ENV, "BLAB_ROCM_ASSEMBLY_MODE", "native"))
    mode = Symbol(lowercase(strip(String(value))))
    aliases = Dict(
        :host => :host_staged,
        :staged => :host_staged,
        :host_staged => :host_staged,
        :native => :native,
        :native_regular => :native,
    )
    normalized = get(aliases, mode, nothing)
    normalized === nothing && error("ROCm assembly mode must be host_staged or native; got $(value).")
    return normalized
end

function _assemble_regular_galerkin_operators_rocm_native(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    rule::TriangleRule{T};
    skip_singular::Bool,
    singular_order::Int,
    element_indices,
    cache,
    timing,
    singular_cache,
    rocm_singular_cache,
    symmetry_mode::Symbol,
) where {T<:AbstractFloat}
    normalized_mode = normalized_symmetry_mode(symmetry_mode)
    native_cache = cache === nothing ? build_rocm_regular_assembly_cache(
        mesh,
        p1_space,
        dp0_space,
        rule;
        singular_order=singular_order,
        element_indices=element_indices,
        symmetry_mode=normalized_mode,
    ) : cache
    native_cache isa RocmRegularAssemblyCache || error("Native ROCm assembly requires a RocmRegularAssemblyCache.")
    native_cache.symmetry_mode == normalized_mode ||
        error("ROCm assembly cache symmetry mode $(native_cache.symmetry_mode) does not match requested $(normalized_mode).")

    operators = nothing
    allocation_elapsed = @elapsed begin
        operators = (
            single_layer=AMDGPU.zeros(Complex{T}, p1_space.global_dof_count, dp0_space.global_dof_count),
            double_layer=AMDGPU.zeros(Complex{T}, p1_space.global_dof_count, p1_space.global_dof_count),
            adjoint_double_layer=AMDGPU.zeros(Complex{T}, p1_space.global_dof_count, dp0_space.global_dof_count),
            hypersingular=AMDGPU.zeros(Complex{T}, p1_space.global_dof_count, p1_space.global_dof_count),
        )
        AMDGPU.synchronize()
    end
    timing !== nothing && (timing["rocm_native_operator_alloc"] = allocation_elapsed)

    regular_kernel_mode = _normalized_rocm_regular_kernel_mode()
    kernel_elapsed = @elapsed begin
        if regular_kernel_mode == :pair_owned
            _launch_rocm_regular_pair_kernels!(operators, native_cache, k)
        else
            _launch_rocm_regular_entry_kernels!(operators, native_cache, k)
        end
        for (transform, image_cache) in zip(native_cache.image_transforms, native_cache.image_singular_caches)
            if regular_kernel_mode == :pair_owned
                _launch_rocm_symmetry_regular_pair_kernels!(
                    operators,
                    native_cache,
                    image_cache,
                    transform,
                    k;
                    skip_image_singular=!skip_singular,
                )
            else
                _launch_rocm_symmetry_regular_entry_kernels!(
                    operators,
                    native_cache,
                    image_cache,
                    transform,
                    k;
                    skip_image_singular=!skip_singular,
                )
            end
        end
        AMDGPU.synchronize()
    end
    timing !== nothing && (timing["rocm_native_regular_kernel"] = kernel_elapsed)

    indices = native_cache.element_indices
    correction_cache = singular_cache === nothing ? build_singular_correction_cache(mesh, singular_order, indices) : singular_cache
    adjacent_pairs = correction_cache.pair_count
    singular_pairs = 0
    if !skip_singular
        owns_device_singular_cache = rocm_singular_cache === nothing
        device_singular_cache = rocm_singular_cache
        cache_elapsed = @elapsed begin
            if device_singular_cache === nothing
                device_singular_cache = build_rocm_singular_correction_cache(correction_cache)
            end
        end
        timing !== nothing && (timing["rocm_native_singular_cache"] = cache_elapsed)
        singular_elapsed = @elapsed begin
            _launch_rocm_singular_block_gather_kernels!(operators, native_cache, device_singular_cache, k)
            AMDGPU.synchronize()
        end
        timing !== nothing && (timing["rocm_native_singular_kernel"] = singular_elapsed)
        singular_pairs = correction_cache.pair_count
        owns_device_singular_cache && _release_rocm_singular_correction_cache!(device_singular_cache)
    end
    image_singular_pairs = 0
    if !skip_singular
        image_elapsed = @elapsed begin
            for (transform, image_cache) in zip(native_cache.image_transforms, native_cache.image_singular_caches)
                image_cache.pair_count == 0 && continue
                _launch_rocm_singular_block_gather_kernels!(operators, native_cache, image_cache, k, transform)
            end
            AMDGPU.synchronize()
        end
        timing !== nothing && (timing["rocm_native_image_singular_kernel"] = image_elapsed)
        image_singular_pairs = native_cache.image_singular_pair_count
    end
    weight_elapsed = @elapsed _apply_rocm_operator_p1_row_weights!(operators, mesh, normalized_mode)
    timing !== nothing && (timing["rocm_native_symmetry_row_weights"] = weight_elapsed)
    total_pairs = length(indices) * length(indices)
    image_count = length(native_cache.image_transforms)
    kernel_groupsize = _rocm_kernel_groupsize()
    color_count = length(native_cache.color_offsets) - 1
    return merge(
        operators,
        (
            regular_pairs=total_pairs - adjacent_pairs + image_count * total_pairs - image_singular_pairs,
            singular_pairs=singular_pairs,
            skipped_pairs=skip_singular ? adjacent_pairs : 0,
            image_singular_pairs=image_singular_pairs,
            on_gpu=true,
            gpu_backend=:rocm,
            host_staged_assembly=false,
            regular_kernel_threads=kernel_groupsize,
            regular_kernel_qpair_count=length(rule.points)^2,
            regular_kernel_total_pairs=(image_count + 1) * total_pairs,
            regular_kernel_color_count=color_count,
            regular_kernel_launches=regular_kernel_mode == :pair_owned ?
                2 * (image_count + 1) * color_count^2 :
                2 * (image_count + 1),
            regular_kernel_operator_mode=regular_kernel_mode == :pair_owned ? :partial_fused : nothing,
            regular_kernel_mode=regular_kernel_mode == :pair_owned ?
                "rocm_native_colored_pair_owned" : "rocm_native_entry_owned",
            regular_assembly_mode=regular_kernel_mode == :pair_owned ?
                :rocm_native_colored_pair_owned : :rocm_native_entry_owned,
        ),
    )
end

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
    assembly_mode=nothing,
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    _require_rocm!()
    return_device || error("ROCm host-staged assembly requires return_device=true.")
    accelerator_quadrature || error("ROCm host-staged assembly requires accelerator_quadrature=true.")
    normalized_mode = normalized_symmetry_mode(symmetry_mode)
    normalized_assembly_mode = _normalized_rocm_assembly_mode(assembly_mode)
    if normalized_assembly_mode == :native
        return _assemble_regular_galerkin_operators_rocm_native(
            mesh,
            p1_space,
            dp0_space,
            k,
            rule;
            skip_singular=skip_singular,
            singular_order=singular_order,
            element_indices=element_indices,
            cache=cache,
            timing=timing,
            singular_cache=singular_cache,
            rocm_singular_cache=rocm_singular_cache,
            symmetry_mode=symmetry_mode,
        )
    end
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
            symmetry_mode=normalized_mode,
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
