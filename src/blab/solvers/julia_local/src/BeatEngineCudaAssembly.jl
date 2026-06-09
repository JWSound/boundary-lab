function assemble_regular_galerkin_operators_cuda_regular(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    rule::TriangleRule{T};
    skip_singular::Bool=true,
    singular_order::Int=2,
    element_indices=eachindex(mesh.faces),
    cache=nothing,
    return_gpu::Bool=true,
    parallel_quadrature::Bool=true,
    timing=nothing,
    singular_cache=nothing,
    cuda_singular_cache=nothing,
    profile_regular_kernel::Bool=false,
    regular_probe_pair_limit::Int=1_000_000,
    regular_kernel_threads_override::Union{Nothing,Int}=nothing,
    regular_assembly_mode::Symbol=:split_atomic_balanced_multipair,
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    CUDA.functional() || error("CUDA regular-pair assembly requested, but CUDA.functional() is false.")
    if regular_kernel_threads_override !== nothing
        regular_kernel_threads_override > 0 || error("regular_kernel_threads_override must be positive.")
        ispow2(regular_kernel_threads_override) || error("regular_kernel_threads_override must be a power of two for the current reduction kernels.")
    end
    regular_assembly_mode in (:split_atomic_balanced, :split_atomic_balanced_multipair, :split_atomic_slp_hyp_separate) || error("Unsupported regular CUDA assembly mode: $(regular_assembly_mode).")
    parallel_quadrature || error("Balanced CUDA regular assembly requires parallel_quadrature=true.")
    return_gpu || error("BEAT Engine is CUDA-only; CPU operator materialization has been removed.")

    indices = cache === nothing ? collect(element_indices) : cache.element_indices
    face_count = cache === nothing ? length(mesh.faces) : cache.face_count
    p1_dof_count = p1_space.global_dof_count
    dp0_dof_count = dp0_space.global_dof_count
    rule_count = cache === nothing ? length(rule.points) : cache.rule_count
    total_pairs = length(indices) * length(indices)
    symmetry_images = symmetry_image_transforms(symmetry_mode)
    kernel_mode = string(regular_assembly_mode)
    kernel_threads = regular_kernel_threads_override === nothing ? _regular_quadrature_threads(rule_count) : regular_kernel_threads_override
    regular_pairs_per_block = regular_assembly_mode == :split_atomic_balanced_multipair ? 8 : 1
    kernel_blocks = cld(total_pairs, regular_pairs_per_block)
    kernel_shmem = kernel_threads * regular_pairs_per_block * 24 * sizeof(T)
    probe_pair_count = regular_probe_pair_limit <= 0 ? total_pairs : min(total_pairs, regular_probe_pair_limit)

    if cache === nothing
        face_vertices, normals, areas, faces, curls = _cuda_geometry_arrays(mesh)
        rule_points, rule_weights = _cuda_rule_arrays(rule)
        color_indices, color_offsets, color_count = _regular_face_color_arrays(mesh, indices)

        _cuda_timed_stage!(timing, "regular_operator_geometry_transfer") do
            d_face_vertices = CuArray(face_vertices)
            d_normals = CuArray(normals)
            d_areas = CuArray(areas)
            d_faces = CuArray(faces)
            d_curls = CuArray(curls)
            d_rule_points = CuArray(rule_points)
            d_rule_weights = CuArray(rule_weights)
            d_test_indices = CuArray(Int32.(indices))
            d_trial_indices = CuArray(Int32.(indices))
            d_color_indices = CuArray(color_indices)
            d_color_offsets = CuArray(color_offsets)
            CUDA.synchronize()
            nothing
        end
    else
        timing !== nothing && (timing["regular_operator_geometry_transfer"] = 0.0)
        d_face_vertices = cache.face_vertices
        d_normals = cache.normals
        d_areas = cache.areas
        d_faces = cache.faces
        d_curls = cache.curls
        d_rule_points = cache.rule_points
        d_rule_weights = cache.rule_weights
        d_test_indices = cache.test_indices
        d_trial_indices = cache.trial_indices
        d_color_indices = cache.color_indices
        d_color_offsets = cache.color_offsets
        color_count = cache.color_count
    end

    slp_re = slp_im = adj_re = adj_im = dlp_re = dlp_im = hyp_re = hyp_im = nothing
    _cuda_timed_stage!(timing, "regular_operator_gpu_alloc") do
        slp_re = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        slp_im = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        adj_re = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        adj_im = CUDA.zeros(T, p1_dof_count, dp0_dof_count)
        dlp_re = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        dlp_im = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        hyp_re = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        hyp_im = CUDA.zeros(T, p1_dof_count, p1_dof_count)
        CUDA.synchronize()
        nothing
    end

    if profile_regular_kernel
        _profile_regular_thread_sweep!(
            timing,
            T,
            p1_dof_count,
            dp0_dof_count,
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
            face_count,
            rule_count,
            total_pairs,
        )
        _profile_regular_quadrature_probes!(
            timing,
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
            face_count,
            rule_count,
            probe_pair_count,
            kernel_threads,
        )
        _profile_regular_slp_adjoint_colored!(
            timing,
            T,
            p1_dof_count,
            dp0_dof_count,
            d_face_vertices,
            d_normals,
            d_areas,
            d_faces,
            d_color_indices,
            d_color_offsets,
            d_trial_indices,
            d_rule_points,
            d_rule_weights,
            k,
            face_count,
            rule_count,
            color_count,
            kernel_threads,
            regular_probe_pair_limit,
        )
        _profile_regular_split_atomic!(
            timing,
            T,
            p1_dof_count,
            dp0_dof_count,
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
            face_count,
            rule_count,
            probe_pair_count,
            kernel_threads,
        )
    else
        timing !== nothing && begin
            timing["regular_operator_probe_green_kernel"] = 0.0
            timing["regular_operator_probe_all_terms_kernel"] = 0.0
        end
        _zero_extended_regular_profile_timings!(timing)
    end

    _cuda_timed_stage!(timing, "regular_operator_kernel") do
        launch_regular_kernel! =
            regular_assembly_mode == :split_atomic_balanced_multipair ? _launch_regular_split_balanced_multipair_atomic_kernel! :
            regular_assembly_mode == :split_atomic_slp_hyp_separate ? _launch_regular_split_slp_hyp_separate_atomic_kernel! :
            _launch_regular_split_balanced_atomic_kernel!
        launch_regular_kernel!(
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
            face_count,
            rule_count,
            total_pairs,
            kernel_threads,
        )
        for transform in symmetry_images
            _launch_regular_symmetry_image_kernel!(
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
                128,
                transform,
            )
        end
        CUDA.synchronize()
        nothing
    end

    correction_cache = singular_cache
    adjacent_pairs = _cuda_timed_stage!(timing, "regular_operator_count_transfer") do
        if correction_cache === nothing
            count_adjacent_pairs(mesh, indices)
        else
            correction_cache.pair_count
        end
    end
    regular_pairs = total_pairs - adjacent_pairs + length(symmetry_images) * total_pairs

    single_layer = double_layer = adjoint_double_layer = hypersingular = nothing
    _cuda_timed_stage!(timing, "regular_operator_complex_materialize") do
        single_layer = _complex_gpu_matrix(slp_re, slp_im)
        double_layer = _complex_gpu_matrix(dlp_re, dlp_im)
        adjoint_double_layer = _complex_gpu_matrix(adj_re, adj_im)
        hypersingular = _complex_gpu_matrix(hyp_re, hyp_im)
        CUDA.synchronize()
        nothing
    end
    timing !== nothing && (timing["regular_operator_cpu_transfer"] = 0.0)

    if skip_singular
        singular_pairs = 0
        skipped_pairs = adjacent_pairs
    else
        correction_cache === nothing && (correction_cache = build_singular_correction_cache(mesh, singular_order, indices))
        singular_pairs = add_singular_corrections_cuda_compact!(
            (
                single_layer=single_layer,
                double_layer=double_layer,
                adjoint_double_layer=adjoint_double_layer,
                hypersingular=hypersingular,
            ),
            mesh,
            p1_space,
            dp0_space,
            k,
            singular_order,
            indices,
            correction_cache,
            cuda_singular_cache=cuda_singular_cache,
            cuda_regular_cache=cache,
            timing=timing,
        )
        skipped_pairs = 0
    end

    image_singular_pairs = _cuda_timed_stage!(timing, "regular_operator_image_singular_corrections") do
        if skip_singular || isempty(symmetry_images)
            0
        else
            add_image_singular_corrections_cuda_compact!(
                (
                    single_layer=single_layer,
                    double_layer=double_layer,
                    adjoint_double_layer=adjoint_double_layer,
                    hypersingular=hypersingular,
                    on_gpu=true,
                ),
                mesh,
                p1_space,
                dp0_space,
                k,
                rule,
                singular_order,
                indices,
                symmetry_mode;
                cuda_regular_cache=cache,
                timing=timing,
            )
        end
    end

    _cuda_timed_stage!(timing, "regular_operator_symmetry_row_weights") do
        _apply_operator_p1_row_weights!(
            (
                single_layer=single_layer,
                double_layer=double_layer,
                adjoint_double_layer=adjoint_double_layer,
                hypersingular=hypersingular,
                on_gpu=true,
            ),
            mesh,
            symmetry_mode,
        )
    end

    return (
        single_layer=single_layer,
        double_layer=double_layer,
        adjoint_double_layer=adjoint_double_layer,
        hypersingular=hypersingular,
        regular_pairs=regular_pairs,
        singular_pairs=singular_pairs,
        skipped_pairs=skipped_pairs,
        image_singular_pairs=image_singular_pairs,
        on_gpu=true,
        regular_kernel_threads=kernel_threads,
        regular_kernel_blocks=kernel_blocks,
        regular_kernel_shared_memory_bytes=kernel_shmem,
        regular_kernel_qpair_count=rule_count * rule_count,
        regular_kernel_total_pairs=total_pairs,
        regular_probe_pair_count=probe_pair_count,
        regular_kernel_mode=kernel_mode,
        regular_assembly_mode=regular_assembly_mode,
        regular_color_count=color_count,
    )
end
