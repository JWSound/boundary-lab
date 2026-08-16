"""
Regular Galerkin BEM assembly owned by the condensed coupled solver.

This is a deliberate fork of `assemble_regular_galerkin_operators_cpu` in
`BeatEngineCpuAssembly.jl`, taken so that assembly can be optimised for this solver without
touching a file that every other CPU path — exterior, monolithic coupled, and the standalone CLI —
also runs through.

**This file must stay behaviourally identical to its origin.** `coupled_condensed_tests.jl` pins
that with a bitwise equivalence test across every symmetry mode; any optimisation added here has to
keep that test passing.

Only the driver is forked. The element data, quadrature data, colouring, pair accumulators,
singular corrections and the cache type itself are all reused from `BeatEngineCore` by qualified
reference: this keeps the duplicated surface to the loop structure that actually needs to change,
and it means `prepare_coupled_cache` — which builds a `BeatCpuAssemblyCache` — stays fully
compatible, with no parallel cache type to keep in sync.

The qualified `BeatEngineCore._beat_cpu_*` references are private names, so a rename upstream
breaks this file; the equivalence test turns that into a loud failure rather than a silent one.
"""

"""
    _condensed_accumulate_pair_blocks!(single, adjoint, double, hyper, test_data, trial_data, test_quad, trial_quad, k)

Add one test/trial pair's contribution into caller-owned stack blocks, **without scattering**.

This is the inner quadrature loop of `BeatEngineCore._beat_cpu_accumulate_regular_pair_signed!`
with the scatter removed, so several contributions to the same pair can be summed before touching
the operators once. `normal_product` and the curl products are recomputed per contribution because
reflection changes normals and curls, even though it preserves areas and DOFs.
"""
@inline function _condensed_accumulate_pair_blocks!(
    single_block,
    adjoint_block,
    double_block,
    hyper_block,
    test_data::BeatEngineCore.BeatCpuElementData{T},
    trial_data::BeatEngineCore.BeatCpuElementData{T},
    test_quad::BeatEngineCore.BeatCpuRegularQuadratureData{T},
    trial_quad::BeatEngineCore.BeatCpuRegularQuadratureData{T},
    k::T,
) where {T<:AbstractFloat}
    normal_product = dot(test_data.normal, trial_data.normal)
    jac_scale = T(4.0) * test_data.area * trial_data.area
    curl_products = MMatrix{3,3,T,9}(undef)
    for local_row in 1:3
        for local_col in 1:3
            curl_products[local_row, local_col] = dot(test_data.curls[local_row], trial_data.curls[local_col])
        end
    end
    k2 = k * k

    @inbounds for test_q in eachindex(test_quad.weights)
        test_basis = test_quad.basis[test_q]
        x = test_quad.points[test_q]
        test_weight = test_quad.weights[test_q]

        for trial_q in eachindex(trial_quad.weights)
            trial_basis = trial_quad.basis[trial_q]
            y = trial_quad.points[trial_q]
            r_vec = y - x
            radius = norm(r_vec)
            radius == zero(T) && continue

            inv_radius = inv(radius)
            green = BeatEngineCore._beat_cpu_green(radius, inv_radius, k)
            grad_scale = green * Complex{T}(-inv_radius, k)
            trial_dot = dot(r_vec, trial_data.normal) * inv_radius
            test_dot = -dot(r_vec, test_data.normal) * inv_radius
            double_value = grad_scale * trial_dot
            adjoint_value = grad_scale * test_dot
            weight = test_weight * trial_quad.weights[trial_q] * jac_scale
            weighted_green = green * weight
            weighted_double = double_value * weight
            weighted_adjoint = adjoint_value * weight

            for local_row in 1:3
                test_value = test_basis[local_row]
                single_block[local_row] += test_value * weighted_green
                adjoint_block[local_row] += test_value * weighted_adjoint

                for local_col in 1:3
                    trial_value = trial_basis[local_col]
                    basis_product = test_value * trial_value
                    double_block[local_row, local_col] += basis_product * weighted_double
                    hyper_block[local_row, local_col] += (
                        curl_products[local_row, local_col] -
                        k2 * basis_product * normal_product
                    ) * weighted_green
                end
            end
        end
    end
    return nothing
end

"""
    _condensed_accumulate_fused_test!(...)

Sweep one test element's row of the pair space **once**, summing the base contribution and every
symmetry image into a single stack block per pair before scattering.

This is the whole reason the assembly is forked. The shared implementation walks the entire pair
space once per image transform, and each pass ends in a 24-entry read-modify-write into four large
column-strided operators. Those scatters land on the *same* rows and columns every pass, because
reflection preserves `p1_dofs` and `dp0_dof` — so with `xy` symmetry the operators are traversed
four times to deposit sums that could have been formed first and deposited once. The scatter is
latency-bound on a working set far larger than cache, which is why it dominates rather than the
quadrature: measured cost fits `constant + small * quadrature_points`, with the constant carrying
most of the time.

**This changes floating-point summation order and is therefore not bitwise identical to the shared
path when images exist.** In the shared path each pair's contributions reach the operators as four
separate additions interleaved across whole sweeps; here they arrive as one presummed value. The
results agree to round-off, and with `symmetry = :off` there are no images so the two remain
bitwise identical. `coupled_condensed_tests.jl` pins both halves of that statement.
"""
function _condensed_accumulate_fused_test!(
    single_layer,
    double_layer,
    adjoint_double_layer,
    hypersingular,
    elements,
    image_element_sets,
    test_index::Int,
    trial_indices,
    k::T,
    regular_quadrature,
    image_quadrature_sets,
) where {T<:AbstractFloat}
    test_data = elements[test_index]
    test_quad = regular_quadrature[test_index]
    single_block = MVector{3,Complex{T}}(undef)
    adjoint_block = MVector{3,Complex{T}}(undef)
    double_block = MMatrix{3,3,Complex{T},9}(undef)
    hyper_block = MMatrix{3,3,Complex{T},9}(undef)

    @inbounds for trial_index in trial_indices
        trial_data = elements[trial_index]
        # The base contribution skips self/adjacent pairs, which the singular correction handles
        # instead. Image contributions never skip: a reflected element is never adjacent to its
        # own source in the sense that matters here.
        include_base = !BeatEngineCore.elements_are_adjacent(test_data.face, trial_data.face)
        isempty(image_element_sets) && !include_base && continue

        fill!(single_block, zero(Complex{T}))
        fill!(adjoint_block, zero(Complex{T}))
        fill!(double_block, zero(Complex{T}))
        fill!(hyper_block, zero(Complex{T}))

        if include_base
            _condensed_accumulate_pair_blocks!(
                single_block, adjoint_block, double_block, hyper_block,
                test_data, trial_data, test_quad, regular_quadrature[trial_index], k,
            )
        end
        for image_index in eachindex(image_element_sets)
            _condensed_accumulate_pair_blocks!(
                single_block, adjoint_block, double_block, hyper_block,
                test_data, image_element_sets[image_index][trial_index],
                test_quad, image_quadrature_sets[image_index][trial_index], k,
            )
        end

        # One scatter for the base and every image together. Reflection preserves the DOFs, so the
        # base element's indices are the correct destination for all of them.
        for local_row in 1:3
            row = test_data.p1_dofs[local_row]
            single_layer[row, trial_data.dp0_dof] += single_block[local_row]
            adjoint_double_layer[row, trial_data.dp0_dof] += adjoint_block[local_row]
            for local_col in 1:3
                col = trial_data.p1_dofs[local_col]
                double_layer[row, col] += double_block[local_row, local_col]
                hypersingular[row, col] += hyper_block[local_row, local_col]
            end
        end
    end
    return nothing
end

"""
    assemble_condensed_regular_operators(mesh, p1_space, dp0_space, k, rule; ...)

Assemble the four Burton-Miller boundary operators for the condensed solver.

Mirrors `assemble_regular_galerkin_operators_cpu` argument for argument and returns the same
NamedTuple, so the call site is a drop-in swap.
"""
function assemble_condensed_regular_operators(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    rule::TriangleRule{T};
    skip_singular::Bool=true,
    singular_order::Int=2,
    element_indices=eachindex(mesh.faces),
    threaded::Bool=true,
    timing=nothing,
    singular_cache=nothing,
    cpu_cache=nothing,
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    symmetry_mode = BeatEngineCore.normalized_symmetry_mode(symmetry_mode)

    if cpu_cache !== nothing
        cpu_cache.symmetry_mode == symmetry_mode || error("CPU assembly cache symmetry mode does not match the requested mode.")
        cpu_cache.singular_order == singular_order || error("CPU assembly cache singular order does not match the requested order.")
        # Pointer identity, not value equality: `TriangleRule` defines no `==`. Strict on purpose.
        cpu_cache.rule == rule || error("CPU assembly cache quadrature rule does not match the requested rule.")
    end

    indices = cpu_cache === nothing ? collect(element_indices) : cpu_cache.indices
    p1_count = p1_space.global_dof_count
    dp0_count = dp0_space.global_dof_count
    single_layer = zeros(Complex{T}, p1_count, dp0_count)
    double_layer = zeros(Complex{T}, p1_count, p1_count)
    adjoint_double_layer = zeros(Complex{T}, p1_count, dp0_count)
    hypersingular = zeros(Complex{T}, p1_count, p1_count)
    elements = cpu_cache === nothing ?
               BeatEngineCore._beat_cpu_element_data(mesh, p1_space, dp0_space) :
               cpu_cache.elements
    regular_quadrature = cpu_cache === nothing ?
                         BeatEngineCore._beat_cpu_regular_quadrature_data(mesh, rule) :
                         cpu_cache.regular_quadrature
    adjacent_pairs = cpu_cache === nothing ?
                     BeatEngineCore.count_adjacent_pairs(mesh, indices) :
                     cpu_cache.adjacent_pairs
    regular_pairs = length(indices) * length(indices) - adjacent_pairs
    threaded_enabled = cpu_cache === nothing ? threaded && Threads.nthreads() > 1 : cpu_cache.threaded_enabled
    color_groups = Vector{Vector{Int}}()
    color_build_elapsed = @elapsed begin
        color_groups = if cpu_cache === nothing
            threaded_enabled ? BeatEngineCore._beat_cpu_element_color_groups(mesh, indices) : [indices]
        else
            cpu_cache.color_groups
        end
    end
    timing !== nothing && (timing["regular_operator_cpu_color_build"] = color_build_elapsed)
    image_transforms = cpu_cache === nothing ?
                       collect(BeatEngineCore.symmetry_image_transforms(symmetry_mode)) :
                       cpu_cache.image_transforms

    # Materialise every image's reflected data up front so the fused sweep can visit all of them
    # per pair. With a cache these are already built and this is just aliasing.
    image_element_sets = [
        cpu_cache === nothing ?
        BeatEngineCore._beat_cpu_reflect_element_data(elements, transform) :
        cpu_cache.image_elements[transform_index]
        for (transform_index, transform) in enumerate(image_transforms)
    ]
    image_quadrature_sets = [
        cpu_cache === nothing ?
        BeatEngineCore._beat_cpu_reflect_regular_quadrature_data(regular_quadrature, transform) :
        cpu_cache.image_quadrature[transform_index]
        for (transform_index, transform) in enumerate(image_transforms)
    ]

    regular_elapsed = @elapsed begin
        # One pass over the pair space instead of one per image. See
        # `_condensed_accumulate_fused_test!` for why this is the fork's whole purpose.
        if threaded_enabled
            for group in color_groups
                Threads.@threads for group_index in eachindex(group)
                    _condensed_accumulate_fused_test!(
                        single_layer,
                        double_layer,
                        adjoint_double_layer,
                        hypersingular,
                        elements,
                        image_element_sets,
                        group[group_index],
                        indices,
                        k,
                        regular_quadrature,
                        image_quadrature_sets,
                    )
                end
            end
        else
            for test_index in indices
                _condensed_accumulate_fused_test!(
                    single_layer,
                    double_layer,
                    adjoint_double_layer,
                    hypersingular,
                    elements,
                    image_element_sets,
                    test_index,
                    indices,
                    k,
                    regular_quadrature,
                    image_quadrature_sets,
                )
            end
        end
    end
    timing !== nothing && (timing["regular_operator_cpu_scatter"] = regular_elapsed)

    singular_pairs = 0
    skipped_pairs = adjacent_pairs

    if !skip_singular
        cache = singular_cache === nothing ?
                build_singular_correction_cache(mesh, singular_order, indices) :
                singular_cache
        singular_elapsed = @elapsed begin
            if threaded_enabled
                for group in color_groups
                    Threads.@threads for group_index in eachindex(group)
                        test_index = group[group_index]
                        BeatEngineCore._beat_cpu_accumulate_singular_test!(
                            single_layer,
                            double_layer,
                            adjoint_double_layer,
                            hypersingular,
                            elements,
                            cache.pairs_by_test[test_index],
                            cache.rules,
                            k,
                        )
                    end
                end
            else
                for test_index in indices
                    BeatEngineCore._beat_cpu_accumulate_singular_test!(
                        single_layer,
                        double_layer,
                        adjoint_double_layer,
                        hypersingular,
                        elements,
                        cache.pairs_by_test[test_index],
                        cache.rules,
                        k,
                    )
                end
            end
        end
        timing !== nothing && (timing["singular_corrections_cpu_scatter"] = singular_elapsed)
        singular_pairs = cache.pair_count
        skipped_pairs = 0
    else
        timing !== nothing && (timing["singular_corrections_cpu_scatter"] = 0.0)
    end

    image_singular_pairs = 0
    image_singular_elapsed = @elapsed begin
        if !skip_singular
            for (transform_index, transform) in enumerate(image_transforms)
                image_cache = cpu_cache === nothing ?
                    BeatEngineCore._beat_cpu_image_singular_cache(mesh, singular_order, indices, transform) :
                    cpu_cache.image_singular_caches[transform_index]
                image_singular_pairs += image_cache.pair_count
                image_cache.pair_count == 0 && continue
                image_elements = cpu_cache === nothing ?
                    BeatEngineCore._beat_cpu_reflect_element_data(elements, transform) :
                    cpu_cache.image_elements[transform_index]
                image_quadrature = cpu_cache === nothing ?
                    BeatEngineCore._beat_cpu_reflect_regular_quadrature_data(regular_quadrature, transform) :
                    cpu_cache.image_quadrature[transform_index]
                if threaded_enabled
                    for group in color_groups
                        Threads.@threads for group_index in eachindex(group)
                            test_index = group[group_index]
                            BeatEngineCore._beat_cpu_accumulate_image_singular_delta_test!(
                                single_layer,
                                double_layer,
                                adjoint_double_layer,
                                hypersingular,
                                elements,
                                image_elements,
                                image_cache.pairs_by_test[test_index],
                                image_cache.rules,
                                k,
                                regular_quadrature,
                                image_quadrature,
                            )
                        end
                    end
                else
                    for test_index in indices
                        BeatEngineCore._beat_cpu_accumulate_image_singular_delta_test!(
                            single_layer,
                            double_layer,
                            adjoint_double_layer,
                            hypersingular,
                            elements,
                            image_elements,
                            image_cache.pairs_by_test[test_index],
                            image_cache.rules,
                            k,
                            regular_quadrature,
                            image_quadrature,
                        )
                    end
                end
            end
        end
    end
    timing !== nothing && (timing["image_singular_corrections_cpu_scatter"] = image_singular_elapsed)

    BeatEngineCore._beat_cpu_apply_operator_p1_row_weights!(
        (
            single_layer=single_layer,
            double_layer=double_layer,
            adjoint_double_layer=adjoint_double_layer,
            hypersingular=hypersingular,
        ),
        mesh,
        symmetry_mode,
    )

    return (
        single_layer=single_layer,
        double_layer=double_layer,
        adjoint_double_layer=adjoint_double_layer,
        hypersingular=hypersingular,
        regular_pairs=regular_pairs + length(image_transforms) * length(indices) * length(indices),
        singular_pairs=singular_pairs,
        skipped_pairs=skipped_pairs,
        image_singular_pairs=image_singular_pairs,
        cpu_color_count=length(color_groups),
        on_gpu=false,
        regular_kernel_mode=threaded_enabled ? "cpu_colored_threads" : "cpu_serial",
        regular_assembly_mode=threaded_enabled ? :cpu_colored_threads : :cpu_serial,
    )
end
