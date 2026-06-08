function _beat_cpu_green(radius::T, k::T) where {T<:AbstractFloat}
    phase = k * radius
    scale = inv(T(4.0) * T(pi) * radius)
    return Complex{T}(cos(phase) * scale, sin(phase) * scale)
end

struct BeatCpuElementData{T<:AbstractFloat}
    face::NTuple{3,Int}
    vertices::NTuple{3,SVector{3,T}}
    normal::SVector{3,T}
    curls::NTuple{3,SVector{3,T}}
    p1_dofs::NTuple{3,Int}
    dp0_dof::Int
    area::T
end

struct BeatCpuRegularQuadratureData{T<:AbstractFloat}
    points::Vector{SVector{3,T}}
    basis::Vector{SVector{3,T}}
    weights::Vector{T}
end

struct BeatCpuImageSingularCache{T<:AbstractFloat}
    pairs_by_test::Vector{Vector{Any}}
    pair_count::Int
end

function _beat_cpu_element_data(mesh::BoundaryMesh{T}, p1_space::P1Space, dp0_space::DP0Space) where {T<:AbstractFloat}
    return [
        BeatCpuElementData(
            mesh.faces[element_index],
            mesh.face_vertices[element_index],
            mesh.normals[element_index],
            surface_curls(mesh.face_vertices[element_index], mesh.normals[element_index]),
            p1_space.local_to_global[element_index],
            dp0_space.local_to_global[element_index],
            mesh.areas[element_index],
        )
        for element_index in eachindex(mesh.faces)
    ]
end

function _beat_cpu_reflect_element_data(elements::Vector{BeatCpuElementData{T}}, transform::SymmetryTransform) where {T<:AbstractFloat}
    return [
        BeatCpuElementData(
            element.face,
            reflect_vertices(transform, element.vertices),
            reflect_normal(transform, element.normal),
            (
                reflect_curl(transform, element.curls[1]),
                reflect_curl(transform, element.curls[2]),
                reflect_curl(transform, element.curls[3]),
            ),
            element.p1_dofs,
            element.dp0_dof,
            element.area,
        )
        for element in elements
    ]
end

function _beat_cpu_regular_quadrature_data(mesh::BoundaryMesh{T}, rule::TriangleRule{T}) where {T<:AbstractFloat}
    return [
        BeatCpuRegularQuadratureData(
            [local_to_global(mesh.face_vertices[element_index], point) for point in rule.points],
            [p1_values(point) for point in rule.points],
            rule.weights,
        )
        for element_index in eachindex(mesh.faces)
    ]
end

function _beat_cpu_reflect_regular_quadrature_data(regular_quadrature::Vector{BeatCpuRegularQuadratureData{T}}, transform::SymmetryTransform) where {T<:AbstractFloat}
    return [
        BeatCpuRegularQuadratureData(
            [reflect_point(transform, point) for point in quad.points],
            quad.basis,
            quad.weights,
        )
        for quad in regular_quadrature
    ]
end

function _beat_cpu_element_color_groups(mesh::BoundaryMesh, indices)
    element_colors = zeros(Int, length(mesh.faces))
    vertex_colors = Dict{Int,Set{Int}}()
    groups = Vector{Int}[]

    for element_index in indices
        used_colors = Set{Int}()
        for vertex in mesh.faces[element_index]
            union!(used_colors, get(vertex_colors, vertex, Set{Int}()))
        end

        color = 1
        while color in used_colors
            color += 1
        end
        element_colors[element_index] = color

        while length(groups) < color
            push!(groups, Int[])
        end
        push!(groups[color], element_index)

        for vertex in mesh.faces[element_index]
            push!(get!(vertex_colors, vertex, Set{Int}()), color)
        end
    end

    return groups
end

function _beat_cpu_tensor_product_rule(rule::TriangleRule{T}) where {T<:AbstractFloat}
    count = length(rule.weights)^2
    test_points = Vector{SVector{2,T}}(undef, count)
    trial_points = Vector{SVector{2,T}}(undef, count)
    weights = Vector{T}(undef, count)

    q = 1
    for test_index in eachindex(rule.weights)
        for trial_index in eachindex(rule.weights)
            test_points[q] = rule.points[test_index]
            trial_points[q] = rule.points[trial_index]
            weights[q] = rule.weights[test_index] * rule.weights[trial_index]
            q += 1
        end
    end

    return test_points, trial_points, weights
end

function _beat_cpu_accumulate_pair!(
    single_layer,
    double_layer,
    adjoint_double_layer,
    hypersingular,
    rows::NTuple{3,Int},
    p1_cols::NTuple{3,Int},
    dp0_col::Int,
    test_vertices::NTuple{3,SVector{3,T}},
    trial_vertices::NTuple{3,SVector{3,T}},
    test_normal::SVector{3,T},
    trial_normal::SVector{3,T},
    test_curls::NTuple{3,SVector{3,T}},
    trial_curls::NTuple{3,SVector{3,T}},
    normal_product::T,
    jac_scale::T,
    k::T,
    test_points,
    trial_points,
    weights,
    scale::Complex{T}=one(Complex{T}),
) where {T<:AbstractFloat}
    curl_products = MMatrix{3,3,T,9}(undef)
    for local_row in 1:3
        for local_col in 1:3
            curl_products[local_row, local_col] = dot(test_curls[local_row], trial_curls[local_col])
        end
    end
    k2 = k * k

    @inbounds for q in eachindex(weights)
        test_basis = p1_values(test_points[q])
        trial_basis = p1_values(trial_points[q])
        x = local_to_global(test_vertices, test_points[q])
        y = local_to_global(trial_vertices, trial_points[q])
        r_vec = y - x
        radius = norm(r_vec)
        radius == zero(T) && continue

        green = _beat_cpu_green(radius, k)
        grad_scale = green * Complex{T}(-inv(radius), k)
        trial_dot = dot(r_vec, trial_normal) / radius
        test_dot = -dot(r_vec, test_normal) / radius
        double_value = grad_scale * trial_dot
        adjoint_value = grad_scale * test_dot
        weight = weights[q] * jac_scale

        for local_row in 1:3
            row = rows[local_row]
            test_value = test_basis[local_row]
            single_layer[row, dp0_col] += scale * test_value * green * weight
            adjoint_double_layer[row, dp0_col] += scale * test_value * adjoint_value * weight

            for local_col in 1:3
                col = p1_cols[local_col]
                trial_value = trial_basis[local_col]
                basis_product = test_value * trial_value
                double_layer[row, col] += scale * basis_product * double_value * weight
                hypersingular[row, col] += scale * (
                    curl_products[local_row, local_col] -
                    k2 * basis_product * normal_product
                ) * green * weight
            end
        end
    end

    return nothing
end

function _beat_cpu_accumulate_regular_pair!(
    single_layer,
    double_layer,
    adjoint_double_layer,
    hypersingular,
    test_data::BeatCpuElementData{T},
    trial_data::BeatCpuElementData{T},
    test_quad::BeatCpuRegularQuadratureData{T},
    trial_quad::BeatCpuRegularQuadratureData{T},
    normal_product::T,
    jac_scale::T,
    k::T,
    scale::Complex{T}=one(Complex{T}),
) where {T<:AbstractFloat}
    single_block = MVector{3,Complex{T}}(undef)
    adjoint_block = MVector{3,Complex{T}}(undef)
    double_block = MMatrix{3,3,Complex{T},9}(undef)
    hyper_block = MMatrix{3,3,Complex{T},9}(undef)
    fill!(single_block, zero(Complex{T}))
    fill!(adjoint_block, zero(Complex{T}))
    fill!(double_block, zero(Complex{T}))
    fill!(hyper_block, zero(Complex{T}))
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

            green = _beat_cpu_green(radius, k)
            inv_radius = inv(radius)
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
                single_block[local_row] += scale * test_value * weighted_green
                adjoint_block[local_row] += scale * test_value * weighted_adjoint

                for local_col in 1:3
                    trial_value = trial_basis[local_col]
                    basis_product = test_value * trial_value
                    double_block[local_row, local_col] += scale * basis_product * weighted_double
                    hyper_block[local_row, local_col] += scale * (
                        curl_products[local_row, local_col] -
                        k2 * basis_product * normal_product
                    ) * weighted_green
                end
            end
        end
    end

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

    return nothing
end

function _beat_cpu_accumulate_regular_test!(
    single_layer,
    double_layer,
    adjoint_double_layer,
    hypersingular,
    elements,
    test_index::Int,
    trial_indices,
    k::T,
    regular_quadrature,
) where {T<:AbstractFloat}
    test_data = elements[test_index]
    test_quad = regular_quadrature[test_index]

    for trial_index in trial_indices
        trial_data = elements[trial_index]
        elements_are_adjacent(test_data.face, trial_data.face) && continue

        _beat_cpu_accumulate_regular_pair!(
            single_layer,
            double_layer,
            adjoint_double_layer,
            hypersingular,
            test_data,
            trial_data,
            test_quad,
            regular_quadrature[trial_index],
            dot(test_data.normal, trial_data.normal),
            T(4.0) * test_data.area * trial_data.area,
            k,
        )
    end

    return nothing
end

function _beat_cpu_accumulate_regular_image_test!(
    single_layer,
    double_layer,
    adjoint_double_layer,
    hypersingular,
    elements,
    image_elements,
    test_index::Int,
    trial_indices,
    k::T,
    regular_quadrature,
    image_quadrature,
) where {T<:AbstractFloat}
    test_data = elements[test_index]
    test_quad = regular_quadrature[test_index]

    for trial_index in trial_indices
        trial_data = image_elements[trial_index]
        _beat_cpu_accumulate_regular_pair!(
            single_layer,
            double_layer,
            adjoint_double_layer,
            hypersingular,
            test_data,
            trial_data,
            test_quad,
            image_quadrature[trial_index],
            dot(test_data.normal, trial_data.normal),
            T(4.0) * test_data.area * trial_data.area,
            k,
        )
    end

    return nothing
end

function _beat_cpu_accumulate_singular_test!(
    single_layer,
    double_layer,
    adjoint_double_layer,
    hypersingular,
    elements,
    pairs,
    rules,
    k::T,
) where {T<:AbstractFloat}
    for pair in pairs
        test_data = elements[pair.test_index]
        trial_data = elements[pair.trial_index]
        duffy = rules[pair.rule_index]

        _beat_cpu_accumulate_pair!(
            single_layer,
            double_layer,
            adjoint_double_layer,
            hypersingular,
            test_data.p1_dofs,
            trial_data.p1_dofs,
            trial_data.dp0_dof,
            test_data.vertices,
            trial_data.vertices,
            test_data.normal,
            trial_data.normal,
            test_data.curls,
            trial_data.curls,
            pair.normal_product,
            pair.jac_scale,
            k,
            duffy.test_points,
            duffy.trial_points,
            duffy.weights,
        )
    end

    return nothing
end

function _beat_cpu_image_singular_cache(
    mesh::BoundaryMesh{T},
    singular_order::Int,
    element_indices,
    transform::SymmetryTransform;
    tolerance::T=T(1e-8),
) where {T<:AbstractFloat}
    base_rules = Dict(
        :coincident => duffy_rule(T, singular_order, :coincident),
        :edge_adjacent => duffy_rule(T, singular_order, :edge_adjacent),
        :vertex_adjacent => duffy_rule(T, singular_order, :vertex_adjacent),
    )
    rules = DuffyRule{T}[]
    rule_indices_by_key = Dict{NTuple{5,Int},Int}()
    pairs_by_test = [Any[] for _ in eachindex(mesh.faces)]
    pair_count = 0

    for (test_index, trial_index) in image_singular_candidates(mesh, element_indices, transform; tolerance=tolerance)
        test_vertices = mesh.face_vertices[test_index]
        trial_vertices = reflect_vertices(transform, mesh.face_vertices[trial_index])
        info = geometric_adjacency_info(test_vertices, trial_vertices; tolerance=tolerance)
        info.kind == :regular && continue
        rule_index = rule_for_singular_orientation!(rules, rule_indices_by_key, base_rules, info)
        test_normal = mesh.normals[test_index]
        trial_normal = reflect_normal(transform, mesh.normals[trial_index])
        push!(pairs_by_test[test_index], (
            test_index=test_index,
            trial_index=trial_index,
            rule=rules[rule_index],
            jac_scale=(T(2.0) * mesh.areas[test_index]) * (T(2.0) * mesh.areas[trial_index]),
            normal_product=dot(test_normal, trial_normal),
        ))
        pair_count += 1
    end

    return BeatCpuImageSingularCache{T}(pairs_by_test, pair_count)
end

function _beat_cpu_accumulate_image_singular_delta_test!(
    single_layer,
    double_layer,
    adjoint_double_layer,
    hypersingular,
    elements,
    image_elements,
    pairs,
    k::T,
    regular_quadrature,
    image_quadrature,
) where {T<:AbstractFloat}
    minus_one = Complex{T}(-1, 0)

    for pair in pairs
        test_data = elements[pair.test_index]
        trial_data = image_elements[pair.trial_index]
        _beat_cpu_accumulate_pair!(
            single_layer,
            double_layer,
            adjoint_double_layer,
            hypersingular,
            test_data.p1_dofs,
            trial_data.p1_dofs,
            trial_data.dp0_dof,
            test_data.vertices,
            trial_data.vertices,
            test_data.normal,
            trial_data.normal,
            test_data.curls,
            trial_data.curls,
            pair.normal_product,
            pair.jac_scale,
            k,
            pair.rule.test_points,
            pair.rule.trial_points,
            pair.rule.weights,
        )
        _beat_cpu_accumulate_regular_pair!(
            single_layer,
            double_layer,
            adjoint_double_layer,
            hypersingular,
            test_data,
            trial_data,
            regular_quadrature[pair.test_index],
            image_quadrature[pair.trial_index],
            pair.normal_product,
            T(4.0) * test_data.area * trial_data.area,
            k,
            minus_one,
        )
    end

    return nothing
end

function _beat_cpu_apply_operator_p1_row_weights!(operators, mesh::BoundaryMesh{T}, symmetry_mode::Symbol) where {T<:AbstractFloat}
    weights = p1_symmetry_orbit_weights(mesh, symmetry_mode)
    operators.single_layer .*= reshape(weights, :, 1)
    operators.adjoint_double_layer .*= reshape(weights, :, 1)
    operators.double_layer .*= reshape(weights, :, 1)
    operators.hypersingular .*= reshape(weights, :, 1)
    return nothing
end

function assemble_regular_galerkin_operators_cpu(
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
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    symmetry_mode = normalized_symmetry_mode(symmetry_mode)

    indices = collect(element_indices)
    p1_count = p1_space.global_dof_count
    dp0_count = dp0_space.global_dof_count
    single_layer = zeros(Complex{T}, p1_count, dp0_count)
    double_layer = zeros(Complex{T}, p1_count, p1_count)
    adjoint_double_layer = zeros(Complex{T}, p1_count, dp0_count)
    hypersingular = zeros(Complex{T}, p1_count, p1_count)
    elements = _beat_cpu_element_data(mesh, p1_space, dp0_space)
    regular_quadrature = _beat_cpu_regular_quadrature_data(mesh, rule)
    adjacent_pairs = count_adjacent_pairs(mesh, indices)
    regular_pairs = length(indices) * length(indices) - adjacent_pairs
    threaded_enabled = threaded && Threads.nthreads() > 1
    color_groups = Vector{Vector{Int}}()
    color_build_elapsed = @elapsed begin
        color_groups = threaded_enabled ? _beat_cpu_element_color_groups(mesh, indices) : [indices]
    end
    timing !== nothing && (timing["regular_operator_cpu_color_build"] = color_build_elapsed)
    image_transforms = symmetry_image_transforms(symmetry_mode)

    regular_elapsed = @elapsed begin
        if threaded_enabled
            for group in color_groups
                Threads.@threads for group_index in eachindex(group)
                    _beat_cpu_accumulate_regular_test!(
                        single_layer,
                        double_layer,
                        adjoint_double_layer,
                        hypersingular,
                        elements,
                        group[group_index],
                        indices,
                        k,
                        regular_quadrature,
                    )
                end
            end
        else
            for test_index in indices
                _beat_cpu_accumulate_regular_test!(
                    single_layer,
                    double_layer,
                    adjoint_double_layer,
                    hypersingular,
                    elements,
                    test_index,
                    indices,
                    k,
                    regular_quadrature,
                )
            end
        end

        for transform in image_transforms
            image_elements = _beat_cpu_reflect_element_data(elements, transform)
            image_quadrature = _beat_cpu_reflect_regular_quadrature_data(regular_quadrature, transform)
            if threaded_enabled
                for group in color_groups
                    Threads.@threads for group_index in eachindex(group)
                        _beat_cpu_accumulate_regular_image_test!(
                            single_layer,
                            double_layer,
                            adjoint_double_layer,
                            hypersingular,
                            elements,
                            image_elements,
                            group[group_index],
                            indices,
                            k,
                            regular_quadrature,
                            image_quadrature,
                        )
                    end
                end
            else
                for test_index in indices
                    _beat_cpu_accumulate_regular_image_test!(
                        single_layer,
                        double_layer,
                        adjoint_double_layer,
                        hypersingular,
                        elements,
                        image_elements,
                        test_index,
                        indices,
                        k,
                        regular_quadrature,
                        image_quadrature,
                    )
                end
            end
        end
    end
    timing !== nothing && (timing["regular_operator_cpu_scatter"] = regular_elapsed)

    singular_pairs = 0
    skipped_pairs = adjacent_pairs

    if !skip_singular
        cache = singular_cache === nothing ? build_singular_correction_cache(mesh, singular_order, indices) : singular_cache
        singular_elapsed = @elapsed begin
            if threaded_enabled
                for group in color_groups
                    Threads.@threads for group_index in eachindex(group)
                        test_index = group[group_index]
                        _beat_cpu_accumulate_singular_test!(
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
                    _beat_cpu_accumulate_singular_test!(
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
            for transform in image_transforms
                image_cache = _beat_cpu_image_singular_cache(mesh, singular_order, indices, transform)
                image_singular_pairs += image_cache.pair_count
                image_cache.pair_count == 0 && continue
                image_elements = _beat_cpu_reflect_element_data(elements, transform)
                image_quadrature = _beat_cpu_reflect_regular_quadrature_data(regular_quadrature, transform)
                if threaded_enabled
                    for group in color_groups
                        Threads.@threads for group_index in eachindex(group)
                            test_index = group[group_index]
                            _beat_cpu_accumulate_image_singular_delta_test!(
                                single_layer,
                                double_layer,
                                adjoint_double_layer,
                                hypersingular,
                                elements,
                                image_elements,
                                image_cache.pairs_by_test[test_index],
                                k,
                                regular_quadrature,
                                image_quadrature,
                            )
                        end
                    end
                else
                    for test_index in indices
                        _beat_cpu_accumulate_image_singular_delta_test!(
                            single_layer,
                            double_layer,
                            adjoint_double_layer,
                            hypersingular,
                            elements,
                            image_elements,
                            image_cache.pairs_by_test[test_index],
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

    _beat_cpu_apply_operator_p1_row_weights!(
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
