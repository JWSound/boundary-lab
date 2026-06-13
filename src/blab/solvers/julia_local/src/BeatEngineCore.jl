module BeatEngineCore

using Base.Threads, LinearAlgebra, StaticArrays

const CUDA_MODULE = try
    @eval import CUDA
    CUDA
catch
    nothing
end

export BoundaryMesh,
    DP0Space,
    P1Space,
    SymmetryTransform,
    TriangleRule,
    assemble_l2_identity_matrix,
    build_cuda_regular_assembly_cache,
    build_cuda_field_evaluation_cache,
    build_field_evaluation_cache,
    build_singular_correction_cache,
    assemble_regular_galerkin_operators_cpu,
    assemble_regular_galerkin_operators_cuda_regular,
    assemble_regular_galerkin_operators,
    adjacency_info,
    build_dp0_space,
    build_p1_space,
    duffy_rule,
    elements_are_adjacent,
    evaluate_galerkin_field_cpu,
    evaluate_galerkin_field_cuda,
    fibonacci_sphere,
    helmholtz_adjoint_double_layer_kernel,
    helmholtz_double_layer_kernel,
    helmholtz_single_layer_kernel,
    load_gmsh22_with_tags,
    mesh_for_frequency,
    release_operator_storage!,
    surface_curls,
    scatter_element_block!,
    build_burton_miller_neumann_cpu_system,
    solve_burton_miller_neumann_cpu_system,
    solve_burton_miller_neumann_cpu,
    solve_burton_miller_neumann,
    reflect_curl,
    reflect_normal,
    reflect_point,
    reflect_vertices,
    p1_symmetry_orbit_weights,
    symmetry_image_transforms,
    symmetry_reduction_factor,
    symmetry_transforms,
    triangle_rule,
    validate_symmetry_fundamental_domain!

function cuda_module()
    CUDA_MODULE === nothing && error("CUDA solve requested, but CUDA.jl could not be loaded.")
    return CUDA_MODULE
end

struct BoundaryMesh{T<:AbstractFloat}
    vertices::Vector{SVector{3,T}}
    faces::Vector{NTuple{3,Int}}
    physical_tags::Vector{Int}
    centroids::Vector{SVector{3,T}}
    normals::Vector{SVector{3,T}}
    areas::Vector{T}
    face_vertices::Vector{NTuple{3,SVector{3,T}}}
end

struct SymmetryTransform
    label::Symbol
    signs::SVector{3,Int}
    determinant::Int
end

struct P1Space
    local_to_global::Vector{NTuple{3,Int}}
    global_dof_count::Int
end

struct DP0Space
    local_to_global::Vector{Int}
    global_dof_count::Int
end

struct TriangleRule{T<:AbstractFloat}
    points::Vector{SVector{2,T}}
    weights::Vector{T}
end

struct DuffyRule{T<:AbstractFloat}
    test_points::Vector{SVector{2,T}}
    trial_points::Vector{SVector{2,T}}
    weights::Vector{T}
end

struct FieldEvaluationCache{T<:AbstractFloat}
    source_points::Vector{SVector{3,T}}
    source_normals::Vector{SVector{3,T}}
    source_weights::Vector{T}
    source_faces::Vector{NTuple{3,Int}}
    source_elements::Vector{Int}
    basis_values::Vector{SVector{3,T}}
end

struct SingularCorrectionPair{T<:AbstractFloat}
    test_index::Int
    trial_index::Int
    rule_index::Int
    jac_scale::T
    normal_product::T
end

struct SingularCorrectionCache{T<:AbstractFloat}
    pairs_by_test::Vector{Vector{SingularCorrectionPair{T}}}
    pairs::Vector{SingularCorrectionPair{T}}
    rules::Vector{DuffyRule{T}}
    curls::Vector{NTuple{3,SVector{3,T}}}
    pair_count::Int
end

function load_gmsh22_with_tags(filepath::String, scale::T) where {T<:AbstractFloat}
    lines = readlines(filepath)
    node_start = findfirst(==("\$Nodes"), lines)
    node_end = findfirst(==("\$EndNodes"), lines)
    elem_start = findfirst(==("\$Elements"), lines)
    elem_end = findfirst(==("\$EndElements"), lines)

    if isnothing(node_start) || isnothing(node_end) || isnothing(elem_start) || isnothing(elem_end)
        error("Only Gmsh 2.2 ASCII meshes with Nodes and Elements sections are supported.")
    end

    node_index_map = Dict{Int,Int}()
    vertices = Vector{SVector{3,T}}()
    for i in (node_start + 2):(node_end - 1)
        parts = split(lines[i])
        length(parts) < 4 && continue
        gmsh_idx = parse(Int, parts[1])
        x = parse(T, parts[2]) * scale
        y = parse(T, parts[3]) * scale
        z = parse(T, parts[4]) * scale
        push!(vertices, SVector{3,T}(x, y, z))
        node_index_map[gmsh_idx] = length(vertices)
    end

    faces = Vector{NTuple{3,Int}}()
    physical_tags = Vector{Int}()
    for i in (elem_start + 2):(elem_end - 1)
        parts = split(lines[i])
        length(parts) < 8 && continue
        parse(Int, parts[2]) == 2 || continue

        n1 = get(node_index_map, parse(Int, parts[end - 2]), 0)
        n2 = get(node_index_map, parse(Int, parts[end - 1]), 0)
        n3 = get(node_index_map, parse(Int, parts[end]), 0)
        (n1 == 0 || n2 == 0 || n3 == 0) && continue

        push!(faces, (n1, n2, n3))
        push!(physical_tags, parse(Int, parts[4]))
    end

    return BoundaryMesh(vertices, faces, physical_tags)
end

function BoundaryMesh(vertices::Vector{SVector{3,T}}, faces::Vector{NTuple{3,Int}}, physical_tags::Vector{Int}) where {T}
    num_faces = length(faces)
    centroids = Vector{SVector{3,T}}(undef, num_faces)
    normals = Vector{SVector{3,T}}(undef, num_faces)
    areas = Vector{T}(undef, num_faces)
    face_vertices = Vector{NTuple{3,SVector{3,T}}}(undef, num_faces)

    for (i, face) in enumerate(faces)
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        v3 = vertices[face[3]]
        cross_prod = cross(v2 - v1, v3 - v1)

        centroids[i] = (v1 + v2 + v3) / T(3.0)
        areas[i] = norm(cross_prod) / T(2.0)
        normals[i] = cross_prod / norm(cross_prod)
        face_vertices[i] = (v1, v2, v3)
    end

    return BoundaryMesh{T}(vertices, faces, physical_tags, centroids, normals, areas, face_vertices)
end

build_p1_space(mesh::BoundaryMesh) = P1Space(mesh.faces, length(mesh.vertices))
build_dp0_space(mesh::BoundaryMesh) = DP0Space(collect(eachindex(mesh.faces)), length(mesh.faces))

function normalized_symmetry_mode(mode)
    mode_symbol = mode isa Symbol ? mode : Symbol(lowercase(strip(String(mode))))
    mode_symbol in (:off, :x, :xy) || error("Unsupported symmetry mode: $(mode). Expected off, x, or xy.")
    return mode_symbol
end

function symmetry_transforms(mode; include_identity::Bool=true)
    mode_symbol = normalized_symmetry_mode(mode)
    transforms = SymmetryTransform[]
    include_identity && push!(transforms, SymmetryTransform(:identity, SVector{3,Int}(1, 1, 1), 1))
    if mode_symbol == :x
        push!(transforms, SymmetryTransform(:x, SVector{3,Int}(-1, 1, 1), -1))
    elseif mode_symbol == :xy
        push!(transforms, SymmetryTransform(:x, SVector{3,Int}(-1, 1, 1), -1))
        push!(transforms, SymmetryTransform(:y, SVector{3,Int}(1, -1, 1), -1))
        push!(transforms, SymmetryTransform(:xy, SVector{3,Int}(-1, -1, 1), 1))
    end
    return tuple(transforms...)
end

symmetry_image_transforms(mode) = symmetry_transforms(mode; include_identity=false)
symmetry_reduction_factor(mode) = length(symmetry_transforms(mode; include_identity=true))

reflect_point(transform::SymmetryTransform, point::SVector{3,T}) where {T} = SVector{3,T}(
    T(transform.signs[1]) * point[1],
    T(transform.signs[2]) * point[2],
    T(transform.signs[3]) * point[3],
)
reflect_normal(transform::SymmetryTransform, normal::SVector{3,T}) where {T} = reflect_point(transform, normal)
reflect_curl(transform::SymmetryTransform, curl::SVector{3,T}) where {T} = SVector{3,T}(
    T(transform.determinant * transform.signs[1]) * curl[1],
    T(transform.determinant * transform.signs[2]) * curl[2],
    T(transform.determinant * transform.signs[3]) * curl[3],
)
reflect_vertices(transform::SymmetryTransform, vertices::NTuple{3,SVector{3,T}}) where {T} = (
    reflect_point(transform, vertices[1]),
    reflect_point(transform, vertices[2]),
    reflect_point(transform, vertices[3]),
)

function validate_symmetry_fundamental_domain!(mesh::BoundaryMesh{T}, mode; tolerance::T=T(1e-9)) where {T<:AbstractFloat}
    mode_symbol = normalized_symmetry_mode(mode)
    active_axes = mode_symbol == :off ? () : mode_symbol == :x ? (1,) : (1, 2)
    for axis in active_axes
        for (vertex_index, vertex) in enumerate(mesh.vertices)
            if vertex[axis] < -tolerance
                axis_name = axis == 1 ? "X" : axis == 2 ? "Y" : "Z"
                error(
                    "Mesh is not in the positive $(axis_name) fundamental domain for $(uppercase(String(mode_symbol))) symmetry. " *
                    "Vertex $(vertex_index) has $(lowercase(axis_name))=$(vertex[axis]) m."
                )
            end
        end
    end
    return nothing
end

function p1_symmetry_orbit_weights(mesh::BoundaryMesh{T}, mode; tolerance::T=T(1e-9)) where {T<:AbstractFloat}
    mode_symbol = normalized_symmetry_mode(mode)
    weights = ones(T, length(mesh.vertices))
    active_axes = mode_symbol == :off ? () : mode_symbol == :x ? (1,) : (1, 2)
    for (vertex_index, vertex) in enumerate(mesh.vertices)
        weight = T(1)
        for axis in active_axes
            abs(vertex[axis]) <= tolerance && (weight *= T(2))
        end
        weights[vertex_index] = weight
    end
    return weights
end

function elements_are_adjacent(face_a::NTuple{3,Int}, face_b::NTuple{3,Int})
    return face_a[1] == face_b[1] ||
        face_a[1] == face_b[2] ||
        face_a[1] == face_b[3] ||
        face_a[2] == face_b[1] ||
        face_a[2] == face_b[2] ||
        face_a[2] == face_b[3] ||
        face_a[3] == face_b[1] ||
        face_a[3] == face_b[2] ||
        face_a[3] == face_b[3]
end

function adjacency_info(face_a::NTuple{3,Int}, face_b::NTuple{3,Int})
    shared_a = Int[]
    shared_b = Int[]

    for i in 1:3
        for j in 1:3
            if face_a[i] == face_b[j]
                push!(shared_a, i)
                push!(shared_b, j)
            end
        end
    end

    if length(shared_a) == 3
        return (kind=:coincident, test_vertices=(1, 2, 3), trial_vertices=(1, 2, 3))
    elseif length(shared_a) == 2
        if shared_b[2] < shared_b[1]
            shared_a[1], shared_a[2] = shared_a[2], shared_a[1]
            shared_b[1], shared_b[2] = shared_b[2], shared_b[1]
        end
        return (kind=:edge_adjacent, test_vertices=(shared_a[1], shared_a[2]), trial_vertices=(shared_b[1], shared_b[2]))
    elseif length(shared_a) == 1
        return (kind=:vertex_adjacent, test_vertices=(shared_a[1],), trial_vertices=(shared_b[1],))
    end

    return (kind=:regular, test_vertices=(), trial_vertices=())
end

function geometric_adjacency_info(test_vertices, trial_vertices; tolerance)
    shared_a = Int[]
    shared_b = Int[]

    for i in 1:3
        for j in 1:3
            if norm(test_vertices[i] - trial_vertices[j]) <= tolerance
                push!(shared_a, i)
                push!(shared_b, j)
            end
        end
    end

    if length(shared_a) == 3
        return (kind=:coincident, test_vertices=(shared_a[1], shared_a[2], shared_a[3]), trial_vertices=(shared_b[1], shared_b[2], shared_b[3]))
    elseif length(shared_a) == 2
        if shared_b[2] < shared_b[1]
            shared_a[1], shared_a[2] = shared_a[2], shared_a[1]
            shared_b[1], shared_b[2] = shared_b[2], shared_b[1]
        end
        return (kind=:edge_adjacent, test_vertices=(shared_a[1], shared_a[2]), trial_vertices=(shared_b[1], shared_b[2]))
    elseif length(shared_a) == 1
        return (kind=:vertex_adjacent, test_vertices=(shared_a[1],), trial_vertices=(shared_b[1],))
    end

    return (kind=:regular, test_vertices=(), trial_vertices=())
end

function geometry_key(point::SVector{3,T}, tolerance::T) where {T<:AbstractFloat}
    return (
        Int(round(point[1] / tolerance)),
        Int(round(point[2] / tolerance)),
        Int(round(point[3] / tolerance)),
    )
end

function triangle_rule(::Type{T}, order::Int=2) where {T<:AbstractFloat}
    if order <= 1
        return TriangleRule([SVector{2,T}(T(1) / T(3), T(1) / T(3))], [T(0.5)])
    elseif order == 4
        return TriangleRule(
            [
                SVector{2,T}(T(0.4459484909159651), T(0.4459484909159651)),
                SVector{2,T}(T(0.0915762135097710), T(0.0915762135097700)),
                SVector{2,T}(T(0.1081030181680700), T(0.4459484909159651)),
                SVector{2,T}(T(0.4459484909159651), T(0.1081030181680700)),
                SVector{2,T}(T(0.8168475729804590), T(0.0915762135097700)),
                SVector{2,T}(T(0.0915762135097710), T(0.8168475729804580)),
            ],
            T(0.5) .* [
                T(0.2233815896780110),
                T(0.1099517436553220),
                T(0.2233815896780110),
                T(0.2233815896780110),
                T(0.1099517436553220),
                T(0.1099517436553220),
            ],
        )
    end

    return TriangleRule(
        [
            SVector{2,T}(T(1) / T(6), T(1) / T(6)),
            SVector{2,T}(T(2) / T(3), T(1) / T(6)),
            SVector{2,T}(T(1) / T(6), T(2) / T(3)),
        ],
        [T(1) / T(6), T(1) / T(6), T(1) / T(6)],
    )
end

function gauss_rule_1d(::Type{T}, order::Int) where {T<:AbstractFloat}
    if order == 1
        return [T(0.5)], [T(1.0)]
    elseif order == 2
        a = T(0.5) / sqrt(T(3.0))
        return [T(0.5) - a, T(0.5) + a], [T(0.5), T(0.5)]
    elseif order == 3
        a = sqrt(T(3.0) / T(5.0)) / T(2.0)
        return [T(0.5) - a, T(0.5), T(0.5) + a], [T(5.0) / T(18.0), T(4.0) / T(9.0), T(5.0) / T(18.0)]
    elseif order == 4
        x1 = sqrt(T(3.0) / T(7.0) - T(2.0) / T(7.0) * sqrt(T(6.0) / T(5.0))) / T(2.0)
        x2 = sqrt(T(3.0) / T(7.0) + T(2.0) / T(7.0) * sqrt(T(6.0) / T(5.0))) / T(2.0)
        w1 = (T(18.0) + sqrt(T(30.0))) / T(72.0)
        w2 = (T(18.0) - sqrt(T(30.0))) / T(72.0)
        return [T(0.5) - x2, T(0.5) - x1, T(0.5) + x1, T(0.5) + x2], [w2, w1, w1, w2]
    end

    error("Duffy 1D Gauss order must be between 1 and 4 in this implementation.")
end

function duffy_rule(::Type{T}, order::Int, adjacency::Symbol) where {T<:AbstractFloat}
    xreg, wreg = gauss_rule_1d(T, order)
    tensor_points = SVector{2,T}[]
    tensor_weights = T[]

    for i in eachindex(xreg)
        for j in eachindex(xreg)
            push!(tensor_points, SVector{2,T}(xreg[j], xreg[i]))
            push!(tensor_weights, wreg[i] * wreg[j])
        end
    end

    points_test = SVector{2,T}[]
    points_trial = SVector{2,T}[]
    weights = T[]

    for test_ind in eachindex(tensor_points)
        for trial_ind in eachindex(tensor_points)
            ptest = tensor_points[test_ind]
            ptrial = tensor_points[trial_ind]
            xsi = ptest[1]
            eta1 = ptest[2]
            eta2 = ptrial[1]
            eta3 = ptrial[2]
            eta12 = eta1 * eta2
            eta123 = eta12 * eta3
            base_weight = tensor_weights[test_ind] * tensor_weights[trial_ind]

            if adjacency == :coincident
                weight = base_weight * xsi^3 * eta1^2 * eta2
                append_duffy_point!(points_test, points_trial, weights, xsi, xsi * (T(1.0) - eta1 + eta12), xsi * (T(1.0) - eta123), xsi * (T(1.0) - eta1), weight)
                append_duffy_point!(points_test, points_trial, weights, xsi * (T(1.0) - eta123), xsi * (T(1.0) - eta1), xsi, xsi * (T(1.0) - eta1 + eta12), weight)
                append_duffy_point!(points_test, points_trial, weights, xsi, xsi * (eta1 - eta12 + eta123), xsi * (T(1.0) - eta12), xsi * (eta1 - eta12), weight)
                append_duffy_point!(points_test, points_trial, weights, xsi * (T(1.0) - eta12), xsi * (eta1 - eta12), xsi, xsi * (eta1 - eta12 + eta123), weight)
                append_duffy_point!(points_test, points_trial, weights, xsi * (T(1.0) - eta123), xsi * (eta1 - eta123), xsi, xsi * (eta1 - eta12), weight)
                append_duffy_point!(points_test, points_trial, weights, xsi, xsi * (eta1 - eta12), xsi * (T(1.0) - eta123), xsi * (eta1 - eta123), weight)
            elseif adjacency == :edge_adjacent
                weight = base_weight * xsi^3 * eta1^2
                append_duffy_point!(points_test, points_trial, weights, xsi, xsi * eta1 * eta3, xsi * (T(1.0) - eta12), xsi * eta1 * (T(1.0) - eta2), weight)
                append_duffy_point!(points_test, points_trial, weights, xsi, xsi * eta1, xsi * (T(1.0) - eta123), xsi * eta1 * eta2 * (T(1.0) - eta3), weight * eta2)
                append_duffy_point!(points_test, points_trial, weights, xsi * (T(1.0) - eta12), xsi * eta1 * (T(1.0) - eta2), xsi, xsi * eta123, weight * eta2)
                append_duffy_point!(points_test, points_trial, weights, xsi * (T(1.0) - eta123), xsi * eta12 * (T(1.0) - eta3), xsi, xsi * eta1, weight * eta2)
                append_duffy_point!(points_test, points_trial, weights, xsi * (T(1.0) - eta123), xsi * eta1 * (T(1.0) - eta2 * eta3), xsi, xsi * eta12, weight * eta2)
            elseif adjacency == :vertex_adjacent
                weight = base_weight * xsi^3 * eta2
                append_duffy_point!(points_test, points_trial, weights, xsi, xsi * eta1, xsi * eta2, xsi * eta2 * eta3, weight)
                append_duffy_point!(points_test, points_trial, weights, xsi * eta2, xsi * eta2 * eta3, xsi, xsi * eta1, weight)
            else
                error("Unknown Duffy adjacency: $adjacency")
            end
        end
    end

    return DuffyRule(points_test, points_trial, weights)
end

function append_duffy_point!(points_test, points_trial, weights, test_x, test_y, trial_x, trial_y, weight)
    push!(points_test, SVector(test_x - test_y, test_y))
    push!(points_trial, SVector(trial_x - trial_y, trial_y))
    push!(weights, weight)
end

function remap_shared_vertex(point::SVector{2,T}, vertex_id::Int) where {T}
    if vertex_id == 1
        return point
    elseif vertex_id == 2
        return SVector{2,T}(T(1.0) - point[1] - point[2], point[2])
    elseif vertex_id == 3
        return SVector{2,T}(point[1], T(1.0) - point[1] - point[2])
    end
    error("vertex_id must be 1, 2, or 3.")
end

function remap_shared_edge(point::SVector{2,T}, shared_vertex1::Int, shared_vertex2::Int) where {T}
    ref_vertices = (
        SVector{2,T}(T(0.0), T(0.0)),
        SVector{2,T}(T(1.0), T(0.0)),
        SVector{2,T}(T(0.0), T(1.0)),
    )
    remaining = 6 - shared_vertex1 - shared_vertex2
    v0 = ref_vertices[shared_vertex1]
    v1 = ref_vertices[shared_vertex2]
    v2 = ref_vertices[remaining]
    return v0 + point[1] * (v1 - v0) + point[2] * (v2 - v0)
end

function fibonacci_sphere(n_points::Int, radius::T) where {T<:AbstractFloat}
    points = Vector{SVector{3,T}}(undef, n_points)
    golden_angle = T(pi * (3.0 - sqrt(5.0)))

    for i in 0:(n_points - 1)
        z = T(1.0 - (2.0 * i + 1.0) / n_points)
        r = sqrt(T(1.0) - z * z)
        phi = T(i) * golden_angle
        points[i + 1] = SVector{3,T}(r * cos(phi) * radius, r * sin(phi) * radius, z * radius)
    end

    return points
end

function mesh_for_frequency(meshes, freq)
    for (max_freq, path) in meshes
        freq <= max_freq && return path
    end
    return meshes[end][2]
end

function local_to_global(vertices::NTuple{3,SVector{3,T}}, local_point::SVector{2,T}) where {T}
    xi, eta = local_point
    return (T(1) - xi - eta) * vertices[1] + xi * vertices[2] + eta * vertices[3]
end

p1_values(local_point::SVector{2,T}) where {T} = SVector{3,T}(T(1) - local_point[1] - local_point[2], local_point[1], local_point[2])

function surface_gradients(vertices::NTuple{3,SVector{3,T}}) where {T}
    e1 = vertices[2] - vertices[1]
    e2 = vertices[3] - vertices[1]
    gram = SMatrix{2,2,T}(dot(e1, e1), dot(e2, e1), dot(e1, e2), dot(e2, e2))
    jac = SMatrix{3,2,T}(e1[1], e1[2], e1[3], e2[1], e2[2], e2[3])
    lift = jac * inv(gram)
    ref_grads = (
        SVector{2,T}(T(-1.0), T(-1.0)),
        SVector{2,T}(T(1.0), T(0.0)),
        SVector{2,T}(T(0.0), T(1.0)),
    )
    return (lift * ref_grads[1], lift * ref_grads[2], lift * ref_grads[3])
end

function surface_curls(vertices::NTuple{3,SVector{3,T}}, normal::SVector{3,T}) where {T}
    grads = surface_gradients(vertices)
    return (cross(normal, grads[1]), cross(normal, grads[2]), cross(normal, grads[3]))
end

function helmholtz_single_layer_kernel(x, y, k::T) where {T<:AbstractFloat}
    radius = norm(y - x)
    radius == zero(T) && return zero(Complex{T})
    return exp(Complex{T}(1im) * k * radius) / (T(4.0) * T(pi) * radius)
end
helmholtz_single_layer_kernel(x, y, test_normal, trial_normal, k::T) where {T<:AbstractFloat} = helmholtz_single_layer_kernel(x, y, k)

function helmholtz_double_layer_kernel(x, y, source_normal, k::T) where {T<:AbstractFloat}
    r_vec = y - x
    radius = norm(r_vec)
    radius == zero(T) && return zero(Complex{T})
    green = exp(Complex{T}(1im) * k * radius) / (T(4.0) * T(pi) * radius)
    grad_source = green * (Complex{T}(1im) * k - T(1.0) / radius) * (r_vec / radius)
    return sum(grad_source .* source_normal)
end
helmholtz_double_layer_kernel(x, y, test_normal, trial_normal, k::T) where {T<:AbstractFloat} = helmholtz_double_layer_kernel(x, y, trial_normal, k)

function helmholtz_adjoint_double_layer_kernel(x, y, test_normal, k::T) where {T<:AbstractFloat}
    r_vec = y - x
    radius = norm(r_vec)
    radius == zero(T) && return zero(Complex{T})
    green = exp(Complex{T}(1im) * k * radius) / (T(4.0) * T(pi) * radius)
    grad_test = -green * (Complex{T}(1im) * k - T(1.0) / radius) * (r_vec / radius)
    return sum(grad_test .* test_normal)
end
helmholtz_adjoint_double_layer_kernel(x, y, test_normal, trial_normal, k::T) where {T<:AbstractFloat} = helmholtz_adjoint_double_layer_kernel(x, y, test_normal, k)

function l2_identity_element_matrix(test_area::T, test_basis::Symbol, trial_basis::Symbol, rule::TriangleRule{T}) where {T}
    test_dofs = test_basis == :p1 ? 3 : 1
    trial_dofs = trial_basis == :p1 ? 3 : 1
    block = zeros(T, test_dofs, trial_dofs)
    jac_scale = T(2.0) * test_area

    for (point, weight) in zip(rule.points, rule.weights)
        test_vals = test_basis == :p1 ? p1_values(point) : SVector{1,T}(T(1.0))
        trial_vals = trial_basis == :p1 ? p1_values(point) : SVector{1,T}(T(1.0))

        for i in 1:test_dofs
            for j in 1:trial_dofs
                block[i, j] += test_vals[i] * trial_vals[j] * weight * jac_scale
            end
        end
    end

    return block
end

function remap_singular_point(point, kind::Symbol, vertices)
    if kind == :coincident
        return point
    elseif kind == :edge_adjacent
        return remap_shared_edge(point, vertices[1], vertices[2])
    elseif kind == :vertex_adjacent
        return remap_shared_vertex(point, vertices[1])
    end
    return point
end

function remap_singular_point(point, kind::Symbol, vertex_a::Int, vertex_b::Int)
    if kind == :coincident
        return point
    elseif kind == :edge_adjacent
        return remap_shared_edge(point, vertex_a, vertex_b)
    elseif kind == :vertex_adjacent
        return remap_shared_vertex(point, vertex_a)
    end
    return point
end

function scatter_element_block!(global_matrix, block, test_dofs, trial_dofs)
    for local_row in eachindex(test_dofs)
        global_row = test_dofs[local_row]
        for local_col in eachindex(trial_dofs)
            global_col = trial_dofs[local_col]
            global_matrix[global_row, global_col] += block[local_row, local_col]
        end
    end
    return global_matrix
end

function adjacent_trial_indices_by_test(mesh::BoundaryMesh, element_indices)
    indices = collect(element_indices)
    index_set = Set(indices)
    vertex_to_elements = Dict{Int,Vector{Int}}()

    for element_index in indices
        for vertex in mesh.faces[element_index]
            push!(get!(vertex_to_elements, vertex, Int[]), element_index)
        end
    end

    adjacent = Dict{Int,Vector{Int}}()
    for test_index in indices
        candidates = Int[]
        seen = Set{Int}()
        for vertex in mesh.faces[test_index]
            for trial_index in get(vertex_to_elements, vertex, Int[])
                if trial_index in index_set && !(trial_index in seen)
                    push!(candidates, trial_index)
                    push!(seen, trial_index)
                end
            end
        end
        adjacent[test_index] = candidates
    end

    return adjacent
end

function singular_kind_code(kind::Symbol)
    kind == :coincident && return 1
    kind == :edge_adjacent && return 2
    kind == :vertex_adjacent && return 3
    error("Unsupported singular adjacency kind: $(kind).")
end

function remapped_duffy_rule(
    base_rule::DuffyRule{T},
    kind::Symbol,
    test_vertex_a::Int,
    test_vertex_b::Int,
    trial_vertex_a::Int,
    trial_vertex_b::Int,
) where {T<:AbstractFloat}
    test_points = Vector{SVector{2,T}}(undef, length(base_rule.weights))
    trial_points = Vector{SVector{2,T}}(undef, length(base_rule.weights))

    for q in eachindex(base_rule.weights)
        test_points[q] = remap_singular_point(base_rule.test_points[q], kind, test_vertex_a, test_vertex_b)
        trial_points[q] = remap_singular_point(base_rule.trial_points[q], kind, trial_vertex_a, trial_vertex_b)
    end

    return DuffyRule(test_points, trial_points, base_rule.weights)
end

function singular_orientation_key(info)
    if info.kind == :coincident
        return (singular_kind_code(info.kind), 0, 0, 0, 0)
    elseif info.kind == :edge_adjacent
        return (
            singular_kind_code(info.kind),
            info.test_vertices[1],
            info.test_vertices[2],
            info.trial_vertices[1],
            info.trial_vertices[2],
        )
    elseif info.kind == :vertex_adjacent
        return (
            singular_kind_code(info.kind),
            info.test_vertices[1],
            0,
            info.trial_vertices[1],
            0,
        )
    end
    error("Cannot build singular correction rule for adjacency kind $(info.kind).")
end

function rule_for_singular_orientation!(rules, rule_indices, base_rules, info)
    key = singular_orientation_key(info)
    existing = get(rule_indices, key, 0)
    existing != 0 && return existing

    kind_code, test_a, test_b, trial_a, trial_b = key
    kind = kind_code == 1 ? :coincident : kind_code == 2 ? :edge_adjacent : :vertex_adjacent
    base_rule = base_rules[kind]
    push!(rules, remapped_duffy_rule(base_rule, kind, test_a, test_b, trial_a, trial_b))
    rule_indices[key] = length(rules)
    return length(rules)
end

function build_singular_correction_cache(
    mesh::BoundaryMesh{T},
    singular_order::Int,
    element_indices=eachindex(mesh.faces),
) where {T<:AbstractFloat}
    adjacent = adjacent_trial_indices_by_test(mesh, element_indices)
    pairs_by_test = [SingularCorrectionPair{T}[] for _ in eachindex(mesh.faces)]
    base_rules = Dict(
        :coincident => duffy_rule(T, singular_order, :coincident),
        :edge_adjacent => duffy_rule(T, singular_order, :edge_adjacent),
        :vertex_adjacent => duffy_rule(T, singular_order, :vertex_adjacent),
    )
    rules = DuffyRule{T}[]
    rule_indices = Dict{NTuple{5,Int},Int}()
    curls = [surface_curls(mesh.face_vertices[element_index], mesh.normals[element_index]) for element_index in eachindex(mesh.faces)]
    pairs = SingularCorrectionPair{T}[]
    pair_count = 0

    for test_index in collect(element_indices)
        test_face = mesh.faces[test_index]
        for trial_index in adjacent[test_index]
            info = adjacency_info(test_face, mesh.faces[trial_index])
            info.kind == :regular && continue
            rule_index = rule_for_singular_orientation!(rules, rule_indices, base_rules, info)
            jac_scale = (T(2.0) * mesh.areas[test_index]) * (T(2.0) * mesh.areas[trial_index])
            normal_product = dot(mesh.normals[test_index], mesh.normals[trial_index])
            pair = SingularCorrectionPair(test_index, trial_index, rule_index, jac_scale, normal_product)
            push!(pairs_by_test[test_index], pair)
            push!(pairs, pair)
            pair_count += 1
        end
    end

    return SingularCorrectionCache(
        pairs_by_test,
        pairs,
        rules,
        curls,
        pair_count,
    )
end

function image_singular_candidates(mesh::BoundaryMesh{T}, element_indices, transform::SymmetryTransform; tolerance::T=T(1e-8)) where {T<:AbstractFloat}
    indices = collect(element_indices)
    index_set = Set(indices)
    test_elements_by_vertex_key = Dict{NTuple{3,Int},Vector{Int}}()
    for test_index in indices
        for vertex in mesh.face_vertices[test_index]
            key = geometry_key(vertex, tolerance)
            push!(get!(test_elements_by_vertex_key, key, Int[]), test_index)
        end
    end

    candidates = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    for trial_index in indices
        reflected_vertices = reflect_vertices(transform, mesh.face_vertices[trial_index])
        for vertex in reflected_vertices
            key = geometry_key(vertex, tolerance)
            for test_index in get(test_elements_by_vertex_key, key, Int[])
                test_index in index_set || continue
                candidate = (test_index, trial_index)
                if !(candidate in seen)
                    push!(candidates, candidate)
                    push!(seen, candidate)
                end
            end
        end
    end
    return candidates
end

function count_adjacent_pairs(mesh::BoundaryMesh, element_indices)
    adjacent = adjacent_trial_indices_by_test(mesh, element_indices)
    return sum(length, values(adjacent))
end

function assemble_l2_identity_matrix(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    rule::TriangleRule{T},
    test_basis::Symbol,
    trial_basis::Symbol,
    ;
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    test_dof_count = test_basis == :p1 ? p1_space.global_dof_count : dp0_space.global_dof_count
    trial_dof_count = trial_basis == :p1 ? p1_space.global_dof_count : dp0_space.global_dof_count
    matrix = zeros(T, test_dof_count, trial_dof_count)

    for element_index in eachindex(mesh.faces)
        test_dofs = test_basis == :p1 ? p1_space.local_to_global[element_index] : (dp0_space.local_to_global[element_index],)
        trial_dofs = trial_basis == :p1 ? p1_space.local_to_global[element_index] : (dp0_space.local_to_global[element_index],)
        block = l2_identity_element_matrix(mesh.areas[element_index], test_basis, trial_basis, rule)
        scatter_element_block!(matrix, block, test_dofs, trial_dofs)
    end

    if test_basis == :p1
        weights = p1_symmetry_orbit_weights(mesh, symmetry_mode)
        matrix .*= reshape(weights, :, 1)
    end

    return matrix
end

function assemble_regular_galerkin_operators(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    rule::TriangleRule{T};
    skip_singular::Bool=true,
    singular_order::Int=2,
    element_indices=eachindex(mesh.faces),
    threaded::Bool=true,
    use_cuda_regular::Bool=true,
    cuda_cache=nothing,
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
    if !use_cuda_regular
        return assemble_regular_galerkin_operators_cpu(
            mesh,
            p1_space,
            dp0_space,
            k,
            rule;
            skip_singular=skip_singular,
            singular_order=singular_order,
            element_indices=element_indices,
            threaded=threaded,
            timing=timing,
            singular_cache=singular_cache,
            symmetry_mode=symmetry_mode,
        )
    end

    parallel_quadrature || error("Balanced CUDA regular assembly requires parallel quadrature.")
    regular_assembly_mode in (:split_atomic_balanced, :split_atomic_balanced_multipair, :split_atomic_slp_hyp_separate) || error("Unsupported CUDA regular assembly mode: $(regular_assembly_mode).")
    return assemble_regular_galerkin_operators_cuda_regular(
        mesh,
        p1_space,
        dp0_space,
        k,
        rule;
        skip_singular=skip_singular,
        singular_order=singular_order,
        element_indices=element_indices,
        cache=cuda_cache,
        return_gpu=return_gpu,
        parallel_quadrature=true,
        timing=timing,
        singular_cache=singular_cache,
        cuda_singular_cache=cuda_singular_cache,
        profile_regular_kernel=profile_regular_kernel,
        regular_probe_pair_limit=regular_probe_pair_limit,
        regular_kernel_threads_override=regular_kernel_threads_override,
        regular_assembly_mode=regular_assembly_mode,
        symmetry_mode=symmetry_mode,
    )
end

function build_cuda_regular_assembly_cache(args...; kwargs...)
    error("CUDA regular-pair assembly cache requested, but CUDA.jl is not loaded.")
end

function assemble_regular_galerkin_operators_cuda_regular(args...; kwargs...)
    error("CUDA regular-pair assembly requested, but CUDA.jl is not loaded.")
end

function build_cuda_field_evaluation_cache(args...; kwargs...)
    error("CUDA field-evaluation cache requested, but CUDA.jl is not loaded.")
end

function evaluate_galerkin_field_cuda(args...; kwargs...)
    error("CUDA field evaluation requested, but CUDA.jl is not loaded.")
end

release_operator_storage!(operators) = nothing

function solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k::T) where {T<:AbstractFloat}
    coupling = Complex{T}(0, 1) / k
    operators_on_gpu = get(operators, :on_gpu, false)
    if !operators_on_gpu
        return solve_burton_miller_neumann_cpu(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)
    end
    cuda = cuda_module()
    cuda.functional() || error("CUDA solve requested, but CUDA.functional() is false.")
    d_identity_p1_p1 = cuda.CuArray(Complex{T}.(identity_p1_p1))
    d_identity_p1_dp0 = cuda.CuArray(Complex{T}.(identity_p1_dp0))
    d_q_neumann = cuda.CuArray(q_neumann)
    d_lhs = Complex{T}(0.5) .* d_identity_p1_p1 .- operators.double_layer .+ coupling .* operators.hypersingular
    d_rhs = (-operators.single_layer .- coupling .* (operators.adjoint_double_layer .+ Complex{T}(0.5) .* d_identity_p1_dp0)) * d_q_neumann
    cuda.unsafe_free!(d_identity_p1_p1)
    cuda.unsafe_free!(d_identity_p1_dp0)
    cuda.unsafe_free!(d_q_neumann)

    d_pressure = d_lhs \ d_rhs
    pressure = Complex{T}.(Array(d_pressure))
    cuda.unsafe_free!(d_lhs)
    cuda.unsafe_free!(d_rhs)
    cuda.unsafe_free!(d_pressure)
    return pressure
end

function build_field_evaluation_cache(mesh::BoundaryMesh{T}, rule::TriangleRule{T}; symmetry_mode::Symbol=:off) where {T<:AbstractFloat}
    transforms = symmetry_transforms(symmetry_mode; include_identity=true)
    source_count = length(mesh.faces) * length(rule.points) * length(transforms)
    source_points = Vector{SVector{3,T}}(undef, source_count)
    source_normals = Vector{SVector{3,T}}(undef, source_count)
    source_weights = Vector{T}(undef, source_count)
    source_faces = Vector{NTuple{3,Int}}(undef, source_count)
    source_elements = Vector{Int}(undef, source_count)
    basis_values = Vector{SVector{3,T}}(undef, source_count)

    source_index = 1
    for element_index in eachindex(mesh.faces)
        vertices = mesh.face_vertices[element_index]
        face = mesh.faces[element_index]
        jac_scale = T(2.0) * mesh.areas[element_index]

        for transform in transforms
            transformed_vertices = reflect_vertices(transform, vertices)
            transformed_normal = reflect_normal(transform, mesh.normals[element_index])
            for q_index in eachindex(rule.points)
                point = rule.points[q_index]
                source_points[source_index] = local_to_global(transformed_vertices, point)
                source_normals[source_index] = transformed_normal
                source_weights[source_index] = rule.weights[q_index] * jac_scale
                source_faces[source_index] = face
                source_elements[source_index] = element_index
                basis_values[source_index] = p1_values(point)
                source_index += 1
            end
        end
    end

    return FieldEvaluationCache(
        source_points,
        source_normals,
        source_weights,
        source_faces,
        source_elements,
        basis_values,
    )
end

include(joinpath(@__DIR__, "BeatEngineCpu.jl"))

if CUDA_MODULE !== nothing
    include(joinpath(@__DIR__, "BeatEngineCuda.jl"))
end

end
