include(joinpath(@__DIR__, "..", "src", "JBEMCore.jl"))

using .JBEMCore
using LinearAlgebra

const CUDA_MODULE = try
    @eval import CUDA
    CUDA
catch
    nothing
end

if CUDA_MODULE === nothing || !CUDA_MODULE.functional()
    println("CUDA unavailable; skipping symmetry operator-action comparison.")
    exit()
end

const STAGED_DIR = joinpath(@__DIR__, "..", "test_meshes", "staged_symmetry")

function point_key(point; tol=1.0f-8)
    return ntuple(i -> Int(round(point[i] / tol)), 3)
end

function reduced_point_key(point, mode::Symbol; tol=1.0f-8)
    x = mode in (:x, :xy) ? abs(point[1]) : point[1]
    y = mode == :xy ? abs(point[2]) : point[2]
    return point_key((x, y, point[3]); tol=tol)
end

function face_reduction_map(mesh::BoundaryMesh{T}, reduced_vertex_by_key, mode::Symbol) where {T}
    mapping = Dict{Tuple{Int,Int,Int},Int}()
    for (face_index, face) in enumerate(mesh.faces)
        reduced_vertices = ntuple(i -> reduced_vertex_by_key[reduced_point_key(mesh.vertices[face[i]], mode)], 3)
        mapping[Tuple(sort(collect(reduced_vertices)))] = face_index
    end
    return mapping
end

function full_face_to_reduced_indices(full_mesh::BoundaryMesh{T}, reduced_vertex_by_key, reduced_face_by_vertices, mode::Symbol) where {T}
    indices = Vector{Int}(undef, length(full_mesh.faces))
    for (face_index, face) in enumerate(full_mesh.faces)
        reduced_vertices = ntuple(i -> reduced_vertex_by_key[reduced_point_key(full_mesh.vertices[face[i]], mode)], 3)
        indices[face_index] = reduced_face_by_vertices[Tuple(sort(collect(reduced_vertices)))]
    end
    return indices
end

function deterministic_p(vertex_index::Int, ::Type{T}) where {T<:AbstractFloat}
    return Complex{T}(sin(T(0.37) * T(vertex_index)), cos(T(0.23) * T(vertex_index)))
end

function deterministic_q(face_index::Int, ::Type{T}) where {T<:AbstractFloat}
    return Complex{T}(cos(T(0.19) * T(face_index)), sin(T(0.41) * T(face_index)))
end

function relative_error(actual, expected)
    denom = max(norm(expected), eps(real(eltype(expected))))
    return norm(actual - expected) / denom
end

function compare_case(label::String, mode::Symbol; subset_count::Int=96, skip_singular::Bool=false)
    reduced_mesh = load_gmsh22_with_tags(joinpath(STAGED_DIR, "$(label)_reduced_clean.msh"), Float32(0.001))
    full_mesh = load_gmsh22_with_tags(joinpath(STAGED_DIR, "$(label)_full_clean.msh"), Float32(0.001))
    validate_symmetry_fundamental_domain!(reduced_mesh, mode)

    reduced_vertex_by_key = Dict(point_key(vertex) => index for (index, vertex) in enumerate(reduced_mesh.vertices))
    full_identity_vertex_by_key = Dict(point_key(vertex) => index for (index, vertex) in enumerate(full_mesh.vertices))
    reduced_face_by_vertices = face_reduction_map(reduced_mesh, reduced_vertex_by_key, mode)
    full_to_reduced_face = full_face_to_reduced_indices(full_mesh, reduced_vertex_by_key, reduced_face_by_vertices, mode)

    subset = collect(1:min(subset_count, length(reduced_mesh.faces)))
    subset_set = Set(subset)
    full_subset = [index for (index, reduced_index) in enumerate(full_to_reduced_face) if reduced_index in subset_set]
    test_reduced_vertices = sort(collect(Set(vertex for face_index in subset for vertex in reduced_mesh.faces[face_index])))
    nonseam_reduced_vertices = [
        vertex_index for vertex_index in test_reduced_vertices
        if all(reduced_mesh.vertices[vertex_index][axis] > Float32(1e-7) for axis in (mode == :x ? (1,) : (1, 2)))
    ]
    full_test_vertices = [full_identity_vertex_by_key[point_key(reduced_mesh.vertices[index])] for index in test_reduced_vertices]
    full_nonseam_vertices = [full_identity_vertex_by_key[point_key(reduced_mesh.vertices[index])] for index in nonseam_reduced_vertices]

    p1_reduced = build_p1_space(reduced_mesh)
    dp0_reduced = build_dp0_space(reduced_mesh)
    p1_full = build_p1_space(full_mesh)
    dp0_full = build_dp0_space(full_mesh)
    rule = triangle_rule(Float32, 2)
    k = Float32(2pi * 1000.0 / 343.0)

    reduced_ops = assemble_regular_galerkin_operators(
        reduced_mesh,
        p1_reduced,
        dp0_reduced,
        k,
        rule;
        skip_singular=skip_singular,
        singular_order=2,
        element_indices=subset,
        use_cuda_regular=true,
        symmetry_mode=mode,
    )
    full_ops = assemble_regular_galerkin_operators(
        full_mesh,
        p1_full,
        dp0_full,
        k,
        rule;
        skip_singular=skip_singular,
        singular_order=2,
        element_indices=full_subset,
        use_cuda_regular=true,
        symmetry_mode=:off,
    )

    p_reduced = zeros(ComplexF32, length(reduced_mesh.vertices))
    for vertex_index in eachindex(p_reduced)
        p_reduced[vertex_index] = deterministic_p(vertex_index, Float32)
    end
    q_reduced = zeros(ComplexF32, length(reduced_mesh.faces))
    for face_index in subset
        q_reduced[face_index] = deterministic_q(face_index, Float32)
    end

    p_full = zeros(ComplexF32, length(full_mesh.vertices))
    for (vertex_index, vertex) in enumerate(full_mesh.vertices)
        reduced_index = reduced_vertex_by_key[reduced_point_key(vertex, mode)]
        p_full[vertex_index] = p_reduced[reduced_index]
    end
    q_full = zeros(ComplexF32, length(full_mesh.faces))
    for face_index in full_subset
        q_full[face_index] = q_reduced[full_to_reduced_face[face_index]]
    end

    reduced_actions = (
        single=reduced_ops.single_layer * q_reduced,
        adjoint=reduced_ops.adjoint_double_layer * q_reduced,
        double=reduced_ops.double_layer * p_reduced,
        hypersingular=reduced_ops.hypersingular * p_reduced,
    )
    full_actions = (
        single=full_ops.single_layer * q_full,
        adjoint=full_ops.adjoint_double_layer * q_full,
        double=full_ops.double_layer * p_full,
        hypersingular=full_ops.hypersingular * p_full,
    )

    errors_all = Dict{Symbol,Float32}()
    errors_nonseam = Dict{Symbol,Float32}()
    for name in (:single, :adjoint, :double, :hypersingular)
        errors_all[name] = Float32(relative_error(reduced_actions[name][test_reduced_vertices], full_actions[name][full_test_vertices]))
        errors_nonseam[name] = isempty(nonseam_reduced_vertices) ? Float32(NaN) :
            Float32(relative_error(reduced_actions[name][nonseam_reduced_vertices], full_actions[name][full_nonseam_vertices]))
    end

    return (
        label=label,
        mode=mode,
        reduced_faces=length(reduced_mesh.faces),
        full_faces=length(full_mesh.faces),
        subset_faces=length(subset),
        skip_singular=skip_singular,
        full_subset_faces=length(full_subset),
        test_vertices=length(test_reduced_vertices),
        nonseam_test_vertices=length(nonseam_reduced_vertices),
        reduced_regular_pairs=reduced_ops.regular_pairs,
        full_regular_pairs=full_ops.regular_pairs,
        image_singular_pairs=get(reduced_ops, :image_singular_pairs, 0),
        errors_all=errors_all,
        errors_nonseam=errors_nonseam,
    )
end

println(compare_case("sample_half", :x; skip_singular=true))
println(compare_case("sample_half", :x; skip_singular=false))
println(compare_case("sample_quarter", :xy; skip_singular=true))
println(compare_case("sample_quarter", :xy; skip_singular=false))
