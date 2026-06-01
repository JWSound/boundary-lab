include(joinpath(@__DIR__, "..", "src", "JBEMCore.jl"))

using LinearAlgebra
using .JBEMCore

mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001))
rule = duffy_rule(Float32, 4, :edge_adjacent)
k = Float32(2pi * 1000.0 / 343.0)

test_index = 1
trial_index = findfirst(i -> i != test_index && adjacency_info(mesh.faces[test_index], mesh.faces[i]).kind != :regular, eachindex(mesh.faces))
trial_index === nothing && error("No adjacent face found for fused singular smoke test.")

info = adjacency_info(mesh.faces[test_index], mesh.faces[trial_index])
rule = duffy_rule(Float32, 4, info.kind)

slp_old = singular_galerkin_element_matrix(
    mesh.face_vertices[test_index],
    mesh.face_vertices[trial_index],
    mesh.areas[test_index],
    mesh.areas[trial_index],
    mesh.normals[test_index],
    mesh.normals[trial_index],
    k,
    helmholtz_single_layer_kernel,
    :p1,
    :dp0,
    rule,
    info,
)
dlp_old = singular_galerkin_element_matrix(
    mesh.face_vertices[test_index],
    mesh.face_vertices[trial_index],
    mesh.areas[test_index],
    mesh.areas[trial_index],
    mesh.normals[test_index],
    mesh.normals[trial_index],
    k,
    helmholtz_double_layer_kernel,
    :p1,
    :p1,
    rule,
    info,
)
adj_old = singular_galerkin_element_matrix(
    mesh.face_vertices[test_index],
    mesh.face_vertices[trial_index],
    mesh.areas[test_index],
    mesh.areas[trial_index],
    mesh.normals[test_index],
    mesh.normals[trial_index],
    k,
    helmholtz_adjoint_double_layer_kernel,
    :p1,
    :dp0,
    rule,
    info,
)
hyp_old = JBEMCore.singular_hypersingular_element_matrix(
    mesh.face_vertices[test_index],
    mesh.face_vertices[trial_index],
    mesh.areas[test_index],
    mesh.areas[trial_index],
    mesh.normals[test_index],
    mesh.normals[trial_index],
    k,
    rule,
    info,
)

slp_new, dlp_new, adj_new, hyp_new = JBEMCore.singular_galerkin_operator_blocks(
    mesh.face_vertices[test_index],
    mesh.face_vertices[trial_index],
    mesh.areas[test_index],
    mesh.areas[trial_index],
    mesh.normals[test_index],
    mesh.normals[trial_index],
    k,
    rule,
    info,
)

println((
    adjacency=info.kind,
    slp_error=norm(Array(slp_new) - slp_old),
    dlp_error=norm(Array(dlp_new) - dlp_old),
    adj_error=norm(Array(adj_new) - adj_old),
    hyp_error=norm(Array(hyp_new) - hyp_old),
))
