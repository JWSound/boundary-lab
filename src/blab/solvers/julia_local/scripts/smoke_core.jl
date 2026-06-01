include(joinpath(@__DIR__, "..", "src", "JBEMCore.jl"))

using .JBEMCore

mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float64(0.001))
p1 = build_p1_space(mesh)
dp0 = build_dp0_space(mesh)
rule = triangle_rule(Float64, 2)

block = regular_galerkin_element_matrix(
    mesh.face_vertices[1],
    mesh.face_vertices[10],
    mesh.areas[1],
    mesh.areas[10],
    mesh.normals[1],
    mesh.normals[10],
    10.0,
    helmholtz_single_layer_kernel,
    :p1,
    :dp0,
    rule,
)

println((
    faces=length(mesh.faces),
    vertices=length(mesh.vertices),
    p1_dofs=p1.global_dof_count,
    dp0_dofs=dp0.global_dof_count,
    block_size=size(block),
    sample_value=block[1, 1],
))
