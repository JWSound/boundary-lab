include(joinpath(@__DIR__, "..", "src", "JBEMCore.jl"))

using LinearAlgebra
using .JBEMCore

mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float64(0.001))
p1 = build_p1_space(mesh)
dp0 = build_dp0_space(mesh)
rule = triangle_rule(Float64, 2)
subset = 1:24
k = 2pi * 1000.0 / 343.0

ops_skip = assemble_regular_galerkin_operators(
    mesh,
    p1,
    dp0,
    k,
    rule;
    element_indices=subset,
)

ops_singular = assemble_regular_galerkin_operators(
    mesh,
    p1,
    dp0,
    k,
    rule;
    skip_singular=false,
    singular_order=2,
    element_indices=subset,
)

println((
    subset_faces=length(subset),
    single_layer_size=size(ops_singular.single_layer),
    double_layer_size=size(ops_singular.double_layer),
    regular_pairs=ops_singular.regular_pairs,
    singular_pairs=ops_singular.singular_pairs,
    skipped_pairs=ops_singular.skipped_pairs,
    skip_mode_skipped_pairs=ops_skip.skipped_pairs,
    single_layer_norm=norm(ops_singular.single_layer),
    double_layer_norm=norm(ops_singular.double_layer),
))
