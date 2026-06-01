include(joinpath(@__DIR__, "..", "src", "JBEMCore.jl"))

using .JBEMCore

mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample_quarter.msh"), Float32(0.001))
rule = triangle_rule(Float32, 2)
off_cache = build_field_evaluation_cache(mesh, rule)
xy_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=:xy)
factor = length(xy_cache.source_points) ÷ length(off_cache.source_points)

factor == 4 || error("Expected XY field cache to have 4x sources; got $(factor)x.")
println((off=length(off_cache.source_points), xy=length(xy_cache.source_points), factor=factor))
