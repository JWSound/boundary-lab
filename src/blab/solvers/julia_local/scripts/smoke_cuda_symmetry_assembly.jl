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
    println("CUDA unavailable; skipping CUDA symmetry assembly smoke test.")
    exit()
end

mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001))
p1 = build_p1_space(mesh)
dp0 = build_dp0_space(mesh)
rule = triangle_rule(Float32, 2)
subset = 1:12
k = Float32(2pi * 1000.0 / 343.0)

off = assemble_regular_galerkin_operators(
    mesh,
    p1,
    dp0,
    k,
    rule;
    skip_singular=true,
    element_indices=subset,
    use_cuda_regular=true,
    symmetry_mode=:off,
)

x = assemble_regular_galerkin_operators(
    mesh,
    p1,
    dp0,
    k,
    rule;
    skip_singular=true,
    element_indices=subset,
    use_cuda_regular=true,
    symmetry_mode=:x,
)

@assert x.regular_pairs == off.regular_pairs + length(subset) * length(subset)
@assert norm(x.single_layer - off.single_layer) > 0

println((
    subset_faces=length(subset),
    off_regular_pairs=off.regular_pairs,
    x_regular_pairs=x.regular_pairs,
    image_single_layer_delta=norm(x.single_layer - off.single_layer),
))
