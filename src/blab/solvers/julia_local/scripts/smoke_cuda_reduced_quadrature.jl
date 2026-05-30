include(joinpath(@__DIR__, "..", "src", "JBEMCore.jl"))

using LinearAlgebra
using .JBEMCore

const CUDA_MODULE = try
    @eval import CUDA
    CUDA
catch
    nothing
end

if CUDA_MODULE === nothing || !CUDA_MODULE.functional()
    println("CUDA unavailable; skipping reduced quadrature smoke test.")
    exit()
end

mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "sample.msh"), Float32(0.001))
p1 = build_p1_space(mesh)
dp0 = build_dp0_space(mesh)
rule = triangle_rule(Float32, 2)
subset = 1:24
k = Float32(2pi * 1000.0 / 343.0)
cache = build_cuda_regular_assembly_cache(mesh, rule; element_indices=subset)

cpu = assemble_regular_galerkin_operators(
    mesh,
    p1,
    dp0,
    k,
    rule;
    skip_singular=false,
    singular_order=2,
    element_indices=subset,
)

gpu = assemble_regular_galerkin_operators(
    mesh,
    p1,
    dp0,
    k,
    rule;
    skip_singular=false,
    singular_order=2,
    element_indices=subset,
    use_cuda_regular=true,
    cuda_cache=cache,
    parallel_quadrature=true,
)

println((
    regular_pairs=(cpu.regular_pairs, gpu.regular_pairs),
    singular_pairs=(cpu.singular_pairs, gpu.singular_pairs),
    single_layer_error=norm(cpu.single_layer - gpu.single_layer),
    double_layer_error=norm(cpu.double_layer - gpu.double_layer),
    adjoint_double_layer_error=norm(cpu.adjoint_double_layer - gpu.adjoint_double_layer),
    hypersingular_error=norm(cpu.hypersingular - gpu.hypersingular),
))
