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
    println("CUDA unavailable; skipping GPU-resident solve smoke test.")
    exit()
end

mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001))
p1 = build_p1_space(mesh)
dp0 = build_dp0_space(mesh)
rule = triangle_rule(Float32, 2)
subset = 1:24
k = Float32(2pi * 1000.0 / 343.0)
q_neumann = zeros(ComplexF32, length(mesh.faces))
q_neumann[1:4] .= ComplexF32(1im)

identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
cache = build_cuda_regular_assembly_cache(mesh, rule; element_indices=subset)

cpu_ops = assemble_regular_galerkin_operators(
    mesh,
    p1,
    dp0,
    k,
    rule;
    skip_singular=false,
    singular_order=2,
    element_indices=subset,
)

gpu_ops = assemble_regular_galerkin_operators(
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
    return_gpu=true,
)

cpu_pressure = solve_burton_miller_neumann(cpu_ops, identity_p1_p1, identity_p1_dp0, q_neumann, k)
gpu_pressure = solve_burton_miller_neumann(gpu_ops, identity_p1_p1, identity_p1_dp0, q_neumann, k)

println((
    subset_faces=length(subset),
    gpu_ops_on_gpu=gpu_ops.on_gpu,
    regular_pairs=(cpu_ops.regular_pairs, gpu_ops.regular_pairs),
    singular_pairs=(cpu_ops.singular_pairs, gpu_ops.singular_pairs),
    pressure_error=norm(cpu_pressure - gpu_pressure),
    pressure_norm=norm(cpu_pressure),
))
