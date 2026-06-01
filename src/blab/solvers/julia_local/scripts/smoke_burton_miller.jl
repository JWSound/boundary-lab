include(joinpath(@__DIR__, "..", "src", "JBEMCore.jl"))

using LinearAlgebra
using .JBEMCore

const CUDA_MODULE = try
    @eval import CUDA
    CUDA
catch err
    println("CUDA smoke check failed: ", typeof(err), ": ", err)
    nothing
end

if CUDA_MODULE === nothing || !CUDA_MODULE.functional()
    println("CUDA unavailable; skipping Burton-Miller solve smoke test.")
    exit()
end

mesh_path = joinpath(@__DIR__, "..", "test_meshes", "two_tetrahedra.msh")
mesh = load_gmsh22_with_tags(mesh_path, Float64(1.0))
p1 = build_p1_space(mesh)
dp0 = build_dp0_space(mesh)
rule = triangle_rule(Float64, 4)
k = 10.0

operators = assemble_regular_galerkin_operators(
    mesh,
    p1,
    dp0,
    k,
    rule;
    skip_singular=false,
    singular_order=4,
)
identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
q_neumann = ones(ComplexF64, dp0.global_dof_count)
pressure = solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)
eval_points = fibonacci_sphere(12, 2.0)
field = evaluate_galerkin_field(eval_points, mesh, pressure, q_neumann, k, rule)

println((
    p1_dofs=p1.global_dof_count,
    dp0_dofs=dp0.global_dof_count,
    pressure_norm=norm(pressure),
    field_norm=norm(field),
    pressure_isfinite=all(isfinite, real.(pressure)) && all(isfinite, imag.(pressure)),
    field_isfinite=all(isfinite, real.(field)) && all(isfinite, imag.(field)),
    gpu_available=true,
))
