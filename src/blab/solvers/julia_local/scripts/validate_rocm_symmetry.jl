using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

relative_error(actual, reference) = norm(actual - reference) / max(norm(reference), eps(real(eltype(reference))))

function validate_fixture(mesh_name::String, symmetry_mode::Symbol)
    mesh_path = joinpath(@__DIR__, "..", "test_meshes", mesh_name)
    mesh = load_gmsh22_with_tags(mesh_path, Float32(0.001))
    validate_symmetry_fundamental_domain!(mesh, symmetry_mode)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 1)
    singular_order = 2
    k = Float32(2pi) * 500.0f0 / 343.0f0
    singular_cache = build_singular_correction_cache(mesh, singular_order)

    cpu_cache = build_beat_cpu_assembly_cache(
        mesh, p1, dp0, rule;
        singular_order=singular_order,
        symmetry_mode=symmetry_mode,
    )
    cpu_operators = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, rule;
        skip_singular=false,
        singular_order=singular_order,
        backend=:cpu,
        singular_cache=singular_cache,
        cpu_cache=cpu_cache,
        symmetry_mode=symmetry_mode,
    )
    rocm_cache = build_rocm_regular_assembly_cache(
        mesh, p1, dp0, rule;
        singular_order=singular_order,
        symmetry_mode=symmetry_mode,
    )
    rocm_operators = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, rule;
        skip_singular=false,
        singular_order=singular_order,
        backend=:rocm,
        device_cache=rocm_cache,
        singular_cache=singular_cache,
        rocm_assembly_mode=:native,
        symmetry_mode=symmetry_mode,
    )
    operator_errors = Dict(
        "single_layer" => relative_error(Array(rocm_operators.single_layer), cpu_operators.single_layer),
        "double_layer" => relative_error(Array(rocm_operators.double_layer), cpu_operators.double_layer),
        "adjoint_double_layer" => relative_error(Array(rocm_operators.adjoint_double_layer), cpu_operators.adjoint_double_layer),
        "hypersingular" => relative_error(Array(rocm_operators.hypersingular), cpu_operators.hypersingular),
    )

    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry_mode)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry_mode)
    q_neumann = zeros(ComplexF32, length(mesh.faces))
    driven_tag = first(mesh.physical_tags)
    q_neumann[mesh.physical_tags .== driven_tag] .= ComplexF32(0, 1)
    cpu_pressure = solve_burton_miller_neumann(
        cpu_operators, identity_p1_p1, identity_p1_dp0, q_neumann, k,
    )
    identity_cache = build_rocm_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, Float32)
    rocm_pressure = solve_burton_miller_neumann(rocm_operators, identity_cache, q_neumann, k)
    pressure_error = relative_error(rocm_pressure, cpu_pressure)

    eval_points = fibonacci_sphere(16, Float32(2))
    cpu_field_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode)
    cpu_field = evaluate_galerkin_field_cpu(eval_points, mesh, cpu_pressure, q_neumann, k, cpu_field_cache)
    rocm_field_cache = build_rocm_field_evaluation_cache(cpu_field_cache)
    rocm_field = evaluate_galerkin_field_rocm(eval_points, mesh, rocm_pressure, q_neumann, k, rocm_field_cache)
    field_error = relative_error(rocm_field, cpu_field)

    println((
        mesh=mesh_name,
        symmetry=symmetry_mode,
        faces=length(mesh.faces),
        image_singular_pairs=rocm_operators.image_singular_pairs,
        operator_errors=operator_errors,
        pressure_error=pressure_error,
        field_error=field_error,
    ))
    flush(stdout)
    maximum(values(operator_errors)) <= 2.0f-6 || error("ROCm symmetry operators differ from BEAT CPU.")
    pressure_error <= 5.0f-3 || error("ROCm symmetry pressure differs from BEAT CPU.")
    field_error <= 5.0f-3 || error("ROCm symmetry field differs from BEAT CPU.")

    release_rocm_field_evaluation_cache!(rocm_field_cache)
    release_rocm_burton_miller_identity_cache!(identity_cache)
    release_operator_storage!(rocm_operators)
    release_rocm_regular_assembly_cache!(rocm_cache)
end

function validate_rocm_symmetry()
    amdgpu = BeatEngineCore.AMDGPU_MODULE
    amdgpu === nothing && error("AMDGPU.jl did not load.")
    amdgpu.functional() || error("AMDGPU.functional() is false.")
    println("device=$(amdgpu.device())")
    validate_fixture("sample_half.msh", :x)
    validate_fixture("sample_quarter.msh", :xy)
    println("ROCM_SYMMETRY_VALIDATION_OK")
end

validate_rocm_symmetry()
