using Test

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

const CUDA_MODULE = try
    @eval import CUDA
    CUDA
catch
    nothing
end

cuda_available() = CUDA_MODULE !== nothing && CUDA_MODULE.functional()

@testset "mesh setup" begin
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001))
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 2)
    singular_cache = build_singular_correction_cache(mesh, 2)

    @test length(mesh.faces) > 0
    @test length(mesh.vertices) > 0
    @test p1.global_dof_count == length(mesh.vertices)
    @test dp0.global_dof_count == length(mesh.faces)
    @test length(rule.points) == length(rule.weights)
    @test singular_cache.pair_count > 0
end

@testset "cpu production pipeline" begin
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001))
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 2)
    k = Float32(2pi * 1000.0 / 343.0)
    element_indices = 1:min(16, length(mesh.faces))
    singular_cache = build_singular_correction_cache(mesh, 2, element_indices)

    operators = assemble_regular_galerkin_operators(
        mesh,
        p1,
        dp0,
        k,
        rule;
        skip_singular=false,
        singular_order=2,
        element_indices=element_indices,
        use_cuda_regular=false,
        singular_cache=singular_cache,
    )

    @test !get(operators, :on_gpu, true)
    expected_cpu_mode = Threads.nthreads() > 1 ? :cpu_colored_threads : :cpu_serial
    expected_cpu_kernel = Threads.nthreads() > 1 ? "cpu_colored_threads" : "cpu_serial"
    @test operators.regular_assembly_mode == expected_cpu_mode
    @test operators.regular_kernel_mode == expected_cpu_kernel
    @test operators.cpu_color_count >= 1
    @test operators.regular_pairs > 0
    @test operators.singular_pairs == singular_cache.pair_count
    @test sum(abs2, operators.single_layer) > 0
    @test sum(abs2, operators.double_layer) > 0
    @test sum(abs2, operators.adjoint_double_layer) > 0
    @test sum(abs2, operators.hypersingular) > 0
    @test all(isfinite, real.(operators.single_layer))
    @test all(isfinite, imag.(operators.single_layer))

    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
    q_neumann = zeros(ComplexF32, length(mesh.faces))
    q_neumann[1] = ComplexF32(0, 1)
    pressure = solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)
    solve_system = build_burton_miller_neumann_cpu_system(operators, identity_p1_p1, identity_p1_dp0, k)
    pressure_from_system = solve_burton_miller_neumann_cpu_system(solve_system, q_neumann, Float32)

    @test length(pressure) == p1.global_dof_count
    @test all(isfinite, real.(pressure))
    @test all(isfinite, imag.(pressure))
    @test pressure_from_system ≈ pressure rtol=Float32(1e-4) atol=Float32(1e-4)

    field_cache = build_field_evaluation_cache(mesh, rule)
    eval_points = fibonacci_sphere(8, Float32(2.0))
    field = evaluate_galerkin_field_cpu(eval_points, mesh, pressure, q_neumann, k, field_cache)
    @test length(field) == length(eval_points)
    @test all(isfinite, real.(field))
    @test all(isfinite, imag.(field))

end

@testset "cpu x symmetry assembly" begin
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample_half.msh"), Float32(0.001))
    validate_symmetry_fundamental_domain!(mesh, :x)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 2)
    k = Float32(2pi * 1000.0 / 343.0)
    element_indices = 1:min(16, length(mesh.faces))
    singular_cache = build_singular_correction_cache(mesh, 2, element_indices)

    operators = assemble_regular_galerkin_operators(
        mesh,
        p1,
        dp0,
        k,
        rule;
        skip_singular=false,
        singular_order=2,
        element_indices=element_indices,
        use_cuda_regular=false,
        singular_cache=singular_cache,
        symmetry_mode=:x,
    )

    @test !get(operators, :on_gpu, true)
    @test operators.regular_pairs > length(element_indices) * length(element_indices)
    @test operators.singular_pairs == singular_cache.pair_count
    @test operators.image_singular_pairs >= 0
    @test sum(abs2, operators.single_layer) > 0
    @test all(isfinite, real.(operators.double_layer))
    @test all(isfinite, imag.(operators.double_layer))

    if cuda_available()
        cuda_cache = build_cuda_regular_assembly_cache(mesh, rule; element_indices=element_indices)
        cuda_singular_cache = BeatEngineCore.build_cuda_singular_correction_cache(singular_cache, p1, dp0)
        cuda_operators = assemble_regular_galerkin_operators(
            mesh,
            p1,
            dp0,
            k,
            rule;
            skip_singular=false,
            singular_order=2,
            element_indices=element_indices,
            cuda_cache=cuda_cache,
            singular_cache=singular_cache,
            cuda_singular_cache=cuda_singular_cache,
            symmetry_mode=:x,
        )

        @test operators.single_layer ≈ Array(cuda_operators.single_layer) rtol=Float32(5e-3) atol=Float32(5e-5)
        @test operators.double_layer ≈ Array(cuda_operators.double_layer) rtol=Float32(5e-3) atol=Float32(5e-5)
        @test operators.adjoint_double_layer ≈ Array(cuda_operators.adjoint_double_layer) rtol=Float32(5e-3) atol=Float32(5e-5)
        @test operators.hypersingular ≈ Array(cuda_operators.hypersingular) rtol=Float32(5e-3) atol=Float32(5e-3)
        release_operator_storage!(cuda_operators)
    end

    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=:x)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=:x)
    q_neumann = zeros(ComplexF32, length(mesh.faces))
    q_neumann[1] = ComplexF32(0, 1)
    pressure = solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)
    @test length(pressure) == p1.global_dof_count
    @test all(isfinite, real.(pressure))
    @test all(isfinite, imag.(pressure))

    field_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=:x)
    eval_points = fibonacci_sphere(8, Float32(2.0))
    field = evaluate_galerkin_field_cpu(eval_points, mesh, pressure, q_neumann, k, field_cache)
    @test length(field) == length(eval_points)
    @test all(isfinite, real.(field))
    @test all(isfinite, imag.(field))
end

@testset "cpu xy symmetry assembly" begin
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample_quarter.msh"), Float32(0.001))
    validate_symmetry_fundamental_domain!(mesh, :xy)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 2)
    k = Float32(2pi * 1000.0 / 343.0)
    element_indices = 1:min(16, length(mesh.faces))
    singular_cache = build_singular_correction_cache(mesh, 2, element_indices)

    operators = assemble_regular_galerkin_operators(
        mesh,
        p1,
        dp0,
        k,
        rule;
        skip_singular=false,
        singular_order=2,
        element_indices=element_indices,
        use_cuda_regular=false,
        singular_cache=singular_cache,
        symmetry_mode=:xy,
    )

    @test !get(operators, :on_gpu, true)
    @test operators.regular_pairs > 2 * length(element_indices) * length(element_indices)
    @test operators.singular_pairs == singular_cache.pair_count
    @test operators.image_singular_pairs >= 0
    @test sum(abs2, operators.single_layer) > 0
    @test all(isfinite, real.(operators.hypersingular))
    @test all(isfinite, imag.(operators.hypersingular))
end

@testset "cuda production pipeline" begin
    if !cuda_available()
        @test_skip "CUDA unavailable; skipping CUDA-only BEAT Engine tests."
    else
        mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001))
        p1 = build_p1_space(mesh)
        dp0 = build_dp0_space(mesh)
        rule = triangle_rule(Float32, 2)
        k = Float32(2pi * 1000.0 / 343.0)
        element_indices = 1:min(16, length(mesh.faces))
        singular_cache = build_singular_correction_cache(mesh, 2, element_indices)
        cuda_cache = build_cuda_regular_assembly_cache(mesh, rule; element_indices=element_indices)
        cuda_singular_cache = BeatEngineCore.build_cuda_singular_correction_cache(singular_cache, p1, dp0)

        operators = assemble_regular_galerkin_operators(
            mesh,
            p1,
            dp0,
            k,
            rule;
            skip_singular=false,
            singular_order=2,
            element_indices=element_indices,
            cuda_cache=cuda_cache,
            singular_cache=singular_cache,
            cuda_singular_cache=cuda_singular_cache,
            regular_assembly_mode=:split_atomic_balanced,
        )

        @test get(operators, :on_gpu, false)
        @test operators.regular_assembly_mode == :split_atomic_balanced
        @test operators.regular_kernel_mode == "split_atomic_balanced"
        @test operators.regular_pairs > 0
        @test operators.singular_pairs == singular_cache.pair_count

        identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
        identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
        q_neumann = zeros(ComplexF32, length(mesh.faces))
        q_neumann[1] = ComplexF32(0, 1)
        pressure = solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)

        @test length(pressure) == p1.global_dof_count
        @test all(isfinite, real.(pressure))
        @test all(isfinite, imag.(pressure))

        field_cache = build_cuda_field_evaluation_cache(build_field_evaluation_cache(mesh, rule))
        eval_points = fibonacci_sphere(8, Float32(2.0))
        field = evaluate_galerkin_field_cuda(eval_points, mesh, pressure, q_neumann, k, field_cache)
        @test length(field) == length(eval_points)
        @test all(isfinite, real.(field))
        @test all(isfinite, imag.(field))

        release_operator_storage!(operators)
    end
end
