using Test
using StaticArrays

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

const CUDA_MODULE = try
    @eval import CUDA
    CUDA
catch
    nothing
end

cuda_available() = CUDA_MODULE !== nothing && CUDA_MODULE.functional()

const AMDGPU_MODULE = try
    @eval import AMDGPU
    AMDGPU
catch
    nothing
end

rocm_available() = AMDGPU_MODULE !== nothing &&
                   AMDGPU_MODULE.functional() &&
                   AMDGPU_MODULE.functional(:rocblas) &&
                   AMDGPU_MODULE.functional(:rocsolver)

@testset "symmetry plane snapping" begin
    vertices = [
        SVector{3,Float64}(-1.2e-8, 0.0, 0.8),
        SVector{3,Float64}(0.5, 0.0, 0.8),
        SVector{3,Float64}(0.0, 0.5, 0.8),
    ]
    mesh = BoundaryMesh(vertices, [(1, 2, 3)], [1])

    tolerance = symmetry_plane_tolerance(mesh.vertices)
    snapped = snap_symmetry_planes(mesh, :x)

    @test tolerance ≈ sqrt(0.5) * 1.0e-7
    @test snapped.vertices[1][1] == 0.0
    @test mesh.vertices[1][1] == -1.2e-8
    validate_symmetry_fundamental_domain!(mesh, :x)
    @test_throws ErrorException validate_symmetry_fundamental_domain!(
        mesh,
        :x;
        tolerance=1.0e-9,
    )
end

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

include(joinpath(@__DIR__, "coupled_solver_tests.jl"))
include(joinpath(@__DIR__, "coupled_condensed_tests.jl"))

@testset "cpu BLAS thread policy" begin
    @test beat_cpu_blas_thread_count(441; available_threads=16) == 1
    @test beat_cpu_blas_thread_count(1390; available_threads=16) == 4
    @test beat_cpu_blas_thread_count(3502; available_threads=16) == 8
    @test beat_cpu_blas_thread_count(5000; available_threads=16) == 16
    @test beat_cpu_blas_thread_count(3502; available_threads=4) == 4
    @test beat_cpu_blas_thread_count(441; available_threads=16, override="3") == 3
    @test_throws ErrorException beat_cpu_blas_thread_count(441; available_threads=16, override="invalid")
    @test_throws ErrorException beat_cpu_blas_thread_count(441; available_threads=16, override="0")
end

@testset "cpu production pipeline" begin
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001))
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float32, 2)
    k = Float32(2pi * 1000.0 / 343.0)
    element_indices = 1:min(16, length(mesh.faces))
    singular_cache = build_singular_correction_cache(mesh, 2, element_indices)
    off_cache = build_beat_cpu_assembly_cache(
        mesh,
        p1,
        dp0,
        rule;
        singular_order=2,
        element_indices=element_indices,
        symmetry_mode=:off,
    )
    @test isempty(off_cache.image_transforms)

    operators = assemble_regular_galerkin_operators(
        mesh,
        p1,
        dp0,
        k,
        rule;
        skip_singular=false,
        singular_order=2,
        element_indices=element_indices,
        backend=:cpu,
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
        backend=:cpu,
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
            device_cache=cuda_cache,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular_cache,
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
        backend=:cpu,
        singular_cache=singular_cache,
        symmetry_mode=:xy,
    )
    cpu_cache = build_beat_cpu_assembly_cache(
        mesh,
        p1,
        dp0,
        rule;
        singular_order=2,
        element_indices=element_indices,
        symmetry_mode=:xy,
    )
    cached_operators = assemble_regular_galerkin_operators(
        mesh,
        p1,
        dp0,
        k,
        rule;
        skip_singular=false,
        singular_order=2,
        element_indices=element_indices,
        backend=:cpu,
        singular_cache=singular_cache,
        cpu_cache=cpu_cache,
        symmetry_mode=:xy,
    )

    @test !get(operators, :on_gpu, true)
    @test operators.regular_pairs > 2 * length(element_indices) * length(element_indices)
    @test operators.singular_pairs == singular_cache.pair_count
    @test operators.image_singular_pairs >= 0
    @test sum(abs2, operators.single_layer) > 0
    @test all(isfinite, real.(operators.hypersingular))
    @test all(isfinite, imag.(operators.hypersingular))
    @test cached_operators.single_layer ≈ operators.single_layer
    @test cached_operators.double_layer ≈ operators.double_layer
    @test cached_operators.adjoint_double_layer ≈ operators.adjoint_double_layer
    @test cached_operators.hypersingular ≈ operators.hypersingular
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
            device_cache=cuda_cache,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular_cache,
        )

        @test get(operators, :on_gpu, false)
        @test operators.regular_assembly_mode == :serial_pair_batched
        @test operators.regular_kernel_mode == "serial_pair_batched"
        @test operators.regular_pairs > 0
        @test operators.singular_pairs == singular_cache.pair_count
        @test BeatEngineCore._cuda_use_matrix_free_burton_miller_rhs(operators)

        identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
        identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
        identity_cache = build_cuda_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, Float32)
        q_neumann = zeros(ComplexF32, length(mesh.faces))
        q_neumann[1] = ComplexF32(0, 1)
        d_q_neumann = CUDA_MODULE.CuArray(q_neumann)
        coupling = ComplexF32(0, 1) / k
        d_expected_rhs = (
            -operators.single_layer .-
            coupling .* (operators.adjoint_double_layer .+ ComplexF32(0.5) .* identity_cache.identity_p1_dp0)
        ) * d_q_neumann
        d_matrix_free_rhs = BeatEngineCore._cuda_burton_miller_rhs(operators, identity_cache, d_q_neumann, coupling)
        @test Array(d_matrix_free_rhs) ≈ Array(d_expected_rhs) rtol=2f-5
        CUDA_MODULE.unsafe_free!(d_q_neumann)
        CUDA_MODULE.unsafe_free!(d_expected_rhs)
        CUDA_MODULE.unsafe_free!(d_matrix_free_rhs)

        pressure = solve_burton_miller_neumann(operators, identity_cache, q_neumann, k)
        release_cuda_burton_miller_identity_cache!(identity_cache)

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

        symmetry_mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "sample_quarter.msh"), Float32(0.001))
        symmetry_p1 = build_p1_space(symmetry_mesh)
        symmetry_dp0 = build_dp0_space(symmetry_mesh)
        symmetry_indices = eachindex(symmetry_mesh.faces)
        symmetry_singular_cache = build_singular_correction_cache(symmetry_mesh, 2, symmetry_indices)
        symmetry_cuda_cache = build_cuda_regular_assembly_cache(symmetry_mesh, rule; element_indices=symmetry_indices)
        symmetry_cuda_singular_cache = BeatEngineCore.build_cuda_singular_correction_cache(
            symmetry_singular_cache,
            symmetry_p1,
            symmetry_dp0,
        )
        image_cache = build_cuda_image_singular_correction_cache(
            symmetry_mesh,
            symmetry_p1,
            symmetry_dp0,
            2,
            symmetry_indices,
            :xy,
        )
        image_timing = Dict{String,Float64}()
        symmetry_operators = assemble_regular_galerkin_operators(
            symmetry_mesh,
            symmetry_p1,
            symmetry_dp0,
            k,
            rule;
            skip_singular=false,
            singular_order=2,
            element_indices=symmetry_indices,
            device_cache=symmetry_cuda_cache,
            singular_cache=symmetry_singular_cache,
            device_singular_cache=symmetry_cuda_singular_cache,
            device_image_singular_cache=image_cache,
            symmetry_mode=:xy,
            timing=image_timing,
        )
        @test image_cache.pair_count > 0
        @test symmetry_operators.image_singular_pairs == image_cache.pair_count
        @test image_timing["image_singular_correction_cuda_cache_build"] == 0.0
        @test !BeatEngineCore._cuda_use_matrix_free_burton_miller_rhs(symmetry_operators)
        release_operator_storage!(symmetry_operators)
        release_cuda_image_singular_correction_cache!(image_cache)
    end
end
