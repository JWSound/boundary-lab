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

    @test tolerance ≈ sqrt(0.5) * 1.0e-6
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

@testset "close-pair higher-order correction" begin
    T = Float32
    vertices = [
        SVector{3,T}(0, 0, 0),
        SVector{3,T}(0.04, 0, 0),
        SVector{3,T}(0, 0.04, 0),
        SVector{3,T}(0, 0, 0.01),
        SVector{3,T}(0.04, 0, 0.01),
        SVector{3,T}(0, 0.04, 0.01),
    ]
    mesh = BoundaryMesh(vertices, [(1, 2, 3), (4, 5, 6)], [1, 1])
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    base_rule = triangle_rule(T, 2)
    near_cache = build_near_correction_cache(mesh, [(1, 2, 4), (2, 1, 6)], 6)
    empty_near_cache = build_near_correction_cache(mesh, Tuple{Int,Int}[], 6)
    singular_cache = build_singular_correction_cache(mesh, 2)
    k = T(2pi * 100.0 / 343.0)

    @test near_cache.pair_count == 2
    @test empty_near_cache.pair_count == 0
    @test empty_near_cache.correction_orders == Int[]
    @test near_cache.correction_order == 6
    @test near_cache.correction_orders == [4, 6]
    @test length(near_cache.correction_rules[1].points) == 16
    @test length(near_cache.correction_rules[2].points) == 36
    @test sum(near_cache.correction_rules[1].weights) ≈ T(0.5) atol=T(1e-6)
    @test sum(near_cache.correction_rules[2].weights) ≈ T(0.5) atol=T(1e-6)

    base = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, base_rule;
        backend=:cpu, skip_singular=false, singular_cache=singular_cache,
    )
    corrected = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, base_rule;
        backend=:cpu, skip_singular=false, singular_cache=singular_cache,
        near_correction_cache=near_cache,
    )
    reference_forward = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, near_cache.correction_rules[1];
        backend=:cpu, skip_singular=false, singular_cache=singular_cache,
    )
    reference_reverse = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, near_cache.correction_rules[2];
        backend=:cpu, skip_singular=false, singular_cache=singular_cache,
    )

    @test corrected.near_pair_count == 2
    @test corrected.near_pair_quadrature_order == 6
    @test corrected.single_layer[1:3, 2] ≈ reference_forward.single_layer[1:3, 2] rtol=T(2e-5)
    @test corrected.double_layer[1:3, 4:6] ≈ reference_forward.double_layer[1:3, 4:6] rtol=T(2e-5) atol=T(2e-7)
    @test corrected.adjoint_double_layer[1:3, 2] ≈ reference_forward.adjoint_double_layer[1:3, 2] rtol=T(2e-5) atol=T(2e-7)
    @test corrected.hypersingular[1:3, 4:6] ≈ reference_forward.hypersingular[1:3, 4:6] rtol=T(2e-5) atol=T(2e-5)
    @test corrected.single_layer[4:6, 1] ≈ reference_reverse.single_layer[4:6, 1] rtol=T(2e-5)
    @test norm(corrected.single_layer[1:3, 2] - base.single_layer[1:3, 2]) > T(1e-9)

    ground_image = SymmetryTransform(:ground_image, SVector{3,Int}(1, -1, 1), -1)
    image_cache = build_near_correction_cache(
        mesh,
        [(1, 2)],
        6;
        trial_transform=ground_image,
    )
    @test image_cache.pair_count == 1
    @test image_cache.trial_transform == ground_image

    image_mesh = BoundaryMesh(
        [
            SVector{3,T}(0.01, 0, 0),
            SVector{3,T}(0.01, 0.04, 0),
            SVector{3,T}(0.01, 0, 0.04),
        ],
        [(1, 2, 3)],
        [1],
    )
    image_p1 = build_p1_space(image_mesh)
    image_dp0 = build_dp0_space(image_mesh)
    x_image = SymmetryTransform(:reflect_x, SVector{3,Int}(-1, 1, 1), -1)
    reflected_near_cache = build_near_correction_cache(
        image_mesh,
        [(1, 1)],
        6;
        trial_transform=x_image,
    )
    image_singular_cache = build_singular_correction_cache(image_mesh, 2)
    reflected_corrected = assemble_regular_galerkin_operators(
        image_mesh, image_p1, image_dp0, k, base_rule;
        backend=:cpu, skip_singular=false, singular_cache=image_singular_cache,
        near_correction_cache=reflected_near_cache, symmetry_mode=:x,
    )
    reflected_reference = assemble_regular_galerkin_operators(
        image_mesh, image_p1, image_dp0, k, reflected_near_cache.correction_rules[1];
        backend=:cpu, skip_singular=false, singular_cache=image_singular_cache,
        symmetry_mode=:x,
    )
    @test reflected_corrected.single_layer ≈ reflected_reference.single_layer rtol=T(2e-5)
    @test reflected_corrected.double_layer ≈ reflected_reference.double_layer rtol=T(2e-5) atol=T(2e-7)
    @test reflected_corrected.adjoint_double_layer ≈ reflected_reference.adjoint_double_layer rtol=T(2e-5) atol=T(2e-7)
    @test reflected_corrected.hypersingular ≈ reflected_reference.hypersingular rtol=T(2e-5) atol=T(2e-5)

    if cuda_available()
        cuda_regular = build_cuda_regular_assembly_cache(mesh, base_rule)
        cuda_singular = BeatEngineCore.build_cuda_singular_correction_cache(singular_cache, p1, dp0)
        cuda_near = build_cuda_near_correction_cache(near_cache, p1, dp0)
        cuda_corrected = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, base_rule;
            skip_singular=false,
            device_cache=cuda_regular,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular,
            near_correction_cache=near_cache,
            device_near_correction_cache=cuda_near,
        )
        @test cuda_corrected.near_pair_count == near_cache.pair_count
        @test Array(cuda_corrected.single_layer) ≈ corrected.single_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_corrected.double_layer) ≈ corrected.double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_corrected.adjoint_double_layer) ≈ corrected.adjoint_double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_corrected.hypersingular) ≈ corrected.hypersingular rtol=T(5e-3) atol=T(5e-3)
        release_operator_storage!(cuda_corrected)
        release_cuda_image_singular_correction_cache!(cuda_near)
    end
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
