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
include(joinpath(@__DIR__, "speaker_rom_tests.jl"))
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

@testset "rigid y0 half-space Green function" begin
    T = Float32
    direct_vertices = [
        SVector{3,T}(0, 0.2, 0),
        SVector{3,T}(0.04, 0.2, 0),
        SVector{3,T}(0, 0.2, 0.04),
    ]
    direct_mesh = BoundaryMesh(direct_vertices, [(1, 2, 3)], [1])
    full_vertices = vcat(
        direct_vertices,
        [SVector{3,T}(point[1], -point[2], point[3]) for point in direct_vertices],
    )
    # Reverse the reflected triangle so its outward normal is the physical
    # reflection of the direct triangle's normal.
    full_mesh = BoundaryMesh(full_vertices, [(1, 2, 3), (4, 6, 5)], [1, 1])
    direct_p1 = build_p1_space(direct_mesh)
    direct_dp0 = build_dp0_space(direct_mesh)
    full_p1 = build_p1_space(full_mesh)
    full_dp0 = build_dp0_space(full_mesh)
    rule = triangle_rule(T, 3)
    k = T(2pi * 80.0 / 343.0)

    direct_singular = build_singular_correction_cache(direct_mesh, 3)
    full_singular = build_singular_correction_cache(full_mesh, 3)
    half_space = assemble_regular_galerkin_operators(
        direct_mesh, direct_p1, direct_dp0, k, rule;
        backend=:cpu, skip_singular=false, singular_cache=direct_singular,
        symmetry_mode=:ground,
    )
    full_space = assemble_regular_galerkin_operators(
        full_mesh, full_p1, full_dp0, k, rule;
        backend=:cpu, skip_singular=false, singular_cache=full_singular,
    )

    @test rigid_ground_transform().signs == SVector{3,Int}(1, -1, 1)
    @test half_space.single_layer[:, 1] ≈
        full_space.single_layer[1:3, 1] + full_space.single_layer[1:3, 2] rtol=T(2e-5)
    @test half_space.adjoint_double_layer[:, 1] ≈
        full_space.adjoint_double_layer[1:3, 1] + full_space.adjoint_double_layer[1:3, 2] rtol=T(2e-5) atol=T(2e-7)
    @test half_space.double_layer ≈
        full_space.double_layer[1:3, 1:3] + full_space.double_layer[1:3, 4:6] rtol=T(2e-5) atol=T(2e-7)
    @test half_space.hypersingular ≈
        full_space.hypersingular[1:3, 1:3] + full_space.hypersingular[1:3, 4:6] rtol=T(2e-5) atol=T(2e-5)

    identity_off = assemble_l2_identity_matrix(direct_mesh, direct_p1, direct_dp0, rule, :p1, :p1)
    identity_ground = assemble_l2_identity_matrix(
        direct_mesh, direct_p1, direct_dp0, rule, :p1, :p1;
        symmetry_mode=:ground,
    )
    @test identity_ground == identity_off

    pressure = Complex{T}[1.0 + 0.2im, 0.7 - 0.1im, 1.2 + 0.05im]
    neumann = Complex{T}[0.3 - 0.2im]
    field_cache = build_field_evaluation_cache(direct_mesh, rule; symmetry_mode=:ground)
    mirrored_field = evaluate_galerkin_field_cpu(
        [SVector{3,T}(0.01, 0.6, 0.02), SVector{3,T}(0.01, -0.6, 0.02)],
        direct_mesh,
        pressure,
        neumann,
        k,
        field_cache,
    )
    @test mirrored_field[1] ≈ mirrored_field[2] rtol=T(2e-6) atol=T(2e-7)

    if cuda_available()
        cuda_regular = build_cuda_regular_assembly_cache(direct_mesh, rule)
        cuda_singular = BeatEngineCore.build_cuda_singular_correction_cache(
            direct_singular,
            direct_p1,
            direct_dp0,
        )
        cuda_image_singular = build_cuda_image_singular_correction_cache(
            direct_mesh,
            direct_p1,
            direct_dp0,
            3,
            eachindex(direct_mesh.faces),
            :ground,
        )
        cuda_half_space = assemble_regular_galerkin_operators(
            direct_mesh, direct_p1, direct_dp0, k, rule;
            skip_singular=false,
            device_cache=cuda_regular,
            singular_cache=direct_singular,
            device_singular_cache=cuda_singular,
            device_image_singular_cache=cuda_image_singular,
            symmetry_mode=:ground,
        )
        @test Array(cuda_half_space.single_layer) ≈ half_space.single_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_half_space.double_layer) ≈ half_space.double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_half_space.adjoint_double_layer) ≈ half_space.adjoint_double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_half_space.hypersingular) ≈ half_space.hypersingular rtol=T(5e-3) atol=T(5e-3)

        cuda_field_cache = build_cuda_field_evaluation_cache(field_cache)
        cuda_field = evaluate_galerkin_field_cuda(
            [SVector{3,T}(0.01, 0.6, 0.02), SVector{3,T}(0.01, -0.6, 0.02)],
            direct_mesh,
            pressure,
            neumann,
            k,
            cuda_field_cache,
        )
        @test cuda_field ≈ mirrored_field rtol=T(5e-4) atol=T(5e-6)
        release_operator_storage!(cuda_half_space)
        release_cuda_image_singular_correction_cache!(cuda_image_singular)
    end
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
    combined_image_corrected = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, base_rule;
        backend=:cpu, skip_singular=false, singular_cache=singular_cache,
        near_correction_cache=near_cache,
        image_near_correction_cache=image_cache,
        symmetry_mode=:ground,
    )
    @test combined_image_corrected.near_pair_count == 3
    @test all(isfinite, real.(combined_image_corrected.single_layer))

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
        cuda_image_near = build_cuda_near_correction_cache(image_cache, p1, dp0)
        cuda_image_singular = build_cuda_image_singular_correction_cache(
            mesh, p1, dp0, 2, eachindex(mesh.faces), :ground,
        )
        cuda_combined = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, base_rule;
            skip_singular=false,
            device_cache=cuda_regular,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular,
            device_image_singular_cache=cuda_image_singular,
            near_correction_cache=near_cache,
            device_near_correction_cache=cuda_near,
            image_near_correction_cache=image_cache,
            device_image_near_correction_cache=cuda_image_near,
            symmetry_mode=:ground,
        )
        @test cuda_combined.near_pair_count == combined_image_corrected.near_pair_count
        @test Array(cuda_combined.single_layer) ≈ combined_image_corrected.single_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_combined.double_layer) ≈ combined_image_corrected.double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_combined.adjoint_double_layer) ≈ combined_image_corrected.adjoint_double_layer rtol=T(5e-3) atol=T(5e-5)
        @test Array(cuda_combined.hypersingular) ≈ combined_image_corrected.hypersingular rtol=T(5e-3) atol=T(5e-3)

        ground_identity_p1_p1 = assemble_l2_identity_matrix(
            mesh, p1, dp0, base_rule, :p1, :p1; symmetry_mode=:ground,
        )
        ground_identity_p1_dp0 = assemble_l2_identity_matrix(
            mesh, p1, dp0, base_rule, :p1, :dp0; symmetry_mode=:ground,
        )
        ground_identity_cache = build_cuda_burton_miller_identity_cache(
            ground_identity_p1_p1, ground_identity_p1_dp0, T,
        )
        direct_q = Complex{T}[Complex{T}(0, 1), Complex{T}(0.25, -0.5)]
        d_direct_q = CUDA_MODULE.CuArray(direct_q)
        direct_coupling = Complex{T}(0, 1) / k
        expected_direct_lhs = (
            Complex{T}(0.5) .* ground_identity_cache.identity_p1_p1 .-
            cuda_combined.double_layer .+
            direct_coupling .* cuda_combined.hypersingular
        )
        expected_direct_rhs = (
            -cuda_combined.single_layer .-
            direct_coupling .* (
                cuda_combined.adjoint_double_layer .+
                Complex{T}(0.5) .* ground_identity_cache.identity_p1_dp0
            )
        ) * d_direct_q
        corrected_direct_system = assemble_burton_miller_neumann_system_cuda(
            mesh,
            p1,
            dp0,
            d_direct_q,
            k,
            base_rule;
            device_cache=cuda_regular,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular,
            device_image_singular_cache=cuda_image_singular,
            near_correction_cache=near_cache,
            device_near_correction_cache=cuda_near,
            image_near_correction_cache=image_cache,
            device_image_near_correction_cache=cuda_image_near,
            symmetry_mode=:ground,
        )
        @test corrected_direct_system.near_pair_count == combined_image_corrected.near_pair_count
        @test Array(corrected_direct_system.matrix) ≈ Array(expected_direct_lhs) rtol=T(5e-3) atol=T(5e-4)
        @test Array(corrected_direct_system.rhs) ≈ Array(expected_direct_rhs) rtol=T(5e-3) atol=T(5e-4)
        release_burton_miller_system_cuda!(corrected_direct_system)
        CUDA_MODULE.unsafe_free!(expected_direct_lhs)
        CUDA_MODULE.unsafe_free!(expected_direct_rhs)
        CUDA_MODULE.unsafe_free!(d_direct_q)
        release_cuda_burton_miller_identity_cache!(ground_identity_cache)
        release_operator_storage!(cuda_combined)
        release_cuda_image_singular_correction_cache!(cuda_image_near)
        release_cuda_image_singular_correction_cache!(cuda_image_singular)
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
        d_expected_lhs = (
            ComplexF32(0.5) .* identity_cache.identity_p1_p1 .-
            operators.double_layer .+
            coupling .* operators.hypersingular
        )
        d_matrix_free_rhs = BeatEngineCore._cuda_burton_miller_rhs(operators, identity_cache, d_q_neumann, coupling)
        direct_timing = Dict{String,Float64}()
        direct_system = assemble_burton_miller_neumann_system_cuda(
            mesh,
            p1,
            dp0,
            d_q_neumann,
            k,
            rule;
            device_cache=cuda_cache,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular_cache,
            timing=direct_timing,
        )
        @test direct_system.assembly_mode == :direct_burton_miller
        @test Array(direct_system.matrix) ≈ Array(d_expected_lhs) rtol=2f-5 atol=2f-6
        @test Array(direct_system.rhs) ≈ Array(d_expected_rhs) rtol=2f-5 atol=2f-6
        @test haskey(direct_timing, "direct_system_regular")
        d_direct_rhs_only = assemble_burton_miller_rhs_cuda(
            mesh,
            p1,
            dp0,
            d_q_neumann,
            k,
            rule;
            device_cache=cuda_cache,
            singular_cache=singular_cache,
            device_singular_cache=cuda_singular_cache,
        )
        @test Array(d_direct_rhs_only) ≈ Array(d_expected_rhs) rtol=2f-5 atol=2f-6
        CUDA_MODULE.unsafe_free!(d_direct_rhs_only)
        direct_pressure = solve_burton_miller_system_cuda!(direct_system)
        @test Array(d_matrix_free_rhs) ≈ Array(d_expected_rhs) rtol=2f-5
        CUDA_MODULE.unsafe_free!(d_q_neumann)
        CUDA_MODULE.unsafe_free!(d_expected_lhs)
        CUDA_MODULE.unsafe_free!(d_expected_rhs)
        CUDA_MODULE.unsafe_free!(d_matrix_free_rhs)

        pressure = solve_burton_miller_neumann(operators, identity_cache, q_neumann, k)
        release_cuda_burton_miller_identity_cache!(identity_cache)

        @test length(pressure) == p1.global_dof_count
        @test direct_pressure ≈ pressure rtol=2f-4 atol=2f-5
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

        symmetry_identity_p1_p1 = assemble_l2_identity_matrix(
            symmetry_mesh, symmetry_p1, symmetry_dp0, rule, :p1, :p1; symmetry_mode=:xy,
        )
        symmetry_identity_p1_dp0 = assemble_l2_identity_matrix(
            symmetry_mesh, symmetry_p1, symmetry_dp0, rule, :p1, :dp0; symmetry_mode=:xy,
        )
        symmetry_identity_cache = build_cuda_burton_miller_identity_cache(
            symmetry_identity_p1_p1, symmetry_identity_p1_dp0, Float32,
        )
        symmetry_q = zeros(ComplexF32, symmetry_dp0.global_dof_count)
        symmetry_q[1] = ComplexF32(0, 1)
        d_symmetry_q = CUDA_MODULE.CuArray(symmetry_q)
        symmetry_expected_lhs = (
            ComplexF32(0.5) .* symmetry_identity_cache.identity_p1_p1 .-
            symmetry_operators.double_layer .+
            coupling .* symmetry_operators.hypersingular
        )
        symmetry_expected_rhs = (
            -symmetry_operators.single_layer .-
            coupling .* (
                symmetry_operators.adjoint_double_layer .+
                ComplexF32(0.5) .* symmetry_identity_cache.identity_p1_dp0
            )
        ) * d_symmetry_q
        symmetry_direct_system = assemble_burton_miller_neumann_system_cuda(
            symmetry_mesh,
            symmetry_p1,
            symmetry_dp0,
            d_symmetry_q,
            k,
            rule;
            device_cache=symmetry_cuda_cache,
            singular_cache=symmetry_singular_cache,
            device_singular_cache=symmetry_cuda_singular_cache,
            device_image_singular_cache=image_cache,
            symmetry_mode=:xy,
        )
        @test symmetry_direct_system.image_singular_pairs == image_cache.pair_count
        @test Array(symmetry_direct_system.matrix) ≈ Array(symmetry_expected_lhs) rtol=5f-4 atol=5f-5
        @test Array(symmetry_direct_system.rhs) ≈ Array(symmetry_expected_rhs) rtol=5f-4 atol=5f-5
        symmetry_rhs_only = assemble_burton_miller_rhs_cuda(
            symmetry_mesh,
            symmetry_p1,
            symmetry_dp0,
            d_symmetry_q,
            k,
            rule;
            device_cache=symmetry_cuda_cache,
            singular_cache=symmetry_singular_cache,
            device_singular_cache=symmetry_cuda_singular_cache,
            device_image_singular_cache=image_cache,
            symmetry_mode=:xy,
        )
        @test Array(symmetry_rhs_only) ≈ Array(symmetry_expected_rhs) rtol=5f-4 atol=5f-5
        CUDA_MODULE.unsafe_free!(symmetry_rhs_only)
        release_burton_miller_system_cuda!(symmetry_direct_system)
        CUDA_MODULE.unsafe_free!(symmetry_expected_lhs)
        CUDA_MODULE.unsafe_free!(symmetry_expected_rhs)
        CUDA_MODULE.unsafe_free!(d_symmetry_q)
        release_cuda_burton_miller_identity_cache!(symmetry_identity_cache)
        release_operator_storage!(symmetry_operators)
        release_cuda_image_singular_correction_cache!(image_cache)
    end
end
