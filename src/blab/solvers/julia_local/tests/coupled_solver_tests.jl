include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
using LinearAlgebra, SparseArrays, StaticArrays

const COUPLED_FIXTURE_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..", "tests", "fixtures"))
const COUPLED_QUADRATURE_ORDER = parse(Int, get(ENV, "BLAB_COUPLED_QUADRATURE_ORDER", "1"))
const COUPLED_SINGULAR_ORDER = parse(Int, get(ENV, "BLAB_COUPLED_SINGULAR_ORDER", "1"))

function structured_unit_cube(divisions::Int)
    divisions > 0 || error("divisions must be positive")
    points_per_axis = divisions + 1
    vertex_index(i, j, k) = 1 + i + points_per_axis * (j + points_per_axis * k)
    vertices = SVector{3,Float64}[
        SVector{3,Float64}(i / divisions, j / divisions, k / divisions)
        for k in 0:divisions
        for j in 0:divisions
        for i in 0:divisions
    ]
    tetrahedra = NTuple{4,Int}[]
    for k in 0:(divisions - 1), j in 0:(divisions - 1), i in 0:(divisions - 1)
        v000 = vertex_index(i, j, k)
        v100 = vertex_index(i + 1, j, k)
        v010 = vertex_index(i, j + 1, k)
        v110 = vertex_index(i + 1, j + 1, k)
        v001 = vertex_index(i, j, k + 1)
        v101 = vertex_index(i + 1, j, k + 1)
        v011 = vertex_index(i, j + 1, k + 1)
        v111 = vertex_index(i + 1, j + 1, k + 1)
        append!(
            tetrahedra,
            (
                (v000, v100, v110, v111),
                (v000, v110, v010, v111),
                (v000, v010, v011, v111),
                (v000, v011, v001, v111),
                (v000, v001, v101, v111),
                (v000, v101, v100, v111),
            ),
        )
    end
    return VolumeMesh(
        vertices,
        tetrahedra,
        ones(Int, length(tetrahedra)),
        NTuple{3,Int}[],
        Int[],
        Dict((3, 1) => "Volume"),
    )
end

@testset "double-precision P1 FEM reference" begin
    fem_mesh = load_gmsh41_volume(joinpath(COUPLED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    @test length(fem_mesh.vertices) == 842
    @test length(fem_mesh.tetrahedra) == 2925
    @test physical_tag(fem_mesh, 3, "Volume") == 1
    @test physical_tag(fem_mesh, 2, "Radiator") == 2
    @test physical_tag(fem_mesh, 2, "Interface") == 3

    stiffness, mass = assemble_p1_fem_matrices(fem_mesh)
    @test eltype(stiffness) == Float64
    @test issymmetric(stiffness)
    @test issymmetric(mass)
    @test norm(stiffness * ones(Float64, size(stiffness, 1))) <= 1e-11
    @test minimum(diag(mass)) > 0

    solution = solve_prescribed_velocity_interior(
        fem_mesh,
        500.0,
        343.0,
        1.21,
        physical_tag(fem_mesh, 2, "Radiator"),
    )
    @test solution.relative_residual < 1e-10
    @test all(isfinite, real.(solution.pressure))
    @test all(isfinite, imag.(solution.pressure))
end

@testset "single-precision P1 FEM assembly" begin
    fem_mesh = load_gmsh41_volume(joinpath(COUPLED_FIXTURE_ROOT, "femvolume.msh"), Float32(0.001))
    stiffness, mass = assemble_p1_fem_matrices(fem_mesh)
    @test eltype(stiffness) == Float32
    @test eltype(mass) == Float32
    @test all(isfinite, nonzeros(stiffness))
    @test all(isfinite, nonzeros(mass))
end

@testset "FEM volume-group restriction compacts active topology" begin
    vertices = SVector{3,Float64}[
        SVector(0.0, 0.0, 0.0),
        SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0),
        SVector(0.0, 0.0, 1.0),
        SVector(2.0, 0.0, 0.0),
        SVector(3.0, 0.0, 0.0),
        SVector(2.0, 1.0, 0.0),
        SVector(2.0, 0.0, 1.0),
    ]
    mesh = VolumeMesh(
        vertices,
        [(1, 2, 3, 4), (5, 6, 7, 8)],
        [10, 20],
        [
            (1, 2, 3),
            (1, 2, 4),
            (1, 3, 4),
            (2, 3, 4),
            (5, 6, 7),
            (5, 6, 8),
            (5, 7, 8),
            (6, 7, 8),
        ],
        [101, 101, 101, 101, 202, 202, 202, 202],
        Dict((3, 10) => "First", (3, 20) => "Second"),
    )

    selection = restrict_volume_mesh(mesh, [20])

    @test length(selection.mesh.vertices) == 4
    @test selection.mesh.tetrahedra == [(1, 2, 3, 4)]
    @test selection.mesh.tetra_physical_tags == [20]
    @test selection.mesh.boundary_physical_tags == fill(202, 4)
    @test selection.vertex_index_map == Dict(5 => 1, 6 => 2, 7 => 3, 8 => 4)
    @test selection.boundary_face_index_map == Dict(5 => 1, 6 => 2, 7 => 3, 8 => 4)
    @test_throws ErrorException restrict_volume_mesh(mesh, [99])
end

@testset "sealed unit-cube modes" begin
    cube = structured_unit_cube(4)
    modes = sealed_cavity_modes(cube, 343.0; count=4)
    analytic_first = 343.0 / 2
    @test length(modes) == 4
    @test modes[1] ≈ analytic_first rtol=0.08
    @test maximum(modes[1:3]) / minimum(modes[1:3]) < 1.08
end

@testset "conforming FEM-BEM interface operators" begin
    fem_mesh = load_gmsh41_volume(joinpath(COUPLED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    bem_mesh = load_gmsh22_with_tags(joinpath(COUPLED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
    interface_map = build_conforming_interface_map(
        fem_mesh,
        bem_mesh,
        physical_tag(fem_mesh, 2, "Interface"),
        2,
    )
    @test length(interface_map.fem_vertex_indices) == 106
    @test length(interface_map.fem_face_indices) == 180
    @test Set(interface_map.normal_sign) ⊆ Set((-1, 1))

    operators = assemble_interface_operators(fem_mesh, bem_mesh, interface_map)
    @test size(operators.fem_load) == (842, 106)
    @test size(operators.bem_flux) == (2424, 106)
    @test size(operators.fem_trace) == (106, 842)
    @test size(operators.bem_trace) == (106, 1214)
    @test all(
        isapprox(sum(operators.bem_flux[face_index, :]), interface_map.normal_sign[local_index])
        for (local_index, face_index) in enumerate(interface_map.bem_face_indices)
    )
end

if get(ENV, "BLAB_RUN_COUPLED_REFERENCE", "0") == "1"
    @testset "direct coupled FEM-BEM reference" begin
        fem_mesh = load_gmsh41_volume(joinpath(COUPLED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
        bem_mesh = load_gmsh22_with_tags(joinpath(COUPLED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
        interface_map = build_conforming_interface_map(
            fem_mesh,
            bem_mesh,
            physical_tag(fem_mesh, 2, "Interface"),
            2,
        )
        solution = solve_coupled(
            fem_mesh,
            bem_mesh,
            interface_map,
            500.0,
            343.0,
            1.21,
            physical_tag(fem_mesh, 2, "Radiator");
            quadrature_order=COUPLED_QUADRATURE_ORDER,
            singular_order=COUPLED_SINGULAR_ORDER,
        )
        @info "Coupled reference diagnostics" relative_residual=solution.relative_residual pressure_continuity_error=solution.pressure_continuity_error flux_conservation_error=solution.flux_conservation_error all_bem_replay_error=solution.all_bem_replay_error
        @test solution.relative_residual < 1e-8
        @test solution.pressure_continuity_error < 1e-8
        @test solution.flux_conservation_error < 1e-10
        @test solution.all_bem_replay_error < 1e-8

        fem_mesh32 = load_gmsh41_volume(
            joinpath(COUPLED_FIXTURE_ROOT, "femvolume.msh"),
            Float32(0.001),
        )
        bem_mesh32 = load_gmsh22_with_tags(
            joinpath(COUPLED_FIXTURE_ROOT, "exterior_conforming.msh"),
            Float32(0.001),
        )
        interface_map32 = build_conforming_interface_map(
            fem_mesh32,
            bem_mesh32,
            physical_tag(fem_mesh32, 2, "Interface"),
            2,
        )
        solution32 = solve_coupled(
            fem_mesh32,
            bem_mesh32,
            interface_map32,
            Float32(500.0),
            Float32(343.0),
            Float32(1.21),
            physical_tag(fem_mesh32, 2, "Radiator");
            quadrature_order=COUPLED_QUADRATURE_ORDER,
            singular_order=COUPLED_SINGULAR_ORDER,
        )
        relative_error(reference, candidate) = norm(
            ComplexF64.(candidate) .- ComplexF64.(reference),
        ) / norm(ComplexF64.(reference))
        @test relative_error(solution.fem_pressure, solution32.fem_pressure) < 1e-4
        @test relative_error(solution.bem_pressure, solution32.bem_pressure) < 1e-4
        @test relative_error(solution.interface_flux, solution32.interface_flux) < 1e-4
        @test solution32.relative_residual < 1e-3
        @test solution32.pressure_continuity_error < 1e-5
        @test solution32.flux_conservation_error < 1e-5
    end
else
    @info "Set BLAB_RUN_COUPLED_REFERENCE=1 to run the full dense FEM-BEM fixture validation."
end

if get(ENV, "BLAB_RUN_COUPLED_CUDA", "0") == "1" && cuda_available()
    @testset "FP32 GPU-resident coupled solve" begin
        fem_mesh = load_gmsh41_volume(
            joinpath(COUPLED_FIXTURE_ROOT, "femvolume.msh"),
            Float32(0.001),
        )
        bem_mesh = load_gmsh22_with_tags(
            joinpath(COUPLED_FIXTURE_ROOT, "exterior_conforming.msh"),
            Float32(0.001),
        )
        interface_map = build_conforming_interface_map(
            fem_mesh,
            bem_mesh,
            physical_tag(fem_mesh, 2, "Interface"),
            2,
        )
        radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
        cpu_system = build_coupled_system(
            fem_mesh,
            bem_mesh,
            interface_map,
            Float32(500),
            Float32(343),
            Float32(1.21);
            quadrature_order=COUPLED_QUADRATURE_ORDER,
            singular_order=COUPLED_SINGULAR_ORDER,
            validation_diagnostics=false,
            bem_backend=:cpu,
        )
        cuda_system = build_coupled_system(
            fem_mesh,
            bem_mesh,
            interface_map,
            Float32(500),
            Float32(343),
            Float32(1.21);
            quadrature_order=COUPLED_QUADRATURE_ORDER,
            singular_order=COUPLED_SINGULAR_ORDER,
            validation_diagnostics=false,
            bem_backend=:cuda,
        )
        try
            cpu_solution = solve_coupled_system(cpu_system, radiator_tag)
            cuda_solutions = solve_coupled_systems(
                cuda_system,
                [radiator_tag, radiator_tag];
                radiator_velocities=ComplexF32[1, 0.5],
            )
            cuda_solution = cuda_solutions[1]
            relative_error(reference, candidate) = norm(candidate - reference) / norm(reference)

            @test cpu_system.linear_backend == :cpu
            @test cuda_system.linear_backend == :cuda
            @test relative_error(cpu_solution.fem_pressure, cuda_solution.fem_pressure) < 5e-4
            @test relative_error(cpu_solution.bem_pressure, cuda_solution.bem_pressure) < 5e-4
            @test relative_error(cpu_solution.interface_flux, cuda_solution.interface_flux) < 5e-4
            @test relative_error(
                0.5f0 .* cuda_solution.bem_pressure,
                cuda_solutions[2].bem_pressure,
            ) < 1e-5
            @test cuda_solution.pressure_continuity_error < 1e-4
            @test cuda_solution.flux_conservation_error < 1e-4
        finally
            release_coupled_system!(cpu_system)
            release_coupled_system!(cuda_system)
        end
    end
elseif get(ENV, "BLAB_RUN_COUPLED_CUDA", "0") == "1"
    @test_skip "CUDA unavailable; skipping GPU-resident coupled solve."
else
    @info "Set BLAB_RUN_COUPLED_CUDA=1 to run the GPU-resident coupled solve validation."
end
