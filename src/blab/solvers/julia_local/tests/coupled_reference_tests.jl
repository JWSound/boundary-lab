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
        solution = solve_coupled_reference(
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
    end
else
    @info "Set BLAB_RUN_COUPLED_REFERENCE=1 to run the full dense FEM-BEM fixture validation."
end
