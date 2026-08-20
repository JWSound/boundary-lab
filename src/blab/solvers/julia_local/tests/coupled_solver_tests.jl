include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
using LinearAlgebra, SparseArrays, StaticArrays

const COUPLED_FIXTURE_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..", "tests", "fixtures"))
const COUPLED_QUADRATURE_ORDER = parse(Int, get(ENV, "BLAB_COUPLED_QUADRATURE_ORDER", "1"))
const COUPLED_SINGULAR_ORDER = parse(Int, get(ENV, "BLAB_COUPLED_SINGULAR_ORDER", "1"))

@testset "FEM consistent/lumped mass blending" begin
    consistent = sparse(Float64[2 1; 1 2] ./ 6)
    lumped = blend_fem_mass_matrix(consistent, 0.0)
    blended = blend_fem_mass_matrix(consistent, 0.5)

    @test blend_fem_mass_matrix(consistent, 1.0) === consistent
    @test Matrix(lumped) ≈ [0.5 0.0; 0.0 0.5]
    @test Matrix(blended) ≈ 0.5 .* (Matrix(consistent) + Matrix(lumped))
    @test vec(sum(blended; dims=2)) ≈ vec(sum(consistent; dims=2))
    @test_throws ErrorException blend_fem_mass_matrix(consistent, 1.1)
end

@testset "affine quadratic tetrahedron FEM assembly" begin
    vertices = SVector{3,Float64}[
        SVector(0.0, 0.0, 0.0),
        SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0),
        SVector(0.0, 0.0, 1.0),
        SVector(0.5, 0.0, 0.0),
        SVector(0.5, 0.5, 0.0),
        SVector(0.0, 0.5, 0.0),
        SVector(0.0, 0.0, 0.5),
        SVector(0.0, 0.5, 0.5),
        SVector(0.5, 0.0, 0.5),
    ]
    mesh = VolumeMesh{Float64}(
        vertices,
        [(1, 2, 3, 4)],
        [1],
        [(1, 2, 3)],
        [2],
        Dict{Tuple{Int,Int},String}(),
        [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)],
        [(1, 2, 3, 5, 6, 7)],
    )

    stiffness, mass = assemble_p2_fem_matrices(mesh)
    boundary_mass = assemble_boundary_mass_matrix(mesh, [1], collect(1:10))
    load = assemble_prescribed_velocity_load(mesh, 2, 1.2, 100.0, 1.0 + 0.0im)

    @test size(stiffness) == (10, 10)
    @test Matrix(stiffness) ≈ transpose(Matrix(stiffness)) atol=1e-12
    @test Matrix(mass) ≈ transpose(Matrix(mass)) atol=1e-12
    @test sum(mass) ≈ 1 / 6 atol=1e-12
    @test norm(stiffness * ones(10)) < 1e-12
    @test sum(boundary_mass) ≈ 0.5 atol=1e-12
    @test sum(load) ≈ 1im * 1.2 * 100.0 * 0.5 atol=1e-12
end

@testset "blocked exact FEM Schur complement" begin
    interior_system = sparse(ComplexF64[
        4.0 + 0.1im 1.0 0.0
        1.0 3.0 + 0.2im 0.5
        0.0 0.5 2.0 + 0.1im
    ])
    interior_to_retained = sparse(ComplexF64[
        1.0 0.0
        0.25 0.5
        0.0 1.0
    ])
    retained_to_interior = sparse(transpose(interior_to_retained))
    retained_system = ComplexF64[
        3.0 + 0.1im 0.2
        0.2 2.5 + 0.1im
    ]
    factorization = lu(interior_system)
    reference = retained_system -
                retained_to_interior * (factorization \ Matrix(interior_to_retained))

    result = BeatEngineCoupled._blocked_umfpack_schur_complement(
        factorization,
        interior_to_retained,
        retained_to_interior,
        retained_system;
        block_size=1,
    )

    @test result.schur ≈ reference rtol=1e-14 atol=1e-14
    @test result.block_size == 1
    @test 1 <= result.thread_count <= min(Threads.nthreads(), 2)
end

@testset "disconnected BEM mesh aggregation" begin
    first = BoundaryMesh(
        SVector{3,Float64}[
            SVector(0.0, 0.0, 0.0),
            SVector(1.0, 0.0, 0.0),
            SVector(0.0, 1.0, 0.0),
        ],
        [(1, 2, 3)],
        [7],
    )
    second = BoundaryMesh(
        SVector{3,Float64}[
            SVector(0.0, 0.0, 2.0),
            SVector(0.0, 1.0, 2.0),
            SVector(1.0, 0.0, 2.0),
        ],
        [(1, 2, 3)],
        [7],
    )

    combined = combine_boundary_meshes([first, second])

    @test combined.vertex_offsets == [0, 3]
    @test combined.face_offsets == [0, 1]
    @test combined.mesh.faces == [(1, 2, 3), (4, 5, 6)]
    @test combined.mesh.physical_tags == [1, 2]
    @test combined.physical_tag_maps == [Dict(7 => 1), Dict(7 => 2)]
    @test build_p1_space(combined.mesh).global_dof_count == 6

    local_interface = ConformingInterfaceMap([1, 3], [2, 4], [5], [6], [-1])
    mapped_interface = offset_interface_map(
        local_interface;
        fem_vertex_offset=10,
        fem_face_offset=20,
        bem_vertex_offset=30,
        bem_face_offset=40,
    )
    @test mapped_interface.fem_vertex_indices == [11, 13]
    @test mapped_interface.fem_to_bem_vertex_indices == [32, 34]
    @test mapped_interface.fem_face_indices == [25]
    @test mapped_interface.bem_face_indices == [46]
end

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

@testset "per-region FEM loss and Miki wall impedance" begin
    mesh = load_gmsh41_volume(joinpath(COUPLED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    stiffness, mass = assemble_p1_fem_matrices(mesh)
    wavenumber = 2pi * 200.0 / 343.0
    scalar_loss = assemble_fem_dynamic_stiffness(
        stiffness,
        mass,
        wavenumber;
        bulk_loss_factor=0.02,
    )
    weighted_loss = assemble_fem_dynamic_stiffness(
        stiffness,
        mass,
        wavenumber;
        bulk_loss_mass=0.02 .* mass,
    )
    @test weighted_loss ≈ scalar_loss rtol=1e-12

    admittance = miki_rigid_backed_surface_admittance(100.0, 343.0, 1.21, 0.03, 5000.0)
    @test real(admittance) > 0.0
    @test imag(admittance) < 0.0
    air_impedance = 1.21 * 343.0
    surface_impedance = inv(admittance)
    reflection = (surface_impedance - air_impedance) / (surface_impedance + air_impedance)
    absorption = 1.0 - abs2(reflection)
    @test 0.01 < absorption < 0.05
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

@testset "empty FEM-BEM interface operators" begin
    fem_mesh = load_gmsh41_volume(joinpath(COUPLED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    bem_mesh = load_gmsh22_with_tags(joinpath(COUPLED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
    interface_map = ConformingInterfaceMap(Int[], Int[], Int[], Int[], Int[])

    operators = assemble_interface_operators(fem_mesh, bem_mesh, interface_map)

    @test size(operators.fem_load) == (length(fem_mesh.vertices), 0)
    @test size(operators.bem_flux) == (length(bem_mesh.faces), 0)
    @test size(operators.fem_trace) == (0, length(fem_mesh.vertices))
    @test size(operators.bem_trace) == (0, length(bem_mesh.vertices))
end

@testset "mixed FEM and BEM prescribed-velocity excitation" begin
    fem_mesh = VolumeMesh(
        SVector{3,Float64}[
            SVector(0.0, 0.0, 0.0),
            SVector(1.0, 0.0, 0.0),
            SVector(0.0, 1.0, 0.0),
            SVector(0.0, 0.0, 1.0),
        ],
        [(1, 2, 3, 4)],
        [1],
        [(1, 2, 3)],
        [9],
        Dict((3, 1) => "Volume", (2, 9) => "FEM source"),
    )
    bem_mesh = BoundaryMesh(
        SVector{3,Float64}[
            SVector(0.0, 0.0, 2.0),
            SVector(1.0, 0.0, 2.0),
            SVector(0.0, 1.0, 2.0),
        ],
        [(1, 2, 3)],
        [11],
    )
    interface_map = ConformingInterfaceMap(Int[], Int[], Int[], Int[], Int[])
    interface_operators = InterfaceOperators{Float64}(
        spzeros(Float64, 4, 0),
        spzeros(Float64, 1, 0),
        spzeros(Float64, 0, 4),
        spzeros(Float64, 0, 3),
    )
    system = (
        fem_mesh=fem_mesh,
        bem_mesh=bem_mesh,
        interface_map=interface_map,
        interface_operators=interface_operators,
        transducers=ElectrodynamicTransducer{Float64}[],
        transducer_operators=(bem_normal_velocity=spzeros(Float64, 1, 0),),
        density=1.0,
        omega=2.0,
        factorization=lu(Matrix{ComplexF64}(I, 7, 7)),
        formulation=:monolithic,
        fem_range=1:4,
        bem_range=5:7,
        flux_range=8:7,
        mechanical_range=1:0,
        electrical_range=1:0,
        prescribed_bem_rhs=reshape(ComplexF64[4, 5, 6], 3, 1),
        prescribed_bem_neumann=reshape(ComplexF64[3im], 1, 1),
        linear_backend=:cpu,
        validation_diagnostics=false,
        scalar_type=Float64,
    )
    excitation = (
        kind=:normal_velocity,
        fem_boundary_tags=[9],
        fem_boundary_weights=[0.5],
        bem_source_index=1,
        transducer_index=0,
        amplitude=ComplexF64(2, 0),
    )

    solution = only(solve_coupled_excitations(system, [excitation]))

    @test solution.fem_pressure[1:3] ≈ fill(ComplexF64(0, 1 / 3), 3)
    @test solution.fem_pressure[4] == 0
    @test solution.bem_pressure == ComplexF64[8, 10, 12]
    @test solution.bem_neumann == ComplexF64[6im]
    @test solution.pressure_continuity_error == 0
    @test solution.flux_conservation_error == 0
end

@testset "rigid-piston electrodynamic surface operators" begin
    fem_mesh = load_gmsh41_volume(joinpath(COUPLED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    bem_mesh = load_gmsh22_with_tags(joinpath(COUPLED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
    radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
    motion_axis = SVector(0.0, 0.0, 1.0)
    transducer = ElectrodynamicTransducer{Float64}(
        "component:test",
        [radiator_tag],
        [1.0],
        [1],
        [-1.0],
        motion_axis,
        2.0,
        1,
        6.0,
        0.0005,
        7.0,
        0.015,
        0.0005,
        1.0,
    )

    operators = assemble_transducer_operators(fem_mesh, bem_mesh, [transducer])

    fem_normals = BeatEngineCoupled._outward_boundary_normals(fem_mesh)
    fem_projected_area = sum(
        dot(fem_normals[index], motion_axis) *
        BeatEngineCoupled._triangle_area(fem_mesh.vertices, fem_mesh.boundary_faces[index])
        for index in eachindex(fem_mesh.boundary_faces)
        if fem_mesh.boundary_physical_tags[index] == radiator_tag
    )
    bem_projected_area = sum(
        -dot(bem_mesh.normals[index], motion_axis) * bem_mesh.areas[index]
        for index in eachindex(bem_mesh.faces)
        if bem_mesh.physical_tags[index] == 1
    )
    @test size(operators.fem_surface) == (length(fem_mesh.vertices), 1)
    @test size(operators.bem_surface) == (length(bem_mesh.vertices), 1)
    @test size(operators.bem_normal_velocity) == (length(bem_mesh.faces), 1)
    @test sum(operators.fem_surface[:, 1]) ≈ fem_projected_area
    @test sum(operators.fem_force[:, 1]) ≈ 2 * fem_projected_area
    @test sum(operators.bem_surface[:, 1]) ≈ bem_projected_area
    @test sum(operators.bem_force[:, 1]) ≈ 2 * bem_projected_area
    @test all(
        operators.bem_normal_velocity[index, 1] ≈
        -dot(bem_mesh.normals[index], motion_axis)
        for index in eachindex(bem_mesh.faces)
        if bem_mesh.physical_tags[index] == 1
    )
    @test all(
        operators.bem_normal_velocity[index, 1] == 0.0
        for index in eachindex(bem_mesh.faces)
        if bem_mesh.physical_tags[index] != 1
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
        bulk_loss_factor = 0.01
        prescribed_bem_normal_velocity = sparse(
            findall(==(1), bem_mesh.physical_tags),
            ones(Int, count(==(1), bem_mesh.physical_tags)),
            ones(Float64, count(==(1), bem_mesh.physical_tags)),
            length(bem_mesh.faces),
            1,
        )
        coupled_system = build_coupled_system(
            fem_mesh,
            bem_mesh,
            interface_map,
            500.0,
            343.0,
            1.21;
            quadrature_order=COUPLED_QUADRATURE_ORDER,
            singular_order=COUPLED_SINGULAR_ORDER,
            bulk_loss_factor=bulk_loss_factor,
            prescribed_bem_normal_velocity=prescribed_bem_normal_velocity,
        )
        expected_fem_system = assemble_fem_dynamic_stiffness(
            coupled_system.cache.stiffness,
            coupled_system.cache.mass,
            coupled_system.wavenumber;
            bulk_loss_factor=bulk_loss_factor,
        )
        @test coupled_system.bulk_loss_factor == bulk_loss_factor
        @test isapprox(
            coupled_system.coupled[coupled_system.fem_range, coupled_system.fem_range],
            Matrix(expected_fem_system),
        )
        solution = solve_coupled_system(
            coupled_system,
            physical_tag(fem_mesh, 2, "Radiator"),
        )
        @info "Coupled reference diagnostics" relative_residual=solution.relative_residual pressure_continuity_error=solution.pressure_continuity_error flux_conservation_error=solution.flux_conservation_error all_bem_replay_error=solution.all_bem_replay_error
        @test solution.relative_residual < 1e-8
        @test solution.pressure_continuity_error < 1e-8
        @test solution.flux_conservation_error < 1e-10
        @test solution.all_bem_replay_error < 1e-8

        bem_solution = only(
            solve_coupled_excitations(
                coupled_system,
                [(
                    kind=:normal_velocity,
                    fem_boundary_tags=Int[],
                    fem_boundary_weights=Float64[],
                    bem_source_index=1,
                    transducer_index=0,
                    amplitude=ComplexF64(1, 0),
                )],
            ),
        )
        expected_bem_neumann = ComplexF64(0, 1.21 * 2pi * 500.0) .* vec(
            prescribed_bem_normal_velocity,
        )
        expected_bem_neumann .+= ComplexF64.(
            coupled_system.interface_operators.bem_flux,
        ) * bem_solution.interface_flux
        @test bem_solution.bem_neumann ≈ expected_bem_neumann
        @test bem_solution.relative_residual < 1e-8
        @test bem_solution.pressure_continuity_error < 1e-8
        @test bem_solution.flux_conservation_error < 1e-10
        @test bem_solution.all_bem_replay_error < 1e-8

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
            bulk_loss_factor=Float32(bulk_loss_factor),
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
        prescribed_bem_normal_velocity = sparse(
            findall(==(1), bem_mesh.physical_tags),
            ones(Int, count(==(1), bem_mesh.physical_tags)),
            ones(Float32, count(==(1), bem_mesh.physical_tags)),
            length(bem_mesh.faces),
            1,
        )
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
            bulk_loss_factor=0.01f0,
            prescribed_bem_normal_velocity=prescribed_bem_normal_velocity,
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
            bulk_loss_factor=0.01f0,
            prescribed_bem_normal_velocity=prescribed_bem_normal_velocity,
        )
        condensed_system = build_coupled_system(
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
            static_condensation=true,
            bulk_loss_factor=0.01f0,
            prescribed_bem_normal_velocity=prescribed_bem_normal_velocity,
        )
        try
            cpu_solution = solve_coupled_system(cpu_system, radiator_tag)
            cuda_solutions = solve_coupled_systems(
                cuda_system,
                [radiator_tag, radiator_tag];
                radiator_velocities=ComplexF32[1, 0.5],
            )
            cuda_solution = cuda_solutions[1]
            condensed_solutions = solve_coupled_systems(
                condensed_system,
                [radiator_tag, radiator_tag];
                radiator_velocities=ComplexF32[1, 0.5],
            )
            condensed_solution = condensed_solutions[1]
            bem_excitation = (
                kind=:normal_velocity,
                fem_boundary_tags=Int[],
                fem_boundary_weights=Float32[],
                bem_source_index=1,
                transducer_index=0,
                amplitude=ComplexF32(1, 0),
            )
            cpu_bem_solution = only(solve_coupled_excitations(cpu_system, [bem_excitation]))
            cuda_bem_solution = only(solve_coupled_excitations(cuda_system, [bem_excitation]))
            condensed_bem_solution = only(
                solve_coupled_excitations(condensed_system, [bem_excitation]),
            )
            relative_error(reference, candidate) = norm(candidate - reference) / norm(reference)

            @test cpu_system.linear_backend == :cpu
            @test cuda_system.linear_backend == :cuda
            @test cpu_system.bulk_loss_factor == 0.01f0
            @test cuda_system.bulk_loss_factor == 0.01f0
            @test condensed_system.bulk_loss_factor == 0.01f0
            @test relative_error(cpu_solution.fem_pressure, cuda_solution.fem_pressure) < 5e-4
            @test relative_error(cpu_solution.bem_pressure, cuda_solution.bem_pressure) < 5e-4
            @test relative_error(cpu_solution.interface_flux, cuda_solution.interface_flux) < 5e-4
            @test relative_error(
                0.5f0 .* cuda_solution.bem_pressure,
                cuda_solutions[2].bem_pressure,
            ) < 1e-5
            @test cuda_solution.pressure_continuity_error < 1e-4
            @test cuda_solution.flux_conservation_error < 1e-4
            @test condensed_system.formulation == :fem_interface_condensed
            @test condensed_system.solved_system_order < condensed_system.full_system_order
            @test relative_error(
                cuda_solution.fem_pressure,
                condensed_solution.fem_pressure,
            ) < 1e-3
            @test relative_error(
                cuda_solution.bem_pressure,
                condensed_solution.bem_pressure,
            ) < 1e-3
            @test relative_error(
                cuda_solution.interface_flux,
                condensed_solution.interface_flux,
            ) < 1e-3
            @test relative_error(
                0.5f0 .* condensed_solution.bem_pressure,
                condensed_solutions[2].bem_pressure,
            ) < 1e-5
            @test condensed_solution.pressure_continuity_error < 1e-4
            @test condensed_solution.flux_conservation_error < 1e-4
            @test relative_error(
                cpu_bem_solution.fem_pressure,
                cuda_bem_solution.fem_pressure,
            ) < 5e-4
            @test relative_error(
                cpu_bem_solution.bem_pressure,
                cuda_bem_solution.bem_pressure,
            ) < 5e-4
            @test relative_error(
                cpu_bem_solution.interface_flux,
                cuda_bem_solution.interface_flux,
            ) < 5e-4
            @test relative_error(
                cuda_bem_solution.fem_pressure,
                condensed_bem_solution.fem_pressure,
            ) < 1e-3
            @test relative_error(
                cuda_bem_solution.bem_pressure,
                condensed_bem_solution.bem_pressure,
            ) < 1e-3
            @test relative_error(
                cuda_bem_solution.interface_flux,
                condensed_bem_solution.interface_flux,
            ) < 1e-3
            @test relative_error(
                cpu_bem_solution.bem_neumann,
                cuda_bem_solution.bem_neumann,
            ) < 5e-4
            @test relative_error(
                cuda_bem_solution.bem_neumann,
                condensed_bem_solution.bem_neumann,
            ) < 1e-3
        finally
            release_coupled_system!(cpu_system)
            release_coupled_system!(cuda_system)
            release_coupled_system!(condensed_system)
        end
    end
elseif get(ENV, "BLAB_RUN_COUPLED_CUDA", "0") == "1"
    @test_skip "CUDA unavailable; skipping GPU-resident coupled solve."
else
    @info "Set BLAB_RUN_COUPLED_CUDA=1 to run the GPU-resident coupled solve validation."
end

if get(ENV, "BLAB_RUN_COUPLED_ROCM", "0") == "1" && rocm_available()
    @testset "FP32 ROCm GPU-resident coupled solve" begin
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
        prescribed_bem_normal_velocity = sparse(
            findall(==(1), bem_mesh.physical_tags),
            ones(Int, count(==(1), bem_mesh.physical_tags)),
            ones(Float32, count(==(1), bem_mesh.physical_tags)),
            length(bem_mesh.faces),
            1,
        )
        transducer = ElectrodynamicTransducer{Float32}(
            "component:rocm-test",
            [radiator_tag],
            Float32[1],
            [1],
            Float32[-1],
            SVector(0f0, 0f0, 1f0),
            2f0,
            1,
            6f0,
            0.0005f0,
            7f0,
            0.015f0,
            0.0005f0,
            1f0,
        )
        common_options = (
            quadrature_order=COUPLED_QUADRATURE_ORDER,
            singular_order=COUPLED_SINGULAR_ORDER,
            validation_diagnostics=false,
            bulk_loss_factor=0.01f0,
            prescribed_bem_normal_velocity=prescribed_bem_normal_velocity,
            transducers=[transducer],
        )
        cpu_system = build_coupled_system(
            fem_mesh,
            bem_mesh,
            interface_map,
            Float32(500),
            Float32(343),
            Float32(1.21);
            common_options...,
            bem_backend=:cpu,
        )
        rocm_system = build_coupled_system(
            fem_mesh,
            bem_mesh,
            interface_map,
            Float32(500),
            Float32(343),
            Float32(1.21);
            common_options...,
            bem_backend=:rocm,
            static_condensation=false,
        )
        condensed_system = build_coupled_system(
            fem_mesh,
            bem_mesh,
            interface_map,
            Float32(500),
            Float32(343),
            Float32(1.21);
            common_options...,
            bem_backend=:rocm,
            static_condensation=true,
        )
        try
            cpu_solution = solve_coupled_system(cpu_system, radiator_tag)
            rocm_solutions = solve_coupled_systems(
                rocm_system,
                [radiator_tag, radiator_tag];
                radiator_velocities=ComplexF32[1, 0.5],
            )
            rocm_solution = rocm_solutions[1]
            condensed_solutions = solve_coupled_systems(
                condensed_system,
                [radiator_tag, radiator_tag];
                radiator_velocities=ComplexF32[1, 0.5],
            )
            condensed_solution = condensed_solutions[1]
            bem_excitation = (
                kind=:normal_velocity,
                fem_boundary_tags=Int[],
                fem_boundary_weights=Float32[],
                bem_source_index=1,
                transducer_index=0,
                amplitude=ComplexF32(1, 0),
            )
            cpu_bem_solution = only(solve_coupled_excitations(cpu_system, [bem_excitation]))
            rocm_bem_solution = only(solve_coupled_excitations(rocm_system, [bem_excitation]))
            condensed_bem_solution = only(
                solve_coupled_excitations(condensed_system, [bem_excitation]),
            )
            voltage_excitation = (
                kind=:voltage,
                fem_boundary_tags=Int[],
                fem_boundary_weights=Float32[],
                bem_source_index=0,
                transducer_index=1,
                amplitude=ComplexF32(2.83, 0),
            )
            cpu_voltage_solution = only(
                solve_coupled_excitations(cpu_system, [voltage_excitation]),
            )
            rocm_voltage_solution = only(
                solve_coupled_excitations(rocm_system, [voltage_excitation]),
            )
            condensed_voltage_solution = only(
                solve_coupled_excitations(condensed_system, [voltage_excitation]),
            )
            relative_error(reference, candidate) = norm(candidate - reference) / norm(reference)

            @test cpu_system.linear_backend == :cpu
            @test rocm_system.linear_backend == :rocm
            @test rocm_system.formulation == :monolithic
            @test condensed_system.linear_backend == :rocm
            @test condensed_system.formulation == :fem_interface_condensed
            @test condensed_system.solved_system_order < condensed_system.full_system_order
            @test condensed_system.condensation.schur_block_size > 0
            @test 1 <= condensed_system.condensation.schur_thread_count <= Threads.nthreads()
            @test rocm_system.bulk_loss_factor == 0.01f0
            @test relative_error(cpu_solution.fem_pressure, rocm_solution.fem_pressure) < 5e-4
            @test relative_error(cpu_solution.bem_pressure, rocm_solution.bem_pressure) < 5e-4
            @test relative_error(cpu_solution.interface_flux, rocm_solution.interface_flux) < 5e-4
            @test relative_error(
                0.5f0 .* rocm_solution.bem_pressure,
                rocm_solutions[2].bem_pressure,
            ) < 1e-5
            @test relative_error(
                rocm_solution.fem_pressure,
                condensed_solution.fem_pressure,
            ) < 1e-3
            @test relative_error(
                rocm_solution.bem_pressure,
                condensed_solution.bem_pressure,
            ) < 1e-3
            @test relative_error(
                rocm_solution.interface_flux,
                condensed_solution.interface_flux,
            ) < 1e-3
            @test relative_error(
                0.5f0 .* condensed_solution.bem_pressure,
                condensed_solutions[2].bem_pressure,
            ) < 1e-5
            @test rocm_solution.pressure_continuity_error < 1e-4
            @test rocm_solution.flux_conservation_error < 1e-4
            @test relative_error(
                cpu_bem_solution.fem_pressure,
                rocm_bem_solution.fem_pressure,
            ) < 5e-4
            @test relative_error(
                cpu_bem_solution.bem_pressure,
                rocm_bem_solution.bem_pressure,
            ) < 5e-4
            @test relative_error(
                cpu_bem_solution.interface_flux,
                rocm_bem_solution.interface_flux,
            ) < 5e-4
            @test relative_error(
                cpu_bem_solution.bem_neumann,
                rocm_bem_solution.bem_neumann,
            ) < 5e-4
            @test relative_error(
                rocm_bem_solution.fem_pressure,
                condensed_bem_solution.fem_pressure,
            ) < 1e-3
            @test relative_error(
                rocm_bem_solution.bem_pressure,
                condensed_bem_solution.bem_pressure,
            ) < 1e-3
            @test relative_error(
                rocm_bem_solution.interface_flux,
                condensed_bem_solution.interface_flux,
            ) < 1e-3
            @test relative_error(
                rocm_bem_solution.bem_neumann,
                condensed_bem_solution.bem_neumann,
            ) < 1e-3
            @test relative_error(
                cpu_voltage_solution.fem_pressure,
                rocm_voltage_solution.fem_pressure,
            ) < 5e-4
            @test relative_error(
                cpu_voltage_solution.bem_pressure,
                rocm_voltage_solution.bem_pressure,
            ) < 5e-4
            @test relative_error(
                cpu_voltage_solution.interface_flux,
                rocm_voltage_solution.interface_flux,
            ) < 2e-3 # This FP32 fixture's coupled matrix has cond(A, 1) ~= 3.8e9.
            @test relative_error(
                cpu_voltage_solution.diaphragm_velocity,
                rocm_voltage_solution.diaphragm_velocity,
            ) < 5e-4
            @test relative_error(
                cpu_voltage_solution.voice_coil_current,
                rocm_voltage_solution.voice_coil_current,
            ) < 5e-4
            @test relative_error(
                rocm_voltage_solution.fem_pressure,
                condensed_voltage_solution.fem_pressure,
            ) < 1e-3
            @test relative_error(
                rocm_voltage_solution.bem_pressure,
                condensed_voltage_solution.bem_pressure,
            ) < 1e-3
            @test relative_error(
                rocm_voltage_solution.interface_flux,
                condensed_voltage_solution.interface_flux,
            ) < 2e-3
            @test relative_error(
                rocm_voltage_solution.diaphragm_velocity,
                condensed_voltage_solution.diaphragm_velocity,
            ) < 1e-3
            @test relative_error(
                rocm_voltage_solution.voice_coil_current,
                condensed_voltage_solution.voice_coil_current,
            ) < 1e-3

            field_points = SVector{3,Float32}[
                SVector(0.20f0, 0.13f0, 0.30f0),
                SVector(-0.27f0, 0.08f0, 0.24f0),
            ]
            cpu_field = evaluate_galerkin_field_cpu(
                field_points,
                bem_mesh,
                cpu_solution.bem_pressure,
                cpu_solution.bem_neumann,
                cpu_system.wavenumber,
                cpu_system.field_cache,
            )
            rocm_field = evaluate_galerkin_field_rocm(
                field_points,
                bem_mesh,
                rocm_solution.bem_pressure,
                rocm_solution.bem_neumann,
                rocm_system.wavenumber,
                rocm_system.field_cache,
            )
            @test relative_error(cpu_field, rocm_field) < 5e-4
            condensed_field = evaluate_galerkin_field_rocm(
                field_points,
                bem_mesh,
                condensed_solution.bem_pressure,
                condensed_solution.bem_neumann,
                condensed_system.wavenumber,
                condensed_system.field_cache,
            )
            @test relative_error(rocm_field, condensed_field) < 1e-3
        finally
            release_coupled_system!(cpu_system)
            release_coupled_system!(rocm_system)
            release_coupled_system!(condensed_system)
        end
    end
elseif get(ENV, "BLAB_RUN_COUPLED_ROCM", "0") == "1"
    @test_skip "ROCm/rocBLAS/rocSOLVER unavailable; skipping ROCm coupled solve."
else
    @info "Set BLAB_RUN_COUPLED_ROCM=1 to run the ROCm coupled solve validation."
end
