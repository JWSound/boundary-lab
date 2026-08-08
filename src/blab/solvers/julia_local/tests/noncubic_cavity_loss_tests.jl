using Test, LinearAlgebra, SparseArrays, Random

const NONCUBIC_FIXTURE_ROOT = normpath(
    joinpath(@__DIR__, "..", "..", "..", "..", "..", "tests", "fixtures", "noncubic_cavity"),
)
const NONCUBIC_DIMENSIONS_M = (0.470, 0.330, 0.220)
const NONCUBIC_SOUND_SPEED_M_PER_S = 343.0
const NONCUBIC_DENSITY_KG_PER_M3 = 1.21
const NONCUBIC_AXIS_FIXTURES = (
    (
        axis=:x,
        axis_index=1,
        file="noncubic_driven_x_center.msh",
        centroid=[-0.235, 0.0, 0.110],
        mode_indices=(1, 0, 0),
    ),
    (
        axis=:y,
        axis_index=2,
        file="noncubic_driven_y_center.msh",
        centroid=[0.0, -0.165, 0.110],
        mode_indices=(0, 1, 0),
    ),
    (
        axis=:z,
        axis_index=3,
        file="noncubic_driven_z_center.msh",
        centroid=[0.0, 0.0, 0.0],
        mode_indices=(0, 0, 1),
    ),
)
const NONCUBIC_RESOLUTIONS = (
    (target_size_mm=20, suffix="_h20"),
    (target_size_mm=14, suffix=""),
    (target_size_mm=10, suffix="_h10"),
)

function resolved_fixture_file(spec, resolution)
    return replace(spec.file, ".msh" => "$(resolution.suffix).msh")
end

function analytic_rectangular_modes(dimensions, sound_speed; count)
    candidates = [
        (
            indices=(nx, ny, nz),
            frequency=sound_speed / 2 * sqrt(
                (nx / dimensions[1])^2 +
                (ny / dimensions[2])^2 +
                (nz / dimensions[3])^2,
            ),
        )
        for nx in 0:5, ny in 0:5, nz in 0:5
        if nx + ny + nz > 0
    ]
    sort!(candidates; by=candidate -> candidate.frequency)
    return candidates[1:count]
end

function rectangular_cosine_mode(mesh, indices)
    minimum_corner = reduce((left, right) -> min.(left, right), mesh.vertices)
    maximum_corner = reduce((left, right) -> max.(left, right), mesh.vertices)
    dimensions = maximum_corner - minimum_corner
    return [
        prod(
            cos(indices[axis] * pi * (vertex[axis] - minimum_corner[axis]) / dimensions[axis])
            for axis in 1:3
        )
        for vertex in mesh.vertices
    ]
end

function rayleigh_eigenvalue(stiffness, mass, vector)
    return real(dot(vector, stiffness * vector) / dot(vector, mass * vector))
end

function mass_orthonormalize(values, mass)
    weighted = mass * values
    gram = Symmetric(Matrix(transpose(values) * weighted))
    factorization = cholesky(gram)
    return values / factorization.U
end

function lowest_generalized_modes(
    mesh,
    stiffness,
    mass,
    analytic_modes;
    oversampling=6,
    iterations=10,
)
    mode_count = length(analytic_modes)
    block_size = mode_count + oversampling + 1
    Random.seed!(20260803)
    vectors = randn(size(stiffness, 1), block_size)
    vectors[:, 1] .= 1.0
    for (index, mode) in enumerate(analytic_modes)
        vectors[:, index + 1] .= rectangular_cosine_mode(mesh, mode.indices)
    end
    vectors = mass_orthonormalize(vectors, mass)
    shifted_factorization = cholesky(Symmetric(stiffness + mass))
    eigenvalues = zeros(block_size)
    residuals = fill(Inf, block_size)
    for _ in 1:iterations
        vectors = shifted_factorization \ (mass * vectors)
        vectors = mass_orthonormalize(vectors, mass)
        projected = Symmetric(Matrix(transpose(vectors) * (stiffness * vectors)))
        decomposition = eigen(projected)
        eigenvalues = decomposition.values
        vectors = vectors * decomposition.vectors
        for index in eachindex(eigenvalues)
            stiffness_vector = stiffness * vectors[:, index]
            mass_vector = mass * vectors[:, index]
            residuals[index] = norm(stiffness_vector - eigenvalues[index] * mass_vector) / max(
                norm(stiffness_vector),
                eigenvalues[index] * norm(mass_vector),
                eps(),
            )
        end
    end
    mode_range = 2:(mode_count + 1) # Exclude the rigid-cavity constant mode.
    return eigenvalues[mode_range], vectors[:, mode_range], residuals[mode_range]
end

function modal_assurance(reference, candidate, mass)
    return abs(dot(reference, mass * candidate))^2 / (
        real(dot(reference, mass * reference)) * real(dot(candidate, mass * candidate))
    )
end

function boundary_patch_metrics(mesh, boundary_tag)
    face_indices = findall(==(boundary_tag), mesh.boundary_physical_tags)
    areas = [
        norm(
            cross(
                mesh.vertices[face[2]] - mesh.vertices[face[1]],
                mesh.vertices[face[3]] - mesh.vertices[face[1]],
            ),
        ) / 2
        for face in mesh.boundary_faces[face_indices]
    ]
    centers = [
        sum(mesh.vertices[vertex] for vertex in face) / 3
        for face in mesh.boundary_faces[face_indices]
    ]
    centroid = sum(areas[index] * centers[index] for index in eachindex(areas)) / sum(areas)
    return sum(areas), centroid
end

function refine_target_mode(stiffness, mass, initial_vector; iterations=3)
    initial_value = rayleigh_eigenvalue(stiffness, mass, initial_vector)
    factorization = lu(stiffness - initial_value * 1.001 * mass)
    vector = initial_vector / sqrt(real(dot(initial_vector, mass * initial_vector)))
    for _ in 1:iterations
        vector = factorization \ (mass * vector)
        vector ./= sqrt(real(dot(vector, mass * vector)))
    end
    value = rayleigh_eigenvalue(stiffness, mass, vector)
    residual = norm(stiffness * vector - value * (mass * vector)) / norm(stiffness * vector)
    return value, vector, residual
end

function modal_compliance(
    stiffness,
    mass,
    mesh,
    mode,
    radiator_tag,
    frequency_hz,
    bulk_loss_factor,
)
    omega = 2pi * frequency_hz
    wavenumber = omega / NONCUBIC_SOUND_SPEED_M_PER_S
    system = assemble_fem_dynamic_stiffness(
        stiffness,
        mass,
        wavenumber;
        bulk_loss_factor=bulk_loss_factor,
    )
    rhs = assemble_prescribed_velocity_load(
        mesh,
        radiator_tag,
        NONCUBIC_DENSITY_KG_PER_M3,
        omega,
        ComplexF64(1, 0),
    )
    pressure = system \ rhs
    compliance = abs(dot(mode, mass * pressure)) / omega
    return compliance, pressure, rhs, system
end

@testset "noncubic rectangular FEM cavity" begin
    undriven_mesh = load_gmsh41_volume(
        joinpath(NONCUBIC_FIXTURE_ROOT, "noncubic.msh"),
        0.001,
    )
    minimum_corner = reduce((left, right) -> min.(left, right), undriven_mesh.vertices)
    maximum_corner = reduce((left, right) -> max.(left, right), undriven_mesh.vertices)

    @test maximum_corner - minimum_corner ≈ collect(NONCUBIC_DIMENSIONS_M) atol=1e-12
    @test physical_tag(undriven_mesh, 3, "Body1") == 1
    @test physical_tag(undriven_mesh, 2, "Body1_boundary") == 2

    stiffness, mass = assemble_p1_fem_matrices(undriven_mesh)
    analytic_modes = analytic_rectangular_modes(
        NONCUBIC_DIMENSIONS_M,
        NONCUBIC_SOUND_SPEED_M_PER_S;
        count=12,
    )
    rayleigh_frequencies = Float64[]
    for analytic_mode in analytic_modes
        mode = rectangular_cosine_mode(undriven_mesh, analytic_mode.indices)
        value = rayleigh_eigenvalue(stiffness, mass, mode)
        frequency = NONCUBIC_SOUND_SPEED_M_PER_S * sqrt(value) / (2pi)
        push!(rayleigh_frequencies, frequency)
        @test frequency ≈ analytic_mode.frequency rtol=0.01
    end
    @info "Noncubic analytic/Rayleigh modes" analytic_hz=[mode.frequency for mode in analytic_modes] rayleigh_hz=rayleigh_frequencies
end

@testset "noncubic centered source-patch fixtures" begin
    topology_reports = NamedTuple[]
    for resolution in NONCUBIC_RESOLUTIONS
        for spec in NONCUBIC_AXIS_FIXTURES
            file = resolved_fixture_file(spec, resolution)
            mesh = load_gmsh41_volume(joinpath(NONCUBIC_FIXTURE_ROOT, file), 0.001)
            radiator_tag = physical_tag(mesh, 2, "Radiator")
            @test physical_tag(mesh, 3, "Body1") == 1
            @test radiator_tag == 2
            @test physical_tag(mesh, 2, "Body1_boundary") == 3
            area, centroid = boundary_patch_metrics(mesh, radiator_tag)
            @test area ≈ 0.008 atol=1e-12
            @test centroid ≈ spec.centroid atol=1e-12

            unit_load = assemble_prescribed_velocity_load(
                mesh,
                radiator_tag,
                1.0,
                1.0,
                ComplexF64(1, 0),
            )
            overlaps = [
                abs(dot(rectangular_cosine_mode(mesh, indices), unit_load))
                for indices in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            ]
            @test overlaps[spec.axis_index] ≈ 0.008 rtol=1e-12
            @test maximum(overlaps[index] for index in 1:3 if index != spec.axis_index) < 5e-7
            push!(
                topology_reports,
                (
                    target_size_mm=resolution.target_size_mm,
                    axis=spec.axis,
                    vertices=length(mesh.vertices),
                    tetrahedra=length(mesh.tetrahedra),
                    overlaps=overlaps,
                ),
            )
        end
    end
    @info "Noncubic centered-patch topology/overlap checks" topology_reports
end

@testset "noncubic sparse-spectrum mesh convergence" begin
    analytic_modes = analytic_rectangular_modes(
        NONCUBIC_DIMENSIONS_M,
        NONCUBIC_SOUND_SPEED_M_PER_S;
        count=12,
    )
    frequency_sets = Vector{Float64}[]
    spectrum_reports = NamedTuple[]
    x_fixture = only(spec for spec in NONCUBIC_AXIS_FIXTURES if spec.axis == :x)
    previous_vertex_count = 0
    previous_tetrahedron_count = 0
    for resolution in NONCUBIC_RESOLUTIONS
        file = resolved_fixture_file(x_fixture, resolution)
        mesh = load_gmsh41_volume(joinpath(NONCUBIC_FIXTURE_ROOT, file), 0.001)
        @test length(mesh.vertices) > previous_vertex_count
        @test length(mesh.tetrahedra) > previous_tetrahedron_count
        previous_vertex_count = length(mesh.vertices)
        previous_tetrahedron_count = length(mesh.tetrahedra)

        stiffness, mass = assemble_p1_fem_matrices(mesh)
        eigenvalues, modes, residuals = lowest_generalized_modes(
            mesh,
            stiffness,
            mass,
            analytic_modes,
        )
        frequencies = NONCUBIC_SOUND_SPEED_M_PER_S .* sqrt.(eigenvalues) ./ (2pi)
        assurances = [
            modal_assurance(
                rectangular_cosine_mode(mesh, analytic_modes[index].indices),
                modes[:, index],
                mass,
            )
            for index in eachindex(analytic_modes)
        ]
        analytic_frequencies = [mode.frequency for mode in analytic_modes]
        relative_errors = (frequencies .- analytic_frequencies) ./ analytic_frequencies
        @test all(relative_errors .> 0)
        @test maximum(relative_errors) < 0.011
        @test minimum(assurances) > 0.99985
        @test maximum(residuals) < 1e-4
        push!(frequency_sets, frequencies)
        push!(
            spectrum_reports,
            (
                target_size_mm=resolution.target_size_mm,
                vertices=length(mesh.vertices),
                tetrahedra=length(mesh.tetrahedra),
                frequencies_hz=frequencies,
                relative_errors=relative_errors,
                modal_assurance=assurances,
                residuals=residuals,
            ),
        )
    end
    for mode_index in eachindex(analytic_modes)
        @test frequency_sets[1][mode_index] > frequency_sets[2][mode_index]
        @test frequency_sets[2][mode_index] > frequency_sets[3][mode_index]
    end
    @info "Noncubic sparse-spectrum convergence" analytic_hz=[mode.frequency for mode in analytic_modes] spectrum_reports
end

@testset "noncubic axial modes with passive bulk loss" begin
    loss_factors = [0.002, 0.005, 0.01, 0.02]
    modal_reports = NamedTuple[]
    for spec in NONCUBIC_AXIS_FIXTURES
        mesh = load_gmsh41_volume(joinpath(NONCUBIC_FIXTURE_ROOT, spec.file), 0.001)
        radiator_tag = physical_tag(mesh, 2, "Radiator")
        analytic_frequency_hz = NONCUBIC_SOUND_SPEED_M_PER_S /
                                (2 * NONCUBIC_DIMENSIONS_M[spec.axis_index])
        stiffness, mass = if spec.axis == :x
            probe = solve_prescribed_velocity_interior(
                mesh,
                analytic_frequency_hz,
                NONCUBIC_SOUND_SPEED_M_PER_S,
                NONCUBIC_DENSITY_KG_PER_M3,
                radiator_tag;
                bulk_loss_factor=0.01,
            )
            @test probe.bulk_loss_factor == 0.01
            @test probe.relative_residual < 1e-9
            probe.stiffness, probe.mass
        else
            assemble_p1_fem_matrices(mesh)
        end

        initial_mode = rectangular_cosine_mode(mesh, spec.mode_indices)
        eigenvalue, mode, modal_residual = refine_target_mode(stiffness, mass, initial_mode)
        modal_frequency_hz = NONCUBIC_SOUND_SPEED_M_PER_S * sqrt(eigenvalue) / (2pi)
        mode_assurance = modal_assurance(initial_mode, mode, mass)
        @test modal_frequency_hz ≈ analytic_frequency_hz rtol=0.003
        @test modal_residual < 1e-8
        @test mode_assurance > 0.999

        wavenumber = 2pi * modal_frequency_hz / NONCUBIC_SOUND_SPEED_M_PER_S
        if spec.axis == :x
            lossless = assemble_fem_dynamic_stiffness(stiffness, mass, wavenumber)
            legacy_lossless = ComplexF64.(stiffness) - wavenumber^2 .* mass
            @test lossless == legacy_lossless
            @test_throws ErrorException assemble_fem_dynamic_stiffness(
                stiffness,
                mass,
                wavenumber;
                bulk_loss_factor=-0.001,
            )
        end

        scaled_compliances = Float64[]
        for loss_factor in loss_factors
            compliance, pressure, rhs, system = modal_compliance(
                stiffness,
                mass,
                mesh,
                mode,
                radiator_tag,
                modal_frequency_hz,
                loss_factor,
            )
            push!(scaled_compliances, loss_factor * compliance)
            dissipative_quadratic_form = imag(dot(pressure, system * pressure))
            expected_dissipation = -loss_factor * wavenumber^2 * real(dot(pressure, mass * pressure))
            @test dissipative_quadratic_form < 0
            @test dissipative_quadratic_form ≈ expected_dissipation rtol=1e-9
            @test imag(dot(pressure, rhs)) ≈ expected_dissipation rtol=1e-8
        end
        @test maximum(scaled_compliances) / minimum(scaled_compliances) < 1.0001

        loss_factor = 0.01
        peak_frequency_hz = modal_frequency_hz / sqrt(1 + loss_factor^2)
        low_frequency_hz = modal_frequency_hz * sqrt((1 - loss_factor) / (1 + loss_factor^2))
        high_frequency_hz = modal_frequency_hz * sqrt((1 + loss_factor) / (1 + loss_factor^2))
        peak_compliance = first(modal_compliance(
            stiffness,
            mass,
            mesh,
            mode,
            radiator_tag,
            peak_frequency_hz,
            loss_factor,
        ))
        low_compliance = first(modal_compliance(
            stiffness,
            mass,
            mesh,
            mode,
            radiator_tag,
            low_frequency_hz,
            loss_factor,
        ))
        high_compliance = first(modal_compliance(
            stiffness,
            mass,
            mesh,
            mode,
            radiator_tag,
            high_frequency_hz,
            loss_factor,
        ))
        @test low_compliance / peak_compliance ≈ inv(sqrt(2)) rtol=0.002
        @test high_compliance / peak_compliance ≈ inv(sqrt(2)) rtol=0.002

        measured_q = peak_frequency_hz / (high_frequency_hz - low_frequency_hz)
        @test measured_q * loss_factor ≈ 1.0 rtol=0.001
        @test abs(peak_frequency_hz - modal_frequency_hz) / modal_frequency_hz < loss_factor^2
        push!(
            modal_reports,
            (
                axis=spec.axis,
                analytic_frequency_hz=analytic_frequency_hz,
                modal_frequency_hz=modal_frequency_hz,
                modal_assurance=mode_assurance,
                measured_q=measured_q,
                scaled_compliances=scaled_compliances,
            ),
        )
    end
    @info "Noncubic axial-mode loss/Q checks" loss_factors modal_reports
end
