"""
Compare targeted rigid-cavity P1 FEM modes on the curved-interface production
fixture and its independently exported detailed mesh.

Modes are obtained with block shift-invert iteration, deduplicated by
mass-weighted modal assurance, and matched across meshes after nearest-node
transfer from the detailed mesh to the baseline nodes. The nearest-node MAC is
an identification aid; frequency convergence remains the primary metric.
"""

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
using LinearAlgebra
using Random
using SparseArrays
using Statistics

const SOUND_SPEED_M_S = 343.0
const FIXTURE_ROOT = normpath(joinpath(
    @__DIR__, "..", "..", "..", "..", "..", "tests", "fixtures",
))
const MESH_SPECS = (
    (name=:baseline, file="curvedinterfaceFEM.msh"),
    (name=:detailed, file="curvedinterfaceFEM_detailed.msh"),
)
const SURFACE_NAMES = ("HF", "MF", "INTERFACE", "FEMVolume (1) (1)_boundary")
const TARGET_FREQUENCIES_HZ = (
    3250.0, 4000.0, 4750.0, 5500.0, 6250.0, 7000.0,
    7750.0, 8500.0, 9250.0, 9625.0, 10000.0,
)
const MAXIMUM_EIGENPAIR_RESIDUAL = 5e-3
const STATS_ONLY = "--stats-only" in ARGS

function tetrahedron_metrics(mesh)
    local_edges = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    edges = Set{Tuple{Int,Int}}()
    qualities = Float64[]
    volumes = Float64[]
    for tetrahedron in mesh.tetrahedra
        lengths = Float64[]
        for (a, b) in local_edges
            push!(edges, minmax(tetrahedron[a], tetrahedron[b]))
            push!(lengths, norm(mesh.vertices[tetrahedron[a]] - mesh.vertices[tetrahedron[b]]))
        end
        x1 = mesh.vertices[tetrahedron[1]]
        volume = abs(det(hcat(
            mesh.vertices[tetrahedron[2]] - x1,
            mesh.vertices[tetrahedron[3]] - x1,
            mesh.vertices[tetrahedron[4]] - x1,
        ))) / 6
        push!(volumes, volume)
        push!(qualities, 12 * (3 * volume)^(2 / 3) / sum(abs2, lengths))
    end
    edge_lengths = sort([norm(mesh.vertices[a] - mesh.vertices[b]) for (a, b) in edges])
    return (
        volume_m3=sum(volumes),
        edge_mm=(
            minimum=minimum(edge_lengths) * 1000,
            median=median(edge_lengths) * 1000,
            p95=quantile(edge_lengths, 0.95) * 1000,
            maximum=maximum(edge_lengths) * 1000,
        ),
        mean_ratio_quality=(
            minimum=minimum(qualities),
            p01=quantile(qualities, 0.01),
            p05=quantile(qualities, 0.05),
            median=median(qualities),
        ),
    )
end

function surface_metrics(mesh)
    reports = NamedTuple[]
    for name in SURFACE_NAMES
        tag = physical_tag(mesh, 2, name)
        face_indices = findall(==(tag), mesh.boundary_physical_tags)
        area = sum(
            norm(cross(
                mesh.vertices[face[2]] - mesh.vertices[face[1]],
                mesh.vertices[face[3]] - mesh.vertices[face[1]],
            )) / 2
            for face in mesh.boundary_faces[face_indices]
        )
        push!(reports, (; name, tag, faces=length(face_indices), area_m2=area))
    end
    return reports
end

function mesh_report(mesh)
    vertex_matrix = reduce(hcat, mesh.vertices)
    return merge((
        vertices=length(mesh.vertices),
        tetrahedra=length(mesh.tetrahedra),
        boundary_faces=length(mesh.boundary_faces),
        bbox_min_m=Tuple(minimum(vertex_matrix; dims=2)[:]),
        bbox_max_m=Tuple(maximum(vertex_matrix; dims=2)[:]),
    ), tetrahedron_metrics(mesh), (;
        surfaces=surface_metrics(mesh),
    ))
end

function mass_orthonormalize(vectors, mass)
    gram = Symmetric(Matrix(transpose(vectors) * (mass * vectors)))
    factorization = cholesky(gram)
    return vectors / factorization.U
end

function modal_assurance(left, right, mass)
    numerator = abs(dot(left, mass * right))^2
    denominator = real(dot(left, mass * left)) * real(dot(right, mass * right))
    return numerator / denominator
end

function targeted_modes(
    stiffness,
    mass,
    target_frequency_hz;
    block_size=16,
    retained_count=12,
    iterations=10,
    seed=20260804,
)
    target_wavenumber = 2pi * target_frequency_hz / SOUND_SPEED_M_S
    shift = target_wavenumber^2
    Random.seed!(seed + round(Int, target_frequency_hz))
    vectors = mass_orthonormalize(randn(size(stiffness, 1), block_size), mass)
    factorization = lu(stiffness - shift * mass)
    eigenvalues = zeros(block_size)
    residuals = fill(Inf, block_size)
    for _ in 1:iterations
        vectors = factorization \ (mass * vectors)
        vectors = mass_orthonormalize(vectors, mass)
        projected_stiffness = Symmetric(Matrix(transpose(vectors) * (stiffness * vectors)))
        projected_mass = Symmetric(Matrix(transpose(vectors) * (mass * vectors)))
        decomposition = eigen(projected_stiffness, projected_mass)
        order = sortperm(abs.(decomposition.values .- shift))
        eigenvalues = decomposition.values[order]
        vectors = vectors * decomposition.vectors[:, order]
    end
    for index in eachindex(eigenvalues)
        stiffness_vector = stiffness * vectors[:, index]
        mass_vector = mass * vectors[:, index]
        residuals[index] = norm(stiffness_vector - eigenvalues[index] * mass_vector) / max(
            norm(stiffness_vector),
            eigenvalues[index] * norm(mass_vector),
            eps(),
        )
    end
    selected = 1:retained_count
    return (
        eigenvalues=eigenvalues[selected],
        modes=vectors[:, selected],
        residuals=residuals[selected],
    )
end

function deduplicate_modes(stiffness, mass, candidates; frequency_tolerance_hz=20.0)
    ordered = sort(candidates; by=candidate -> candidate.frequency_hz)
    retained = NamedTuple[]
    for candidate in ordered
        duplicate_index = findfirst(eachindex(retained)) do index
            existing = retained[index]
            abs(existing.frequency_hz - candidate.frequency_hz) <= frequency_tolerance_hz &&
                modal_assurance(existing.mode, candidate.mode, mass) > 0.8
        end
        if isnothing(duplicate_index)
            push!(retained, candidate)
        elseif candidate.residual < retained[duplicate_index].residual
            retained[duplicate_index] = candidate
        end
    end
    return retained
end

function compute_modes(mesh, stiffness, mass)
    candidates = NamedTuple[]
    for target_frequency_hz in TARGET_FREQUENCIES_HZ
        @info "Targeted eigensolve" nodes=length(mesh.vertices) target_frequency_hz
        result = targeted_modes(stiffness, mass, target_frequency_hz)
        for index in eachindex(result.eigenvalues)
            eigenvalue = result.eigenvalues[index]
            eigenvalue > 0 || continue
            frequency_hz = SOUND_SPEED_M_S * sqrt(eigenvalue) / (2pi)
            2500 <= frequency_hz <= 10500 || continue
            push!(candidates, (
                frequency_hz=frequency_hz,
                mode=copy(result.modes[:, index]),
                residual=result.residuals[index],
                target_frequency_hz=target_frequency_hz,
            ))
        end
    end
    deduplicated = deduplicate_modes(stiffness, mass, candidates)
    accepted = [mode for mode in deduplicated if mode.residual <= MAXIMUM_EIGENPAIR_RESIDUAL]
    rejected = length(deduplicated) - length(accepted)
    if rejected > 0
        @warn "Rejected unconverged targeted eigenpairs" nodes=length(mesh.vertices) rejected MAXIMUM_EIGENPAIR_RESIDUAL
    end
    return accepted
end

function nearest_vertex_map(source_vertices, target_vertices)
    target_matrix = reduce(hcat, target_vertices)
    minimum_corner = minimum(target_matrix; dims=2)[:]
    maximum_corner = maximum(target_matrix; dims=2)[:]
    cell_size = maximum(maximum_corner - minimum_corner) / cbrt(length(target_vertices))
    buckets = Dict{NTuple{3,Int},Vector{Int}}()
    cell_key(point) = ntuple(axis -> floor(Int, (point[axis] - minimum_corner[axis]) / cell_size), 3)
    for (index, point) in enumerate(target_vertices)
        push!(get!(buckets, cell_key(point), Int[]), index)
    end

    mapping = Vector{Int}(undef, length(source_vertices))
    distances = Vector{Float64}(undef, length(source_vertices))
    for (source_index, point) in enumerate(source_vertices)
        center = cell_key(point)
        best_index = 0
        best_distance = Inf
        for radius in 0:4
            for dx in -radius:radius, dy in -radius:radius, dz in -radius:radius
                max(abs(dx), abs(dy), abs(dz)) == radius || continue
                key = (center[1] + dx, center[2] + dy, center[3] + dz)
                for target_index in get(buckets, key, Int[])
                    distance = norm(point - target_vertices[target_index])
                    if distance < best_distance
                        best_index = target_index
                        best_distance = distance
                    end
                end
            end
            best_index > 0 && best_distance <= radius * cell_size && break
        end
        best_index > 0 || error("No detailed-mesh vertex found near baseline vertex $source_index.")
        mapping[source_index] = best_index
        distances[source_index] = best_distance
    end
    return mapping, distances
end

function surface_participation(mesh, mode)
    energies = Float64[]
    overlaps = Float64[]
    for name in SURFACE_NAMES
        tag = physical_tag(mesh, 2, name)
        energy = 0.0
        overlap = 0.0
        for face_index in findall(==(tag), mesh.boundary_physical_tags)
            face = mesh.boundary_faces[face_index]
            area = norm(cross(
                mesh.vertices[face[2]] - mesh.vertices[face[1]],
                mesh.vertices[face[3]] - mesh.vertices[face[1]],
            )) / 2
            values = mode[collect(face)]
            energy += area / 12 * (
                2sum(abs2, values) +
                2real(values[1] * values[2] + values[1] * values[3] + values[2] * values[3])
            )
            overlap += area * sum(values) / 3
        end
        push!(energies, energy)
        push!(overlaps, abs(overlap))
    end
    energy_sum = sum(energies)
    return (
        energy_fraction=energies ./ energy_sum,
        normalized_overlap=overlaps ./ sqrt.(max.(energies, eps())),
    )
end

function match_modes(baseline, detailed, baseline_mass, detailed_to_baseline_map)
    pairs = NamedTuple[]
    for baseline_index in eachindex(baseline), detailed_index in eachindex(detailed)
        baseline_mode = baseline[baseline_index].mode
        transferred_mode = detailed[detailed_index].mode[detailed_to_baseline_map]
        mac = modal_assurance(baseline_mode, transferred_mode, baseline_mass)
        relative_frequency_delta = abs(
            baseline[baseline_index].frequency_hz - detailed[detailed_index].frequency_hz
        ) / detailed[detailed_index].frequency_hz
        relative_frequency_delta <= 0.08 || continue
        push!(pairs, (; baseline_index, detailed_index, mac, relative_frequency_delta))
    end
    sort!(pairs; by=pair -> (-pair.mac, pair.relative_frequency_delta))
    used_baseline = Set{Int}()
    used_detailed = Set{Int}()
    matches = NamedTuple[]
    for pair in pairs
        pair.baseline_index in used_baseline && continue
        pair.detailed_index in used_detailed && continue
        pair.mac >= 0.5 || continue
        push!(used_baseline, pair.baseline_index)
        push!(used_detailed, pair.detailed_index)
        push!(matches, pair)
    end
    sort!(matches; by=match -> detailed[match.detailed_index].frequency_hz)
    return matches
end

meshes = Dict(
    spec.name => load_gmsh41_volume(joinpath(FIXTURE_ROOT, spec.file), 0.001)
    for spec in MESH_SPECS
)
reports = Dict(name => mesh_report(mesh) for (name, mesh) in meshes)
for spec in MESH_SPECS
    println((; mesh=spec.name, reports[spec.name]...))
end

baseline_report = reports[:baseline]
detailed_report = reports[:detailed]
println((
    geometry_comparison=(
        relative_volume_delta=abs(detailed_report.volume_m3 - baseline_report.volume_m3) /
                              detailed_report.volume_m3,
        bbox_min_delta_m=maximum(abs.(collect(detailed_report.bbox_min_m) .- collect(baseline_report.bbox_min_m))),
        bbox_max_delta_m=maximum(abs.(collect(detailed_report.bbox_max_m) .- collect(baseline_report.bbox_max_m))),
    ),
))
STATS_ONLY && exit()

systems = Dict{Symbol,NamedTuple}()
mode_sets = Dict{Symbol,Vector{NamedTuple}}()
for spec in MESH_SPECS
    mesh = meshes[spec.name]
    @info "Assembling FEM matrices" mesh=spec.name nodes=length(mesh.vertices) tetrahedra=length(mesh.tetrahedra)
    stiffness, mass = assemble_p1_fem_matrices(mesh)
    systems[spec.name] = (; stiffness, mass)
    modes = compute_modes(mesh, stiffness, mass)
    mode_sets[spec.name] = modes
    println((
        mesh=spec.name,
        modes=length(modes),
        frequency_range_hz=(minimum(mode.frequency_hz for mode in modes), maximum(mode.frequency_hz for mode in modes)),
        maximum_residual=maximum(mode.residual for mode in modes),
    ))
end

baseline_mesh = meshes[:baseline]
detailed_mesh = meshes[:detailed]
baseline_to_detailed, transfer_distances = nearest_vertex_map(
    baseline_mesh.vertices,
    detailed_mesh.vertices,
)
println((
    nearest_node_transfer_mm=(
        median=median(transfer_distances) * 1000,
        p95=quantile(transfer_distances, 0.95) * 1000,
        maximum=maximum(transfer_distances) * 1000,
    ),
))

baseline_modes = mode_sets[:baseline]
detailed_modes = mode_sets[:detailed]
matches = match_modes(
    baseline_modes,
    detailed_modes,
    systems[:baseline].mass,
    baseline_to_detailed,
)
println("baseline_hz\tdetailed_hz\tcoarse_shift_pct\tMAC\tparticipation_delta\tcoarse_residual\tfine_residual\tHF_fraction\tMF_fraction\tinterface_fraction")
reports_by_mode = NamedTuple[]
for match in matches
    baseline = baseline_modes[match.baseline_index]
    detailed = detailed_modes[match.detailed_index]
    baseline_participation = surface_participation(baseline_mesh, baseline.mode)
    detailed_participation = surface_participation(detailed_mesh, detailed.mode)
    participation_delta = maximum(abs.(
        baseline_participation.energy_fraction .- detailed_participation.energy_fraction
    ))
    signed_shift = (baseline.frequency_hz - detailed.frequency_hz) / detailed.frequency_hz
    println(join((
        round(baseline.frequency_hz; digits=3),
        round(detailed.frequency_hz; digits=3),
        round(100signed_shift; digits=4),
        round(match.mac; digits=6),
        round(participation_delta; digits=6),
        baseline.residual,
        detailed.residual,
        round(detailed_participation.energy_fraction[1]; digits=6),
        round(detailed_participation.energy_fraction[2]; digits=6),
        round(detailed_participation.energy_fraction[3]; digits=6),
    ), '\t'))
    push!(reports_by_mode, (;
        detailed_frequency_hz=detailed.frequency_hz,
        signed_shift,
        mac=match.mac,
        participation_delta,
    ))
end

matched_baseline_indices = Set(match.baseline_index for match in matches)
matched_detailed_indices = Set(match.detailed_index for match in matches)
println("Unmatched accepted modes (nearest frequency counterpart and best nearby MAC):")
for (label, modes, other_modes, matched_indices, transfer_from_detailed) in (
    (:baseline, baseline_modes, detailed_modes, matched_baseline_indices, false),
    (:detailed, detailed_modes, baseline_modes, matched_detailed_indices, true),
)
    for index in eachindex(modes)
        index in matched_indices && continue
        mode = modes[index]
        nearest_index = argmin(abs(other.frequency_hz - mode.frequency_hz) for other in other_modes)
        nearby = [
            other_index for other_index in eachindex(other_modes)
            if abs(other_modes[other_index].frequency_hz - mode.frequency_hz) / mode.frequency_hz <= 0.03
        ]
        assurances = Float64[]
        for other_index in nearby
            left, right = if transfer_from_detailed
                (other_modes[other_index].mode, mode.mode[baseline_to_detailed])
            else
                (mode.mode, other_modes[other_index].mode[baseline_to_detailed])
            end
            push!(assurances, modal_assurance(left, right, systems[:baseline].mass))
        end
        println((
            mesh=label,
            frequency_hz=mode.frequency_hz,
            residual=mode.residual,
            nearest_other_hz=other_modes[nearest_index].frequency_hz,
            nearest_relative_delta=abs(other_modes[nearest_index].frequency_hz - mode.frequency_hz) / mode.frequency_hz,
            maximum_nearby_mac=maximum(assurances; init=0.0),
        ))
    end
end

for (lower_hz, upper_hz) in ((3000.0, 5000.0), (5000.0, 8000.0), (8000.0, 10500.0))
    selected = [
        report for report in reports_by_mode
        if lower_hz <= report.detailed_frequency_hz < upper_hz
    ]
    isempty(selected) && continue
    println((
        band_hz=(lower_hz, upper_hz),
        matched_modes=length(selected),
        median_absolute_shift=median(abs(report.signed_shift) for report in selected),
        maximum_absolute_shift=maximum(abs(report.signed_shift) for report in selected),
        minimum_mac=minimum(report.mac for report in selected),
        median_participation_delta=median(report.participation_delta for report in selected),
        maximum_participation_delta=maximum(report.participation_delta for report in selected),
    ))
end
