"""
Measure P1 FEM spatial-dispersion error versus nominal elements per wavelength
on the noncubic rectangular-cavity convergence fixtures.

The diagnostic evaluates Rayleigh quotients of exact rigid-wall cosine modes.
It deliberately samples axial, planar-oblique, and fully oblique modes from
about 2 to 10 kHz rather than attempting to enumerate the dense spectrum in
that band.
"""

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
using LinearAlgebra
using Statistics

const DIMS_M = (0.470, 0.330, 0.220)
const SOUND_SPEED_M_S = 343.0
const FIXTURE_DIR = normpath(joinpath(
    @__DIR__, "..", "..", "..", "..", "..",
    "tests", "fixtures", "noncubic_cavity",
))

function cosine_mode(mesh, indices)
    lo = reduce((a, b) -> min.(a, b), mesh.vertices)
    hi = reduce((a, b) -> max.(a, b), mesh.vertices)
    lengths = hi - lo
    return [
        prod(cos(indices[axis] * pi * (vertex[axis] - lo[axis]) / lengths[axis]) for axis in 1:3)
        for vertex in mesh.vertices
    ]
end

function analytic_frequency(indices)
    return SOUND_SPEED_M_S / 2 * sqrt(sum((indices[axis] / DIMS_M[axis])^2 for axis in 1:3))
end

function sampled_modes()
    directions = (
        (name=:x, active=(1,)),
        (name=:y, active=(2,)),
        (name=:z, active=(3,)),
        (name=:xy, active=(1, 2)),
        (name=:xz, active=(1, 3)),
        (name=:yz, active=(2, 3)),
        (name=:xyz, active=(1, 2, 3)),
    )
    cases = NamedTuple[]
    seen = Set{Tuple{Int,Int,Int}}()
    for direction in directions, target_hz in 2000.0:250.0:10000.0
        component_wavenumber = 2 * target_hz / (SOUND_SPEED_M_S * sqrt(length(direction.active)))
        indices = ntuple(
            axis -> axis in direction.active ?
                max(1, round(Int, component_wavenumber * DIMS_M[axis])) : 0,
            3,
        )
        indices in seen && continue
        push!(seen, indices)
        frequency = analytic_frequency(indices)
        1800 <= frequency <= 10200 || continue
        push!(cases, (direction=direction.name, indices=indices, analytic_hz=frequency))
    end
    sort!(cases; by=value -> value.analytic_hz)
    return directions, cases
end

function analyze_resolution(resolution, cases, directions)
    mesh = load_gmsh41_volume(joinpath(FIXTURE_DIR, resolution.file), 0.001)
    stiffness, mass = assemble_p1_fem_matrices(mesh)
    results = NamedTuple[]
    for case in cases
        mode = cosine_mode(mesh, case.indices)
        rayleigh_value = real(dot(mode, stiffness * mode) / dot(mode, mass * mode))
        fem_hz = SOUND_SPEED_M_S * sqrt(rayleigh_value) / (2pi)
        relative_error = (fem_hz - case.analytic_hz) / case.analytic_hz
        ppw = SOUND_SPEED_M_S / (case.analytic_hz * resolution.h_mm / 1000)
        push!(results, merge(case, (; fem_hz, relative_error, ppw)))
    end

    println((
        h_mm=resolution.h_mm,
        nodes=length(mesh.vertices),
        maximum_error=maximum(result.relative_error for result in results),
    ))
    for target_hz in (3000.0, 5000.0, 10000.0)
        nearest = [
            sort(
                [result for result in results if result.direction == direction.name];
                by=result -> abs(result.analytic_hz - target_hz),
            )[1]
            for direction in directions
        ]
        println((
            target_hz,
            ppw=median(result.ppw for result in nearest),
            median_error=median(result.relative_error for result in nearest),
            maximum_error=maximum(result.relative_error for result in nearest),
        ))
    end
    return results
end

directions, cases = sampled_modes()
println((sample_count=length(cases), min_hz=first(cases).analytic_hz, max_hz=last(cases).analytic_hz))

resolutions = (
    (h_mm=20, file="noncubic_driven_x_center_h20.msh"),
    (h_mm=14, file="noncubic_driven_x_center.msh"),
    (h_mm=10, file="noncubic_driven_x_center_h10.msh"),
)
all_results = NamedTuple[]
for resolution in resolutions
    results = analyze_resolution(resolution, cases, directions)
    append!(all_results, [(; h_mm=resolution.h_mm, result...) for result in results])
    flush(stdout)
end

println("Empirical thresholds across all sampled meshes and mode directions:")
for (relative_error, recommended_ppw) in ((0.01, 17), (0.02, 12), (0.05, 8), (0.10, 6))
    maximum_failing_ppw = maximum(
        result.ppw for result in all_results if result.relative_error > relative_error
    )
    println((; relative_error, maximum_failing_ppw, recommended_ppw))
end

println("Error by nominal elements-per-wavelength bin:")
for (lower, upper) in ((0, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 10), (10, 12), (12, 16), (16, Inf))
    subset = [result for result in all_results if lower <= result.ppw < upper]
    isempty(subset) && continue
    errors = [result.relative_error for result in subset]
    println((
        ppw_bin=(lower, upper),
        samples=length(errors),
        median_error=median(errors),
        maximum_error=maximum(errors),
    ))
end
