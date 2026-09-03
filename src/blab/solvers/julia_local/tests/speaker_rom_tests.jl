using Test
using LinearAlgebra

isdefined(Main, :BeatEngineCore) ||
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
isdefined(Main, :BeatEngineCoupled) ||
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
isdefined(Main, :BeatEngineSpeakerRom) ||
    include(joinpath(@__DIR__, "..", "src", "BeatEngineSpeakerRom.jl"))
using .BeatEngineSpeakerRom

@testset "speaker ROM snapshot rank selection" begin
    rank_deficient = ComplexF64[
        3 0 0 0 0
        0 2 0 0 0
        0 0 1 0 0
        0 0 0 1.0e-8 0
    ]
    coefficients, spectrum, effective_rank =
        BeatEngineSpeakerRom._snapshot_coefficients(rank_deficient, 4, Float32)

    @test effective_rank == 3
    @test size(coefficients) == (5, 3)
    @test eltype(coefficients) == ComplexF32
    @test length(spectrum) == 4
    @test spectrum[1] ≈ 9.0
    @test spectrum[3] ≈ 1.0

    full_rank = Matrix{ComplexF64}(I, 4, 4)
    coefficients, _spectrum, effective_rank =
        BeatEngineSpeakerRom._snapshot_coefficients(full_rank, 3, Float64)
    @test effective_rank == 3
    @test size(coefficients) == (4, 3)

    @test_throws ErrorException BeatEngineSpeakerRom._snapshot_coefficients(
        zeros(ComplexF64, 4, 4),
        4,
        Float64,
    )
end

@testset "speaker ROM symmetry sectors and reconstruction" begin
    x_symmetric_points = [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 3.0, 0.0],
        [1.0, 3.0, 0.0],
    ]
    x_maps = BeatEngineSpeakerRom._image_maps(x_symmetric_points, :x)
    x_orbits, x_orbit_sizes = BeatEngineSpeakerRom._orbits(x_maps)
    x_sectors = BeatEngineSpeakerRom._parity_sectors(:x)

    @test length(x_maps) == 2
    @test length(x_sectors) == 2
    @test [sector.name for sector in x_sectors] == ["even_x", "odd_x"]
    @test all(length(orbit) == 2 for orbit in x_orbits)
    @test all(==(2), x_orbit_sizes)
    @test_throws ErrorException BeatEngineSpeakerRom._image_maps(x_symmetric_points, :xy)

    values = reshape(ComplexF64.(1:6), 6, 1)
    reconstructed = zeros(ComplexF64, size(values))
    for sector in x_sectors
        projected = BeatEngineSpeakerRom._parity_project(values, x_maps, sector.image_signs)
        compact = BeatEngineSpeakerRom._compact_parity_values(
            projected,
            x_orbits,
            sector.image_signs,
        )
        reconstructed .+= BeatEngineSpeakerRom._reconstruct_parity_values(
            compact,
            x_orbits,
            sector.image_signs,
            length(x_symmetric_points),
        )
    end
    @test reconstructed ≈ values

    off_maps = BeatEngineSpeakerRom._image_maps(x_symmetric_points, :off)
    off_orbits, _ = BeatEngineSpeakerRom._orbits(off_maps)
    @test length(BeatEngineSpeakerRom._parity_sectors(:off)) == 1
    @test all(length(orbit) == 1 for orbit in off_orbits)
end
