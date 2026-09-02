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
