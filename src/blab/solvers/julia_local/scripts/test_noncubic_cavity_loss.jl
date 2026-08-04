include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
include(joinpath(@__DIR__, "..", "tests", "noncubic_cavity_loss_tests.jl"))
