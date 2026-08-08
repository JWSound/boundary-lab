using Test

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

include(joinpath(@__DIR__, "..", "tests", "coupled_solver_tests.jl"))
