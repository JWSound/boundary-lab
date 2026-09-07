# Required CPU numerical extraction gate; no Boundary Lab imports or assets.
using Test, StaticArrays, LinearAlgebra

# This entrypoint deliberately runs the full references, regardless of ambient
# opt-in settings. Accelerator qualification has separate hardware gates.
ENV["BLAB_RUN_COUPLED_REFERENCE"] = "1"
ENV["BLAB_RUN_COUPLED_CUDA"] = "0"
ENV["BLAB_RUN_COUPLED_ROCM"] = "0"
ENV["BLAB_COUPLED_QUADRATURE_ORDER"] = "1"
ENV["BLAB_COUPLED_SINGULAR_ORDER"] = "1"
cuda_available() = false
rocm_available() = false

include(joinpath(@__DIR__, "fixture_integrity_tests.jl"))
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "analytical_exterior_tests.jl"))
include(joinpath(@__DIR__, "coupled_solver_tests.jl"))
include(joinpath(@__DIR__, "coupled_condensed_tests.jl"))
