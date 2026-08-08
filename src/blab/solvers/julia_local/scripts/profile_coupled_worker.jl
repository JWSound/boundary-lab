"""
Profiler-safe entry point for the coupled Julia worker.

Nsight process-tree injection can alter `ptxas --version` output on Windows,
which prevents CUDA.jl from initializing. Set
`BLAB_CUDA_COMPILER_VERSION_OVERRIDE` to the version reported by the selected
ptxas executable when using this wrapper under a profiler.
"""

import CUDA

compiler_override = strip(get(ENV, "BLAB_CUDA_COMPILER_VERSION_OVERRIDE", ""))
if !isempty(compiler_override)
    compiler_version = VersionNumber(compiler_override)
    Core.eval(CUDA.CUDACore, :(compiler_version() = $compiler_version))
end

include(joinpath(@__DIR__, "..", "coupled_solver.jl"))
