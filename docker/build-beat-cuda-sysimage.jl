import Pkg

const PROJECT_DIR = abspath(joinpath(@__DIR__, "..", "src", "blab", "solvers", "julia_cuda"))
const SYSIMAGE_PATH = get(ENV, "BLAB_JULIA_SYSIMAGE", "/app/blab-beat-cuda.so")
const PRECOMPILE_FILE = abspath(joinpath(@__DIR__, "precompile-beat-cuda.jl"))
const CPU_TARGET = get(ENV, "BLAB_JULIA_CPU_TARGET", "generic,+aes")

Pkg.activate(mktempdir())
Pkg.add("PackageCompiler")

using PackageCompiler

PackageCompiler.create_sysimage(
    ["CUDA", "JSON", "StaticArrays"];
    project=PROJECT_DIR,
    sysimage_path=SYSIMAGE_PATH,
    precompile_execution_file=PRECOMPILE_FILE,
    cpu_target=CPU_TARGET,
)
