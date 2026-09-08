"""Prepare the pinned engine's CUDA artifacts without requiring a build GPU."""

import subprocess

from beat_engine import engine_paths

project = engine_paths("cuda").project
command = ["julia", f"--project={project}", "--startup-file=no", "-e"]
subprocess.run(command + ['using Pkg; Pkg.instantiate(); using CUDA; CUDA.set_runtime_version!(v"12.8")'], check=True)
subprocess.run(command + ["using CUDA, CUDSS; CUDA.precompile_runtime()"], check=True)
