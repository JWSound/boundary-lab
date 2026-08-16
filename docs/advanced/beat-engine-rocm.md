# BEAT Engine ROCm development

The first ROCm milestone supports non-symmetric exterior Burton–Miller BEM solves.
It intentionally uses a correctness-first pipeline:

- regular and singular Galerkin assembly use the existing BEAT CPU implementation;
- dense operators are uploaded to `ROCArray` storage;
- the Burton–Miller right-hand side uses rocBLAS;
- the dense complex solve uses rocSOLVER;
- exterior field reconstruction currently uses the BEAT CPU implementation.

Results identify this implementation as `rocm_host_staged_cpu_assembly`. Symmetry,
native ROCm assembly and field kernels, coupled FEM-BEM, and performance tuning are
later milestones.

## Julia environment

Instantiate the dedicated environment once:

```powershell
julia --project=src/blab/solvers/julia_rocm -e 'using Pkg; Pkg.instantiate()'
```

Set `BLAB_ROCM_PATH` to the SDK root before launching Boundary Lab. The subprocess
adapter maps it to `ROCM_PATH`, `ROCM_HOME`, and `HIP_PATH`, and prepends the SDK's
`bin` directory to `PATH` for ROCm workers.

```powershell
$env:BLAB_ROCM_PATH = 'E:\ROCm-TheRock\10.1.0a20260816-gfx103X'
```

The SDK must make `AMDGPU.functional()`, `AMDGPU.functional(:rocblas)`, and
`AMDGPU.functional(:rocsolver)` return `true`.

## Exterior fixture validation

Run the CPU-versus-ROCm validation against the complete non-symmetry `sample.msh`
fixture from a Julia process configured for the selected ROCm SDK:

```powershell
julia --project=src/blab/solvers/julia_rocm `
  src/blab/solvers/julia_local/scripts/validate_rocm_exterior.jl
```

The script independently assembles BEAT CPU and ROCm-path operators, compares all
four Galerkin operators, solves the same boundary condition on CPU and rocSOLVER,
and compares boundary pressure, residual, and exterior field pressure. It exits
with an error when a comparison exceeds its tolerance.
