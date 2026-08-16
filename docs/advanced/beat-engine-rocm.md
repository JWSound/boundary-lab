# BEAT Engine ROCm development

The ROCm backend supports exterior Burton-Miller BEM solves, including X and XY
symmetry, with GPU-resident operator assembly and field evaluation:

- regular Galerkin quadrature is evaluated by native ROCm entry-owned kernels;
- Duffy singular quadrature is evaluated once into compact per-pair blocks and
  gathered into the dense operators by race-free entry-owned kernels;
- all four dense operators are allocated and assembled in `ROCArray` storage;
- the Burton-Miller right-hand side uses rocBLAS;
- the dense complex solve uses rocSOLVER;
- exterior field source weighting and observation integration use native ROCm kernels.

Results identify native assembly as `rocm_native_entry_owned`. Set
`BLAB_ROCM_ASSEMBLY_MODE=host_staged` to use the original CPU-assembly/upload path
as a diagnostic fallback. The default workgroup size is 64 on RDNA2; set
`BLAB_ROCM_KERNEL_GROUPSIZE` to `64`, `128`, or `256` for hardware-specific tuning.
Coupled FEM-BEM remains a later milestone.

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

The script independently assembles BEAT CPU and native ROCm operators, compares
all four Galerkin operators, solves the same boundary condition on CPU and
rocSOLVER, and compares boundary pressure, residual, and exterior field pressure.
It exits with an error when a comparison exceeds its tolerance.

To validate only native regular-pair kernels, excluding Duffy singular pairs, run
`validate_rocm_native_regular.jl` from the same directory.

Set `BLAB_VALIDATE_REGULAR_ORDER` and `BLAB_VALIDATE_SINGULAR_ORDER` to validate
production quadrature, for example q4/s4. Run `validate_rocm_symmetry.jl` for the
half-mesh X and quarter-mesh XY fixtures.

## Warm-worker benchmark

`benchmark_rocm.jl` builds geometry, singular, identity, and field caches once,
warms the compiled kernels, and then reports steady-state medians. This matches
the reuse boundary of Boundary Lab's persistent Julia worker more closely than
restarting Julia for every frequency.

```powershell
julia --project=src/blab/solvers/julia_rocm `
  src/blab/solvers/julia_local/scripts/benchmark_rocm.jl `
  --mesh src/blab/solvers/julia_local/test_meshes/sample_detailed.msh `
  --quadrature-order 4 --singular-order 4 --eval-points 144 `
  --warmups 2 --repetitions 5
```

Benchmark JSON is written beneath `src/blab/solvers/julia_local/results/`, which
is intentionally ignored by Git.
