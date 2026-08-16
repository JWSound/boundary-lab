# BEAT Engine ROCm development

The ROCm backend supports exterior Burton-Miller BEM solves, including X and XY
symmetry, with GPU-resident operator assembly and field evaluation:

- regular Galerkin quadrature is evaluated once per element pair by native ROCm
  pair-owned kernels; vertex-disjoint element colors make direct dense-operator
  scatter race-free without atomics;
- Duffy singular quadrature is evaluated once into compact per-pair blocks and
  gathered into the dense operators by race-free entry-owned kernels;
- all four dense operators are allocated and assembled in `ROCArray` storage;
- the Burton-Miller right-hand side uses rocBLAS;
- the dense complex solve uses rocSOLVER;
- exterior field source weighting and observation integration use native ROCm kernels.

Results identify the default native assembly as `rocm_native_colored_pair_owned`.
Set `BLAB_ROCM_REGULAR_KERNEL_MODE=entry_owned` to retain the earlier entry-owned
kernel as a correctness and performance reference. Set
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

For a high-frequency, end-to-end directivity check, run
`scripts/diagnose_rocm_polar.jl`. It compares CPU and ROCm operators, boundary
pressure, integrated radiator pressure (the impedance numerator), and all four
CPU/ROCm boundary-and-field combinations on a full horizontal polar. The default
is the XY-symmetry waveguide fixture at 8 kHz and q4/s4. Environment variables
prefixed with `BLAB_DIAG_` select another mesh, frequency, quadrature order,
symmetry mode, driven tag, or repeated-solve count.

AMDGPU.jl reads rocSOLVER's `getrf` status from device memory. On this TheRock
RDNA2 stack that read can rarely return an impossible large negative value even
though factorization inputs are valid. Boundary Lab synchronizes the solve and
retries only this corrupt-status signature; real LAPACK argument and singularity
errors still propagate. A worker that nevertheless reports a failed job is
retired instead of being reused by the next solve.

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

On the RX 6700 XT (`gfx1031`) with workgroup size 64, q4/s4, two warmups, and
five measured repetitions, the initial colored pair-owned implementation gave:

| Fixture | Entry-owned regular | Pair-owned regular | Regular speedup | Entry-owned total assembly | Pair-owned total assembly | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sample.msh` | 222.6 ms | 101.7 ms | 2.19x | 278.0 ms | 157.1 ms | 1.77x |
| `sample_detailed.msh` | 1410.8 ms | 740.4 ms | 1.91x | 1710.0 ms | 1038.1 ms | 1.65x |

These are warmed-worker medians and exclude cold Julia/GPU compilation. The
sample fixture uses 10 element colors (200 regular-kernel launches); the detailed
fixture uses 12 colors (288 launches). Including the unchanged dense solve and a
144-point GPU field evaluation, the summed stage medians improved from 296.1 ms
to 175.9 ms (1.68x) on `sample.msh`, and from 1777.1 ms to 1104.4 ms (1.61x) on
`sample_detailed.msh`.
