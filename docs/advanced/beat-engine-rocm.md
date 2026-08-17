# BEAT Engine AMD ROCm

BEAT Engine AMD ROCm is Boundary Lab's local AMD GPU backend. It uses the same
mesh model, Burton-Miller formulation, symmetry rules, coupled-system equations,
and result protocol as the other BEAT Engine backends while moving the dense BEM
work to ROCm through AMDGPU.jl.

The backend supports:

- exterior Burton-Miller BEM solves;
- coupled FEM-BEM-LEM physical-system solves;
- `off`, `x`, and `xy` symmetry;
- GPU-resident regular and singular operator assembly;
- rocBLAS construction of dense algebraic terms;
- rocSOLVER dense complex factorization and solve; and
- GPU exterior-field evaluation for polar, spherical, and arbitrary observation
  points.

Production solves use `Float32` and `ComplexF32`. See [BEAT Engine
Core](beat-engine-core.md) for the shared boundary-integral formulation and
[Coupled Solver](../Coupled%20Solver.md) for the physical-system model.

## Execution model

### Exterior BEM

Boundary Lab prepares mesh topology, quadrature rules, symmetry transforms, and
frequency-independent cache data on the CPU. The ROCm worker then:

1. allocates the single-layer, double-layer, adjoint double-layer, and
   hypersingular matrices as `ROCArray` objects;
2. evaluates regular Galerkin pairs with native pair-owned ROCm kernels;
3. evaluates adjacent and coincident pairs with Duffy singular quadrature and
   gathers their compact correction blocks into the dense operators;
4. applies symmetry-image contributions and reduced-domain row weights;
5. forms the Burton-Miller system with rocBLAS and solves it with rocSOLVER; and
6. evaluates the exterior field with native ROCm kernels.

The default regular assembly uses element coloring so pair-owned kernels can
scatter into the dense operators without atomics. Frequency-independent geometry,
singular-pair, identity, and field caches are retained by the persistent Julia
worker and reused across a frequency sweep.

### Coupled FEM-BEM-LEM

Coupled solves use hybrid static condensation. Sparse FEM assembly and the
UMFPACK factorization of FEM interior degrees of freedom run on the CPU. Boundary
Lab forms the exact Schur complement, retains the FEM-BEM interfaces, moving
surfaces, BEM unknowns, and lumped electromechanical variables, and uploads the
reduced coupled system for the rocSOLVER dense solve. Eliminated FEM pressure is
reconstructed after solving.

This keeps sparse FEM work on the CPU while using the GPU for the dense BEM and
retained coupled system. Exterior field evaluation remains on the GPU.

### Symmetry

The ROCm backend supports the same positive-domain symmetry convention as BEAT
Engine CPU and Nvidia CUDA. `x` represents a positive-X half model, while `xy`
represents a positive-X/positive-Y quarter model. Reflected regular and singular
source contributions are assembled directly into the reduced operators. Field
evaluation includes the reflected sources, and radiator impedance is scaled to
represent the complete physical radiator set.

## Requirements

A functional installation requires a supported AMD GPU and driver, Julia, the
dedicated `src/blab/solvers/julia_rocm` environment, AMDGPU.jl core functionality,
rocBLAS, and rocSOLVER with its runtime dependencies.

GPU support varies by SDK release. Check AMD's current
[Windows system requirements](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/shared/hipsdk/reference/system-requirements.html)
before installing the [Windows HIP SDK](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/).

The four dense BEM operators scale quadratically with the boundary unknown count,
so available GPU memory normally determines the largest practical mesh. The first
solve in a new Julia process also includes kernel compilation; later frequencies
and solves reuse the warm worker and its caches.

## Windows setup

`01_install_update_boundary-lab.bat` detects and validates an existing ROCm SDK,
prepares the dedicated Julia environment, and runs the AMDGPU.jl runtime checks.
It does not download AMD's SDK because AMD's installer requires separate license
acceptance and may require administrator access.

Boundary Lab checks these locations in order:

1. `BLAB_ROCM_PATH`, for a temporary or explicitly managed override;
2. the saved Boundary Lab ROCm configuration;
3. `HIP_PATH`, `ROCM_PATH`, and `ROCM_HOME`;
4. installed versions below `%ProgramFiles%\AMD\ROCm`, newest first.

Inspect the selected SDK and all candidates with:

```powershell
blab rocm detect --json
```

AMD's standard Windows HIP SDK normally needs no Boundary Lab-specific path.
After installing it, open a new terminal or rerun the Boundary Lab installer.
For a portable TheRock SDK outside the standard location, save its root without
setting a machine-wide environment variable:

```powershell
blab rocm configure "$env:USERPROFILE\SDKs\ROCm-TheRock"
```

The saved path is stored under the current user's local application-data folder.
Use `blab rocm clear` to return to environment-variable and standard-installation
discovery.

### Manual Julia setup

To prepare the Julia environment manually from the repository root:

```powershell
julia --project=src/blab/solvers/julia_rocm -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

For one terminal session, `BLAB_ROCM_PATH` remains available as the highest-priority
override:

```powershell
$env:BLAB_ROCM_PATH = (Resolve-Path "$env:USERPROFILE\SDKs\ROCm-TheRock").Path
```

The subprocess adapter maps the selected SDK to `ROCM_PATH`, `ROCM_HOME`, and
`HIP_PATH`, and prepends its `bin` directory to `PATH` for ROCm workers. A portable
TheRock layout used with the currently pinned AMDGPU.jl must expose an unversioned
`amdhip64.dll` at its SDK root. Official AMD installations use AMDGPU.jl's standard
Program Files discovery. In every case the SDK must make `AMDGPU.functional()`,
`AMDGPU.functional(:rocblas)`, and `AMDGPU.functional(:rocsolver)` return `true`.

## Selecting the backend

In application preferences, select **BEAT Engine (AMD ROCm)**. The corresponding
backend identifier used by project and server workflows is `beat_rocm`.

The ROCm backend is loaded in its own Julia project, so AMDGPU.jl is not required
for CPU or Nvidia CUDA solves. If a ROCm worker fails a job, Boundary Lab retires
that worker instead of reusing potentially invalid device state.

## Runtime controls

Normal application use does not require these environment variables. They are
available for SDK selection, compatibility diagnosis, and kernel comparison.

| Variable | Default | Purpose |
|---|---|---|
| `BLAB_ROCM_PATH` | Automatically discovered | Select a specific SDK root for the current process. |
| `BLAB_ROCM_ASSEMBLY_MODE` | `native` | Use `host_staged` to assemble operators on the CPU and upload them as a diagnostic fallback. |
| `BLAB_ROCM_REGULAR_KERNEL_MODE` | `pair_owned` | Use `entry_owned` as an alternate regular-assembly correctness reference. |
| `BLAB_ROCM_KERNEL_GROUPSIZE` | `64` | Set the native kernel workgroup size to `32`, `64`, `128`, or `256`. |

The production defaults are `native` assembly and `pair_owned` regular kernels.
The alternate modes are diagnostic paths rather than separate application
backends.

## Verification

First verify SDK discovery:

```powershell
blab rocm detect --json
```

Then verify the Julia runtime directly:

```powershell
julia --project=src/blab/solvers/julia_rocm -e 'using AMDGPU; AMDGPU.functional() || error("ROCm unavailable"); AMDGPU.functional(:rocblas) || error("rocBLAS unavailable"); AMDGPU.functional(:rocsolver) || error("rocSOLVER unavailable"); AMDGPU.versioninfo()'
```

The repository includes CPU-versus-ROCm validation scripts for each production
path:

| Script | Coverage |
|---|---|
| `validate_rocm_exterior.jl` | Operators, boundary pressure, residual, and exterior field for an exterior solve. |
| `validate_rocm_symmetry.jl` | X and XY reduced-domain assembly and solve parity. |
| `validate_rocm_coupled.jl` | Coupled FEM-BEM-LEM assembly, static condensation, solution, and reconstruction. |
| `validate_rocm_native_regular.jl` | Native regular-pair kernels without singular-pair corrections. |

For example:

```powershell
julia --project=src/blab/solvers/julia_rocm `
  src/blab/solvers/julia_local/scripts/validate_rocm_exterior.jl
```

The validation scripts exit with an error when CPU-versus-ROCm differences exceed
their tolerances. `BLAB_VALIDATE_REGULAR_ORDER` and
`BLAB_VALIDATE_SINGULAR_ORDER` can override the quadrature orders used by the
exterior and symmetry fixtures.

## Operational behavior

- A cold Julia worker compiles ROCm kernels before its first solve. Steady-state
  solve time should be evaluated after warm-up.
- Frequency-independent caches remain resident for the worker's lifetime and are
  released when the worker exits or is retired.
- A recognized corrupt rocSOLVER status read is retried a bounded number of times;
  argument errors, singular systems, and other factorization failures still
  propagate normally.
- `host_staged` assembly is useful for separating GPU-kernel issues from SDK or
  linear-solver issues, but it is not the production performance path.
