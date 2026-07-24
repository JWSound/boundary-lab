# BEAT Engine CUDA

The BEAT Engine CUDA backend is an Nvidia GPU-accelerated Julia solver path. It uses the same BEAT Engine request protocol, mesh handling, Burton-Miller formulation, symmetry model, and result stream described in [BEAT Engine Core](beat-engine-core.md), but performs regular-pair assembly, singular corrections, dense solve, and field evaluation with CUDA.jl.

The application exposes this path as `BEAT Engine (CUDA)` / `beat_cuda`.

## CUDA Operator Assembly

The CUDA path assembles dense Galerkin operators by splitting triangle pairs into regular pairs and adjacent/coincident singular pairs. Regular pairs are assembled by CUDA kernels over test/trial element pairs. Adjacent, edge-sharing, vertex-sharing, and coincident pairs are handled by a separate GPU Duffy correction path.

The application backend requires CUDA and moves geometry and quadrature arrays to the GPU:

- face vertices
- normals
- areas
- global face indices
- surface curls
- quadrature points and weights
- test/trial element index arrays

`build_cuda_regular_assembly_cache` keeps these arrays resident across frequencies, avoiding repeated host-to-device transfers for fixed mesh geometry.

For each frequency, CUDA allocates real and imaginary dense matrices for:

- SLP real/imag
- DLP real/imag
- adjoint DLP real/imag
- hypersingular real/imag

The regular-pair kernel skips adjacent pairs. Singular and near-singular corrections use Duffy quadrature and are added after regular assembly.

## CUDA Singular Corrections

A Duffy block kernel computes compact per-pair correction blocks directly on device. A CUDA scatter kernel atomically accumulates those compact blocks into dense correction buffers before adding them to the resident operators. The Duffy kernel reuses the regular assembly geometry cache, so the per-frequency singular correction path does not transfer dense CPU correction matrices. Symmetry image-singular topology is also mirrored to a device cache once during job initialization and reused across the frequency sweep.

The CUDA singular correction cache stores:

- test and trial element indices
- rule indices for orientation-remapped Duffy rules
- P1 row/column dofs and DP0 columns
- pair Jacobian scales and normal products
- flattened remapped Duffy points and weights

Symmetry image-singular pairs use the same compact correction idea, with reflected image geometry and GPU scatter.

## CUDA Kernel Modes

The CUDA backend has a regular assembly split kernel, one singular correction block kernel, and two field-evaluation kernels.

Regular assembly uses a serial pair batched split path. A CUDA block carries 128 independent regular test/trial element pairs, with one CUDA thread responsible for the full quadrature loop for one pair. For the current order-4 regular triangle rule, each thread evaluates the 36 quadrature-point pairs for its element pair directly in registers and then atomically scatters the resulting element-block entries.

The regular path uses two regular assembly launches:

- `_cuda_regular_quadrature_slp_hyp_kernel!` computes single-layer and hypersingular contributions.
- `_cuda_regular_quadrature_dlp_adjoint_kernel!` computes double-layer and adjoint double-layer contributions.

This grouping keeps both launches at 24 accumulator slots, which reduces register/shared-memory pressure compared to a fused all-operator kernel.

The serial pair batched split kernels atomically scatter real and imaginary element-block entries into dense operator buffers. Singular adjacent/coincident pairs are skipped during regular assembly and handled afterward by the Duffy correction path.

The split regular launches use an 80-register compiler cap. On the reference RTX 2080 Ti this raises achieved occupancy from about 49% to 73% while keeping compiler spill traffic predominantly L2-resident. The cap was selected by comparing unrestricted, 88, 84, 80, and 72-register variants on `sample_detailed.msh`; lower caps lost performance to spill traffic.

`_cuda_duffy_blocks_kernel!` maps GPU threads over cached adjacent/coincident element pairs. Each thread computes the compact singular correction block for one or more pairs using the cached remapped Duffy rule. `_cuda_singular_scatter_kernel!` then atomically scatters those compact blocks into dense GPU correction buffers.

`_cuda_weighted_field_sources_kernel!` maps GPU threads over cached source quadrature points and builds weighted pressure and Neumann source strengths. `_cuda_field_eval_kernel!` maps one CUDA block to one observation point and reduces source contributions in dynamic shared memory.

## CUDA Atomics

The GPU kernels accumulate element-block contributions into global dense matrices. Multiple CUDA blocks can target the same global row/column entries, especially because P1 basis functions are shared across adjacent faces. The regular assembly and singular scatter paths use device atomics for global accumulation:

```julia
@inline function _cuda_atomic_add!(array, index, value)
    CUDA.@atomic array[index] += value
    return nothing
end
```

Dense operator accumulation buffers are stored as separate real and imaginary arrays during kernel execution, so atomic additions operate on scalar `Float32` values. After the kernel finishes, real and imaginary matrices are materialized into complex matrices on GPU:

$$
A = A_{\mathrm{re}} + i A_{\mathrm{im}}.
$$

This avoids a slow serial scatter stage and lets regular-pair assembly remain massively parallel. The serial pair batched regular assembly path preserves this property: it atomically accumulates into the same dense buffers through two balanced operator-family kernels instead of one all-operator kernel. The tradeoff is that floating-point atomic accumulation is order-dependent, so tiny run-to-run differences can occur at the last few bits.

## GPU Dense Solve

If operators are returned on GPU, `solve_burton_miller_neumann` forms the Burton-Miller matrix and RHS directly as `CuArray`s:

$$
A = \frac{1}{2}I - D + \eta H,
$$

$$
b = \left(-S - \eta(D^{*} + \frac{1}{2}I_{P1,DP0})\right)q.
$$

For systems above 768 P1 unknowns, the RHS is evaluated matrix-free with three
accumulating GPU matrix-vector products, avoiding a dense combined
RHS-operator temporary. Smaller systems retain the single-GEMV materialized
path because it benchmarks faster. The solver then calls Julia's dense linear
solve:

```julia
d_pressure = d_lhs \ d_rhs
```

With CUDA arrays, this dispatches through CUDA.jl to GPU dense linear algebra. The solved pressure vector is currently copied back to CPU after the solve; CUDA field evaluation then transfers pressure and Neumann vectors back to the GPU for the observation pass. This keeps the public result path simple while leaving room to keep pressure resident across downstream stages later.

Temporary GPU allocations are explicitly released with `CUDA.unsafe_free!` after assembly or solve stages to reduce memory pressure during frequency sweeps.

## CUDA Field Evaluation

`build_cuda_field_evaluation_cache` mirrors the shared field-evaluation cache to GPU as a `CudaFieldEvaluationCache`, using structure-of-arrays storage for coalesced source-point and normal reads.

For each frequency, CUDA field evaluation uses two stages:

1. `_cuda_weighted_field_sources_kernel!` interpolates solved P1 pressure to every source quadrature point, multiplies by quadrature weights, and builds the weighted Neumann source term.
2. `_cuda_field_eval_kernel!` assigns one CUDA block to each observation point. Threads in the block stride over all source quadrature points, evaluate the single-layer and double-layer kernels, accumulate local real/imaginary potentials, and reduce those partial sums in dynamic shared memory.

The result is materialized as a compact potential vector and copied back for SPL conversion and result serialization.

## Quadrature Mode

CUDA currently uses fixed regular quadrature order for the frequency sweep. Wavelength-driven regular quadrature is implemented only for the BEAT CPU path at this time.

CUDA's regular assembly cache is built from `mesh + rule`, so future CUDA wavelength quadrature would need per-order CUDA regular caches and corresponding field/identity cache handling.

## Performance Notes

CUDA accelerates regular-pair assembly, singular Duffy corrections, the dense solve, and field evaluation. The remaining dominant cost in CUDA solves is usually regular-pair assembly for the dense operators, followed by the dense solve for larger P1 systems. The serial pair batched regular assembly mode reduces regular-kernel pressure by assembling SLP/hypersingular and DLP/adjoint in separate launches while keeping dense operators resident on the GPU and batching 128 regular pairs per CUDA block.

Temporary allocation, dense GPU memory footprint, and atomic accumulation cost are important practical limits. CUDA memory use scales with dense P1/DP0 matrix dimensions, so symmetry is especially useful on large meshes.

`scripts/benchmark_cuda.jl` measures the serial pair batched regular assembly path:

```powershell
julia scripts\benchmark_cuda.jl --skip-solve --skip-field --warmups 3 --repetitions 5
```

Use `sample_detailed.msh` with multiple warmups for hardware comparisons. Nsight Compute can profile the measured regular kernels with a kernel filter such as `regex:.*regular_quadrature.*`; on Windows, detailed counters require NVIDIA performance-counter permission to avoid `ERR_NVGPUCTRPERM`.

## Important Files

- `src/blab/solvers/julia_local/src/BeatEngineCuda.jl`: include hub for the CUDA implementation files.
- `src/blab/solvers/julia_local/src/BeatEngineCudaCommon.jl`: CUDA package setup, shared cache structs, and shared device helpers.
- `src/blab/solvers/julia_local/src/BeatEngineCudaRegular.jl`: CUDA geometry/rule cache builders and regular-pair kernels.
- `src/blab/solvers/julia_local/src/BeatEngineCudaSingular.jl`: GPU Duffy corrections, singular cache mirroring, image-singular corrections, and scatter kernels.
- `src/blab/solvers/julia_local/src/BeatEngineCudaOperators.jl`: GPU operator storage helpers, timing helpers, and regular kernel launch helpers.
- `src/blab/solvers/julia_local/src/BeatEngineCudaAssembly.jl`: public CUDA Galerkin operator assembly entry point.
- `src/blab/solvers/julia_local/src/BeatEngineCudaField.jl`: GPU field-evaluation cache, source weighting, and observation kernels.
- `src/blab/solvers/julia_local/src/BeatEngineCudaProfiling.jl`: optional CUDA regular-kernel probe and profiling launches used by benchmark scripts.
