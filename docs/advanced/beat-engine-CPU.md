# BEAT Engine CPU

The BEAT Engine CPU backend is a hardware-agnostic Julia/OpenBLAS solver path. It uses the same BEAT Engine request protocol, mesh handling, Burton-Miller formulation, symmetry model, and result stream described in [BEAT Engine Core](beat-engine-core.md), but performs operator assembly, dense solve, and field evaluation on the host.

The application exposes this path as `BEAT Engine (CPU)` / `beat_cpu`.

## CPU Solve Path

For each frequency, the CPU path:

1. Selects the regular quadrature rule for that frequency.
2. Assembles dense Galerkin operators on host matrices.
3. Applies singular and image-singular Duffy corrections directly into those matrices.
4. Builds the Burton-Miller dense system.
5. Solves through Julia's dense BLAS/LAPACK path.
6. Evaluates requested polar and spherical fields on the CPU.

The CPU implementation lives under `src/blab/solvers/julia_local/src/BeatEngineCpu*.jl`.

## CPU Operator Assembly

`BeatEngineCpuAssembly.jl` is the CPU Galerkin operator assembly entry point. It precomputes per-element geometry and per-element quadrature data, then loops over test/trial element pairs.

Regular pair assembly skips adjacent/coincident pairs, which are handled afterward by singular corrections. When threaded assembly is enabled and Julia has more than one thread, element coloring is used to avoid write conflicts while scattering element-block contributions into shared dense matrices.

Symmetry image contributions are assembled on the host by reflecting trial/source element geometry and quadrature data. Reflected image pairs that become singular across a symmetry plane use image-singular correction caches.

## Dynamic Quadrature

The CPU path defaults to wavelength-driven regular quadrature. This is a CPU-only production feature; CUDA currently remains fixed-order as the GPU solver does not meaningfully benefit from dynamic quadrature.

The goal is to reduce low-frequency regular-pair assembly cost without changing singular-pair quadrature. Regular-pair assembly dominates dense BEM workload, and low frequencies do not need the same regular quadrature density as high frequencies on typical Boundary Lab loudspeaker meshes.

The selector computes:

$$
k h = \frac{2\pi f}{c}\sqrt{A_{\mathrm{stat}}},
$$

where \(A_{\mathrm{stat}}\) is a mesh element-area statistic. The current default statistic is `p90`, so \(h = \sqrt{A_{p90}}\).

Default CPU thresholds:

- q1 disabled: `wavelength_kh_q1_max = 0.0`
- q2 when `k*h <= 2.0`
- q4 above `k*h > 2.0`
- base order: `quadrature_order = 4`
- mesh statistic: `wavelength_mesh_stat = "p90"`

The selected order is cached per frequency order. The CPU path reuses per-order regular rules, identity/mass matrices, and field-evaluation caches. Singular quadrature remains controlled by `singular_order` and is not reduced by the wavelength selector.

Result diagnostics include the selected mode, selected order, mesh statistic, element length, and `k*h` value:

- `regular_quadrature_mode`
- `regular_quadrature_order`
- `regular_quadrature_base_order`
- `regular_quadrature_wavelength_mesh_stat`
- `regular_quadrature_wavelength_element_length_m`
- `regular_quadrature_wavelength_kh`
- `regular_quadrature_wavelength_kh_q1_max`
- `regular_quadrature_wavelength_kh_q2_max`

## Validation Notes

The tuned defaults were validated against fixed q4 on `sample.msh` using output-level checks from `compare_cpu_quadrature.jl`.

The strongest current candidate was:

- q1 disabled
- q2 while `k*h <= 2.0`
- q4 above that
- `p90` mesh-area statistic

On `sample.msh`, this selected q2 through 4 kHz and q4 at 4.5 kHz and above. The full-mesh output comparison measured approximately:

- median operator assembly speedup: `3.26x`
- max pressure relative L2 error: `0.148%`
- max field relative L2 error: `0.442%`
- max SPL RMS delta: `0.116 dB`
- max SPL p95 delta: `0.265 dB`
- max SPL max delta: `0.356 dB`

Raw operator relative error can exceed 1% even when solved pressure and field outputs remain well below the intended output-error target. Threshold tuning should therefore prioritize solved/output metrics over operator norms alone.

## CPU Field Evaluation

`BeatEngineCpuField.jl` evaluates the same field integral described in [BEAT Engine Core](beat-engine-core.md). It uses the CPU field-evaluation cache for the selected quadrature order and computes the potential at concatenated horizontal, vertical, and optional spherical observation points.

Field evaluation is usually not the dominant CPU cost compared with dense operator assembly and dense solve.

## CPU Dense Solve

`BeatEngineCpuSolve.jl` forms the Burton-Miller system on host matrices and calls Julia's dense solve path. Runtime depends heavily on BLAS/LAPACK performance and P1 unknown count. Symmetry can reduce solve cost significantly because dense solve complexity scales roughly as \(O(N_p^3)\).

## CPU Benchmark And Comparison Scripts

Useful scripts:

- `src/blab/solvers/julia_local/scripts/benchmark_cpu.jl`: CPU timing benchmark, including fixed and wavelength regular quadrature modes.
- `src/blab/solvers/julia_local/scripts/compare_cpu_quadrature.jl`: fixed-reference versus candidate comparison artifact generator with operator, pressure, field, and SPL error metrics.

Example comparison:

```powershell
& 'C:\Users\John\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe' `
  src\blab\solvers\julia_local\scripts\compare_cpu_quadrature.jl `
  --frequencies 20,50,100,200,500,1000,1500,2000,3000,4000,4500,5000 `
  --subset-faces 0 `
  --output-points 72 `
  --wavelength-kh-q1-max 0 `
  --wavelength-kh-q2-max 2.0 `
  --json src\blab\solvers\julia_local\results\cpu_quadrature_compare_output_no_q1_q2_2p0_full.json
```

## Important Files

- `src/blab/solvers/julia_local/src/BeatEngineCpu.jl`: include hub for the CPU implementation files.
- `src/blab/solvers/julia_local/src/BeatEngineCpuAssembly.jl`: CPU Galerkin operator assembly entry point.
- `src/blab/solvers/julia_local/src/BeatEngineCpuField.jl`: CPU field-evaluation path.
- `src/blab/solvers/julia_local/src/BeatEngineCpuSolve.jl`: CPU Burton-Miller dense solve through Julia's LAPACK/BLAS path.
- `src/blab/solvers/julia_local/solver.jl`: CPU backend dispatch, wavelength quadrature selection, per-order caches, and result diagnostics.
