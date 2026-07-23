# BEAT Engine CPU Performance Baseline - 2026-07-23

This note records a warm-run performance baseline for the BEAT Engine CPU
solver. The principal workload uses `sample_quarter.msh` with `xy` symmetry,
which represents the complete symmetric acoustic problem through three image
transforms.

The first half of this investigation captured the unmodified baseline. The
optimization pass described below was then measured against those artifacts.

## Test Environment

- CPU: AMD Ryzen 7 5700X, 8 cores / 16 hardware threads
- Memory: 31.9 GiB
- Julia: 1.12.6
- BLAS: Julia ILP64 OpenBLAS (`libopenblas64_.dll`)
- Precision: `Float32` / `ComplexF32`
- Julia threads: 16
- Production baseline BLAS threads: 16
- Mesh scale: 0.001

Each microbenchmark used two complete warmups followed by three or five
measured repetitions. The end-to-end sweep used one complete 48-frequency job
as its warmup. Reported microbenchmark values are medians.

## Workloads

The reduced mesh has:

- 798 triangular elements
- 441 P1 unknowns
- 798 DP0 unknowns
- 2,537,296 regular real/image element pairs under `xy` symmetry
- 9,920 singular real-element pairs
- 74 default 10-degree horizontal and vertical polar observation points

Production quadrature settings were used: wavelength-selected regular q2/q4,
singular order 4, the p90 element-size statistic, and the default `k*h = 2`
q2 cutoff.

## Results

### Isolated Frequencies

| Workload | Total | Operator assembly | Regular | Singular | Image singular | Dense system/solve | Field |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 kHz, q2 | 0.246 s | 0.218 s | 0.156 s | 0.047 s | 0.010 s | 0.023 s | 0.0019 s |
| 20 kHz, q4 | 0.659 s | 0.626 s | 0.556 s | 0.056 s | 0.010 s | 0.026 s | 0.0038 s |

At 1 kHz, operator assembly is 88% of measured time. At 20 kHz it is
95%. The regular-pair loop alone is 63% and 84%, respectively.

The q2/q4 change is important because a q4 regular rule performs four times as
many test/trial quadrature combinations per element pair as q2. At 1 kHz,
fixed q4 took 0.632 s total versus 0.246 s with wavelength-selected q2, a
2.57x end-to-end improvement. Regular assembly improved by 3.37x.

### Typical 48-Frequency Job

The measured production-backend payload used:

- 400 Hz to 20 kHz, 48 logarithmically spaced frequencies
- 10-degree horizontal and vertical polar sampling
- one radiator/channel
- spherical sampling disabled
- `sample_quarter.msh` with `xy` symmetry
- one complete warmup job before timing

The measured wall time was **20.52 s**:

| Stage | Total | Share |
|---|---:|---:|
| Operator assembly | 17.65 s | 86.0% |
| Dense system and solve | 2.71 s | 13.2% |
| Field evaluation and result construction | 0.13 s | 0.6% |
| Process/protocol overhead | 0.04 s | 0.2% |

For this mesh, the q2 cutoff is approximately 4.50 kHz. The default sweep
therefore contains 30 q2 frequencies and 18 q4 frequencies.

### Thread Scaling

At 1 kHz q2 with matching Julia and BLAS thread counts:

| Threads | Total | Operator assembly |
|---:|---:|---:|
| 1 | 1.523 s | 1.500 s |
| 2 | 0.811 s | 0.792 s |
| 4 | 0.438 s | 0.423 s |
| 8 | 0.314 s | 0.295 s |
| 16 | 0.248 s | 0.209 s |

Assembly achieves a 7.17x speedup at 16 threads. Element coloring and its ten
thread groups do not prevent useful scaling, although efficiency is about 45%
at 16 threads.

OpenBLAS behaves differently for the small 441-by-441 dense system. With
Julia fixed at 16 threads, the median linear solve was 3.1 ms with one BLAS
thread and 19.7 ms with 16 BLAS threads. The production policy currently sets
BLAS threads equal to Julia threads, so an unknown-count-aware BLAS policy is a
promising small-mesh optimization. It must also be tested on larger systems,
where multithreaded factorization should become beneficial.

### Symmetry and Spherical Sampling

The equivalent full `sample.msh` problem has 2,776 elements and 1,390 P1
unknowns. Compared with the quarter mesh plus `xy` images, full-mesh timings
were:

| Frequency | Quarter + `xy` | Full mesh | Quarter speedup |
|---:|---:|---:|---:|
| 1 kHz q2 | 0.246 s | 0.915 s | 3.71x |
| 20 kHz q4 | 0.659 s | 2.041 s | 3.10x |

The quarter-domain assembly still evaluates three image transforms, so its
regular work is approximately one quarter, rather than one sixteenth, of the
full mesh. Its smaller dense system becomes increasingly valuable as unknown
count grows.

With 6,000 spherical observation points, field evaluation grew to 0.127 s at
q2 and 0.295 s at q4. It accounted for approximately 27-30% of the measured
single-frequency total. Field optimization is therefore secondary for normal
polar jobs but important for balloon generation.

## Bottlenecks

1. **Regular Galerkin pair assembly.** The innermost loop evaluates distance,
   trigonometric Green-function terms, normal projections, and four operator
   blocks for every element-pair quadrature combination. This is the dominant
   cost in both q2 and q4 jobs.
2. **Singular corrections at low frequency.** Their roughly 50 ms cost is
   almost independent of regular quadrature order, so they are about 26% of
   q2 operator time but only about 11% of q4 operator time when image-singular
   work is included.
3. **Dense solve policy on small reduced meshes.** OpenBLAS thread startup and
   synchronization exceed the benefit of parallel factorization at 441 P1
   unknowns.
4. **Field evaluation for spherical payloads.** The default 74-point polar
   field is negligible, while 6,000-point balloon evaluation is material.
5. **Allocation is not the immediate runtime limiter.** A q4 run allocated
   about 41 MB in total, but operator time not attributed to measured regular
   and singular loops was only about 3 ms. Reuse remains useful for memory
   scaling and GC stability on larger meshes.

## Implemented Optimization Pass

The first optimization pass made four bounded changes:

1. A P1-unknown-aware BLAS thread policy selects 1, 4, or 8 threads for the
   small, medium, and large fixtures. Systems above 4,096 P1 unknowns retain
   all available threads. `BLAB_BEAT_CPU_BLAS_THREADS` provides an explicit
   override.
2. The regular and singular kernels now compute sine and cosine together,
   reuse inverse radius, and avoid multiplying every operator contribution by
   a general complex scale when the production paths only need addition or a
   final block subtraction.
3. CPU assembly caches now retain element data, per-order quadrature geometry,
   coloring, reflected image data, and image-singular topology across
   frequencies. Cache memory scales with elements times quadrature size, not
   with the square of the element count.
4. Image-singular pairs now use the concrete `SingularCorrectionPair` type
   instead of `Vector{Any}` storage.

### Before and After by Fixture

All isolated cases used two warmups. Small cases used five measured
repetitions; medium and large cases used three. The isolated optimized totals
include CPU cache construction, even though production sweeps amortize it
across frequencies.

| Fixture | Elements / P1 | Frequency | Baseline | Optimized | Speedup |
|---|---:|---:|---:|---:|---:|
| Small quarter + `xy` | 798 / 441 | 1 kHz q2 | 0.246 s | 0.176 s | 1.40x |
| Small quarter + `xy` | 798 / 441 | 20 kHz q4 | 0.659 s | 0.422 s | 1.56x |
| Medium full | 2,776 / 1,390 | 1 kHz q2 | 0.915 s | 0.741 s | 1.23x |
| Medium full | 2,776 / 1,390 | 20 kHz q4 | 2.041 s | 1.424 s | 1.43x |
| Large detailed | 7,000 / 3,502 | 1 kHz q2 | 5.140 s | 4.397 s | 1.17x |
| Large detailed | 7,000 / 3,502 | 20 kHz q4 | 11.298 s | 8.151 s | 1.39x |

The warmed 48-frequency small job improved from **20.52 s to 12.70 s**, a
38.1% reduction or 1.62x throughput:

| Stage | Baseline | Optimized | Reduction |
|---|---:|---:|---:|
| Operator assembly | 17.65 s | 11.96 s | 32.3% |
| Dense system and solve | 2.71 s | 0.56 s | 79.4% |
| Field/result work | 0.13 s | 0.14 s | no material change |
| Whole solver job | 20.52 s | 12.70 s | 38.1% |

The large q2 workload improves less because its lower arithmetic intensity is
closer to memory and synchronization limits. Using eight Julia assembly
threads instead of sixteen made it slower, so SMT remains useful on the test
machine.

Pressure-norm differences between baseline and optimized artifacts were at
most `2.94e-7` relative; field-norm differences were at most `4.30e-7`.
Cached and uncached assembly paths are also compared matrix-by-matrix in the
Julia CPU tests.

## Ranked Optimization Opportunities

### 1. Tune BLAS threads by P1 unknown count - implemented

The current policy is validated on all three fixture sizes. Recalibrate the
thresholds when changing BLAS implementations or adding substantially larger
fixtures.

### 2. Optimize the regular-pair arithmetic kernel - first pass implemented

Implemented:

- compute sine and cosine together with `sincos`;
- remove the general complex `scale` multiplication from the common positive
  regular-pair path and use a specialized subtraction path only for singular
  deltas;
- reuse inverse radius in the Green function and gradient terms.

Remaining work is to inspect generated code and hardware counters, then test
SIMD-friendly batching over trial quadrature points or trial elements.

### 3. Reduce repeated per-frequency geometry work - bounded cache implemented

The job-level cache covers linear-size geometry and topology. Pairwise normal
products, area products, and q4 source-pair geometry remain uncached because
their memory use scales quadratically and can exceed the four dense operators.

### 4. Improve singular correction representation - implemented

Image-singular topology is now typed and reused per job. Removing the unused
general scale from the singular arithmetic path also reduced small q2 real
singular time from roughly 47 ms to 33-38 ms.

### 5. Avoid RHS-operator materialization

The CPU solve currently materializes a dense combined RHS operator before
applying it to each channel. A fused sequence of matrix-vector products may
reduce allocation and setup time, especially for one-channel large jobs.
Benchmark the tradeoff for multiple channels before changing this path.

### 6. Add a balloon-specific field path - deferred

For thousands of observation points, investigate point batching and SIMD in
the source loop. Geometry-only field terms may be cached for small polar
payloads, but a full 6,000-point-by-source cache is likely too large. A
matrix-free batched or accelerated far-field path is preferable for balloon
jobs.

### 7. Longer-term algorithmic scaling

Dense operators and direct LU retain quadratic memory and cubic solve scaling.
For substantially larger meshes, hierarchical matrices, FMM-accelerated
iterative solves, or frequency-to-frequency initial guesses will eventually
offer larger gains than local kernel tuning. These are higher-risk numerical
projects and should follow the lower-risk scheduling and kernel work above.

## Reproduction

Isolated 1 kHz benchmark:

```powershell
& 'C:\Users\John\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe' `
  --threads=16 `
  --project=src\blab\solvers\julia_local `
  --startup-file=no `
  src\blab\solvers\julia_local\scripts\benchmark_cpu.jl `
  --mesh src\blab\solvers\julia_local\test_meshes\sample_quarter.msh `
  --freq 1000 `
  --quadrature-mode wavelength `
  --quadrature-order 4 `
  --singular-order 4 `
  --eval-points 74 `
  --subset-faces 0 `
  --symmetry xy `
  --warmups 2 `
  --repetitions 5 `
  --blas-threads 16
```

For the optimized small-fixture policy, use `--blas-threads 1 --cpu-cache`.
The production backend selects the equivalent BLAS setting automatically.

Real-system BLAS crossover benchmark:

```powershell
& 'C:\Users\John\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe' `
  --threads=16 `
  --project=src\blab\solvers\julia_local `
  --startup-file=no `
  src\blab\solvers\julia_local\scripts\benchmark_cpu_blas.jl `
  --mesh src\blab\solvers\julia_local\test_meshes\sample_detailed.msh `
  --quadrature-order 2 `
  --singular-order 4 `
  --thread-counts 1,2,4,8,16 `
  --warmups 2 `
  --repetitions 5
```

Typical production-backend sweep:

```powershell
python -u scripts\benchmark_gui_pipeline.py `
  --mode julia `
  --solver-backend beat_cpu `
  --mesh src\blab\solvers\julia_local\test_meshes\sample_quarter.msh `
  --symmetry xy `
  --solver-warmups 1 `
  --julia-threads auto `
  --freq-min 400 `
  --freq-max 20000 `
  --freq-count 48 `
  --angle-step 10 `
  --sphere-points 0
```

Raw JSON artifacts from this run are under the ignored local paths
`src/blab/solvers/julia_local/results/benchmark_cpu_*_final*.json`,
`runs/benchmark_beat_cpu_quarter_xy_typical.json`, and
`runs/benchmark_beat_cpu_quarter_xy_optimized_final.json`.
