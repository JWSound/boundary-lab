# S218BP Level 3 parity-ROM rank experiment — 2026-09-01

## Result

A four-sector parity-aware reduction is feasible for the S218BP exact condensed
interior. At 20, 100, and 250 Hz, a rank of 16 per parity sector gives roughly
one-percent p95 held-out boundary-output error. Rank 32 per sector reduces p95
error below 0.21% at all three frequencies.

This is an output-subspace feasibility result, not yet a production ROM. The next
step must construct a stable reduced state model and validate driven electrical,
mechanical, impedance, and audience-field quantities.

## Method

- Exact Level 3 S218BP system, 4,006 condensed interior states.
- Full exterior boundary: 4,156 P1 pressure nodes and 8,308 DP0 flux faces.
- Four parity sectors: even/even, odd/even, even/odd, and odd/odd.
- 128 training and 32 held-out boundary fields per sector.
- Field family: 50% plane waves and 50% exterior point sources, with point-source
  clearance from 0.02 to 0.80 cabinet spans.
- The exact interior operator was factored on the RTX 2080 Ti. POD Gram matrices
  were accumulated in ComplexF64 to avoid Float32 tail-spectrum artifacts.
- Error is relative projection error in the exterior boundary normal-derivative
  response. Reported p95 values are preferred over maxima because an odd/odd
  held-out field can have nearly zero exact response, making pure relative error
  ill-conditioned.

## Held-out p95 relative error

| Frequency | Rank/sector | even/even | odd/even | even/odd | odd/odd | Worst |
|---:|---:|---:|---:|---:|---:|---:|
| 20 Hz | 8 | 10.26% | 4.75% | 6.15% | 7.92% | 10.26% |
| 20 Hz | 16 | 1.10% | 0.76% | 0.63% | 1.43% | 1.43% |
| 20 Hz | 32 | 0.040% | 0.031% | 0.101% | 0.208% | 0.208% |
| 100 Hz | 8 | 4.25% | 1.85% | 5.67% | 7.40% | 7.40% |
| 100 Hz | 16 | 0.44% | 0.27% | 0.51% | 1.23% | 1.23% |
| 100 Hz | 32 | 0.029% | 0.029% | 0.090% | 0.110% | 0.110% |
| 250 Hz | 8 | 5.27% | 5.04% | 5.01% | 5.48% | 5.48% |
| 250 Hz | 16 | 0.56% | 0.58% | 0.67% | 0.76% | 0.76% |
| 250 Hz | 32 | 0.029% | 0.026% | 0.100% | 0.085% | 0.100% |

The median parity leakage of the exact responses was approximately 1e-5 to 4e-5.
This confirms that the reflection maps and the four-sector decomposition are
numerically consistent with the expanded S218BP geometry.

## Experimental runtime

| Frequency | Exact interior factor | 640 exact RHS solves | Rank analysis | Experiment total |
|---:|---:|---:|---:|---:|
| 20 Hz | 1.189 s | 0.211 s | 2.812 s | 6.764 s |
| 100 Hz | 1.202 s | 0.235 s | 2.851 s | 6.905 s |
| 250 Hz | 1.194 s | 0.234 s | 2.882 s | 6.976 s |

Each standalone command took about 75 seconds cold. Roughly 68 seconds was Julia,
CUDA, exact FEM/BEM assembly, and the ordinary coupled solve; it is not ROM online
cost and is amortized by a warmed worker.

## Estimated package footprint

The estimate includes symmetry-reduced state, input, and output bases plus four
small reduced operators, stored as Complex64 at every frequency. It does not yet
include archive overhead or the Level 2 payload.

| Rank/sector | Per frequency | 100 frequencies |
|---:|---:|---:|
| 8 | 1.01 MiB | 100.7 MiB |
| 16 | 2.02 MiB | 201.8 MiB |
| 32 | 4.05 MiB | 405.2 MiB |
| 64 | 8.17 MiB | 816.7 MiB |
| 96 | 12.34 MiB | 1.21 GiB |
| 128 | 16.58 MiB | 1.62 GiB |

Rank 32 therefore points to about 0.4 GiB for 100 frequencies, over two orders of
magnitude smaller than the 100–150 GiB dense exact-macro failure mode.

## Recommendation

Use rank 32 per parity sector as the first production-ROM target and retain rank 16
as an optional compact tier. Build the online coupling as a Schur-eliminated or
matrix-free low-rank boundary update; directly appending all reduced states to the
already VRAM-limited eight-cabinet dense BEM system would spend avoidable memory.

Before adopting the format, validate a state-space or Petrov-Galerkin realization
at all 100 export frequencies against:

1. audience pressure (complex and SPL),
2. terminal impedance and voice-coil current,
3. diaphragm velocity,
4. mutual-loading changes for close two-, four-, and eight-cabinet layouts, and
5. passivity/stability under interpolation between exported frequencies.

Raw reports are generated under `runs/.speaker_rom_rank/` and remain uncommitted.
The reproducible entry point is `scripts/experiment_speaker_rom_rank.py`.
