# S218BP rank-32 parity Petrov–Galerkin Deploy benchmark — 2026-09-01

## Outcome

The rank-32-per-sector model and Schur-eliminated Deploy path meet the interactive
single-frequency target on the test machine. A warmed eight-cabinet solve completes
in 38.08 seconds at 100 Hz on the RTX 2080 Ti. One through four cabinets complete
in 1.63–8.61 seconds.

The package model has four parity sectors (even/even, odd/even, even/odd, and
odd/odd). At each exported frequency it stores reduced `K/C/D/B/E` operators,
exact affine drive maps, and transducer velocity/current feedback maps. The full
condensed FEM state and the dense sampled macro matrices are not stored.

## Reduced-model construction

- Exact condensed S218BP interior: 4,006 states.
- Exterior boundary: 4,156 P1 pressure nodes and 8,308 DP0 flux faces.
- Compact parity boundary: 1,092 pressure-node orbits and 2,077 flux-face orbits.
- Rank: 32 states per sector, 128 reduced states across all four sectors.
- Training: 96 pressure fields per sector.
- Held-out validation: 24 pressure fields per sector.
- Trial basis: boundary-output-weighted response POD.
- Petrov test basis: operator-induced `Kᴴ W = V`, giving a well-conditioned
  reduced `Wᴴ K V` operator.

The independent rank experiment found at most 0.21% p95 held-out flux-response
error over 20, 100, and 250 Hz. The production Petrov–Galerkin realization at
100 Hz produced p95 sector errors of 0.027%, 0.030%, 0.096%, and 0.170%.

Raw rank-32 operators occupy approximately 3.26 MiB per frequency (about 326 MiB
for 100 frequencies) before ZIP/NPZ compression. The measured one-frequency ROM
NPZ payload is about 0.61 MiB. This remains over two orders of magnitude below
the 100–150 GiB dense sampled failure mode.

## Online formulation

For each speaker and parity sector, Deploy eliminates the reduced interior state:

`K_r a + C_r P_s p = B_r u`

`q = Σ_s R_s (D_r a + E_exact,r u)`

The exterior Burton–Miller matrix `L` is assembled and factored once. GMRES solves
the left-preconditioned Schur system

`(I - L⁻¹ R Y) p = L⁻¹ R q_source`,

where each matrix-vector product evaluates the speaker ROMs on the CPU, assembles
only the CUDA Burton–Miller RHS `R q_feedback`, and applies the retained exterior
LU factorization. The final coupled flux is used for audience-field evaluation.

The direct CUDA assembler now writes real and imaginary lanes into one interleaved
allocation and reinterprets it as `ComplexF32` without copying. This removes the
previous real + imaginary + complex dense-matrix peak that caused the eight-cabinet
case to thrash at the 11 GiB VRAM limit.

## Warmed 100 Hz benchmark

All cases use the same S218BP rank-32 package, 1.5 m cabinet spacing, FP32 BEAT
CUDA, quadrature order 2, rigid Y=0 half-space, a persistent Julia worker, and a
5 x 5 audience plane. `Solve` includes the exterior LU factorization and Schur
GMRES; it excludes Python staging and field evaluation.

| Cabinets | Wall | Python prepare | Exterior assembly | Schur solve | GMRES iterations | Relative residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.633 s | 0.054 s | 0.332 s | 0.352 s | 6 | 6.76e-5 |
| 2 | 2.183 s | 0.102 s | 0.346 s | 1.412 s | 7 | 5.00e-5 |
| 4 | 8.608 s | 0.197 s | 1.410 s | 6.589 s | 8 | 3.91e-5 |
| 8 | 38.084 s | 0.377 s | 6.045 s | 30.943 s | 8 | 4.45e-5 |

For comparison, the warmed one-cabinet Level 2 path completes in 1.281 seconds
wall time with a 0.036-second linear solve after the same interleaved-matrix change.

## Exact-oracle validation

The reduced model was compared with the exact frequency-parametric Level 3 path
at 100 Hz using identical drive, placement, rigid ground, and a 17 x 17 audience
plane. The exact affine drive response is stored directly; Petrov–Galerkin is
used only for pressure-induced loading feedback.

| Cabinets | Complex field relative error | SPL RMS | SPL max | Diaphragm velocity | Voice-coil current |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.15% | 0.190 dB | 0.232 dB | 1.09% | 1.15% |
| 2 | 2.08% | 0.188 dB | 0.249 dB | 1.06% | 1.10% |

The two-cabinet result includes the exterior pressure feedback responsible for
mutual loading and supports the selected rank at this representative frequency
and spacing.

## Current validation boundary

This establishes package size, numerical convergence, online feasibility, and a
one-/two-cabinet exact comparison at 100 Hz. It does not yet close the full-band
model-acceptance gate. Before removing the exact path or cleaning up experimental
code, compare the reduced and exact models over the export band for:

1. complex audience pressure and SPL,
2. terminal impedance and voice-coil current,
3. diaphragm velocity,
4. mutual-loading changes in close two-, four-, and eight-cabinet layouts, and
5. interpolation/passivity behavior between exported frequencies.

The reproducible tools are `scripts/build_speaker_rom_package.py`,
`scripts/benchmark_deploy_level3.py`, and `scripts/validate_deploy_speaker_rom.py`.

## Full-band 20-250 Hz validation

The production export was repeated on the corrected subwoofer band using 100
logarithmically spaced frequencies from 20 through 250 Hz. The resulting
rank-32-per-sector package is 78.44 MiB on disk. Its compressed ROM payload is
60.79 MiB (61.24 MiB before the package ZIP layer), while the portable exact
system oracle package is 6.31 MiB. Relative to the former 100-150 GiB dense
sampled packages, the ROM is approximately 1,300-1,960 times smaller.

The exact and ROM paths were compared at every exported frequency for one
cabinet, identical 2.83 V drive, rigid ground, and a 17 x 17 audience plane.
The exact 100-frequency batch completed in 165.7 seconds. The first cold ROM
solve took 30.64 seconds; after geometry/JIT warm-up, ROM solves had a 0.703
second median, 0.764 second p95, and 1.722 second maximum. Schur GMRES used a
median of 7 and a maximum of 8 iterations.

| Quantity | Median | p95 | Maximum | Worst frequency |
| --- | ---: | ---: | ---: | ---: |
| Complex audience-field relative error | 2.05% | 18.11% | 23.28% | 21.05 Hz |
| Audience SPL RMS error | 0.143 dB | 1.303 dB | 1.802 dB | 21.05 Hz |
| Audience SPL maximum point error | 0.180 dB | 1.348 dB | 1.847 dB | 21.05 Hz |
| Diaphragm-velocity relative error | 0.45% | 5.32% | 8.38% | 38.82 Hz |
| Voice-coil-current relative error | 0.64% | 2.81% | 4.43% | 46.42 Hz |
| Schur GMRES relative residual | 6.74e-5 | 9.55e-5 | 9.98e-5 | 29.32 Hz |

The rank-32 model therefore does not close a tight full-band acceptance gate.
Its main weakness is localized to the bottom octave: 20-25.16 Hz has 0.62-1.80
dB RMS SPL error and 7.50-23.28% complex-field error. At 25.81 Hz the RMS SPL
error falls to 0.393 dB, and from 26.48 through 250 Hz it remains below 0.40 dB.
The 250 Hz endpoint is 0.295 dB RMS and 3.08% complex-field error. The terminal
and diaphragm quantities are substantially better than the low-frequency field,
which points to insufficient pressure-feedback/output subspace coverage rather
than an affine-drive-map error.

A full 100-frequency, two-cabinet exact oracle was initially stopped at a
two-hour cutoff, then rerun overnight without a timeout. The completed result is
reported below.

The full one-cabinet report is
`runs/s218bp_rom_rank32_20_250_fullband_validation.json`. The associated package
is `runs/s218bp_rom_rank32_20_250_100.blabsp`; both are generated artifacts and
remain outside version control. The reusable validator is
`scripts/validate_deploy_speaker_rom_band.py`.

## Completed two-cabinet full-band validation

The two-cabinet exact oracle completed all 100 frequencies in 26,327.8 seconds
(7 h 18 min 48 s). The scene uses 1.5 m cabinet spacing and coherent in-phase
drive. The final report contains exactly the same frequency grid as the
one-cabinet result.

| Quantity | Median | p95 | Maximum | Worst frequency |
| --- | ---: | ---: | ---: | ---: |
| Complex audience-field relative error | 2.02% | 18.23% | 23.43% | 21.05 Hz |
| Audience SPL RMS error | 0.142 dB | 1.299 dB | 1.803 dB | 21.05 Hz |
| Audience SPL maximum point error | 0.204 dB | 1.341 dB | 1.845 dB | 21.05 Hz |
| Diaphragm-velocity relative error | 0.46% | 4.88% | 7.59% | 38.82 Hz |
| Voice-coil-current relative error | 0.61% | 2.90% | 4.04% | 46.42 Hz |
| Schur GMRES relative residual | 6.84e-5 | 9.60e-5 | 9.97e-5 | 119.30 Hz |

The first cold ROM solve took 29.16 seconds. Excluding it, the 99 warmed solves
had a 1.984 second median, 2.148 second p95, and 2.820 second maximum. GMRES used
a median of 7 and a maximum of 8 iterations. This is comfortably inside the
one-minute interaction target for two cabinets.

The exact array data confirms material mutual loading. Relative to the isolated
cabinet, the complex diaphragm state changes by up to 26.2% at 81.36 Hz and the
complex coil-current state changes by up to 21.0% at 47.62 Hz. At those loading
regions, the ROM has 4.87% diaphragm error and 3.64% current error,
respectively, while audience SPL RMS error is 0.127 dB and 0.077 dB. Thus the
model is responding to non-trivial array feedback rather than reproducing two
uncoupled source solutions.

The two-cabinet error envelope is nearly identical to the one-cabinet envelope:
median SPL RMS changes from 0.143 to 0.142 dB, median complex-field error from
2.05% to 2.02%, and worst SPL RMS from 1.802 to 1.803 dB. Array coupling at this
spacing therefore does not amplify the rank-32 approximation error. The same
20-25 Hz deficiency remains the limiting issue. This result does not yet cover
other spacings, independently phased drives, or full-band four- and
eight-cabinet exact oracles.

The completed generated report is
`runs/s218bp_rom_rank32_20_250_fullband_validation_2cab.json`, and its exact
cache is retained under `runs/.deploy_rom_band_validation/2-cabinets/`.
