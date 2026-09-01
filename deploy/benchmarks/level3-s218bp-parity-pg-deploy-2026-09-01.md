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
