# Boundary Lab Deploy — System Model

[User Guide](user-guide.md) · [Setup](../README.md)

This document describes the current Deploy calculation paths, not a replacement
for the main solver documentation. Start with Boundary Lab's
[Physical System Model](../../docs/Physical%20System%20Model.md),
[Model Assumptions](../../docs/Model%20Assumptions.md),
[BEAT Engine Core](../../docs/advanced/beat-engine-core.md), and
[Coupled Solver](../../docs/Coupled%20Solver.md) for the underlying formulations.

## Inputs and execution

The renderer evaluates Pattern responses in TypeScript. Electron starts the
Qt-free Python Deploy worker, which prepares geometry and package operators and
submits Boundary/Coupled requests to a persistent Julia BEAT worker. The current
desktop renderer requests CUDA. Worker APIs having a CPU option does not imply
an exposed or automatic desktop CPU fallback.

Speaker-package schema v1 contains sampled complex spherical pressure (Level 1),
exterior geometry and pressure/normal-derivative traces (Level 2), and, for the
supported Level 3 path, frequency-indexed parity Petrov–Galerkin ROM operators.
See [Inputs and Outputs](../../docs/Inputs%20and%20Outputs.md) for archive and
coordinate contracts. Deploy supports multiple instances of one disk-backed
package for Boundary/Coupled; mixed-package scenes are currently Pattern-only.
New packages do not include an isolated acoustic-impedance reference. Legacy
optional reference members are ignored.

## Conventions and drive

Distances are metres, frequencies Hz, pressures Pa, electrical inputs RMS, and
angles in the UI degrees. Scene coordinates use Y up and ground Y = 0. The source
local scene vector maps to package coordinates as `(x, y, z) -> (x, z, -y)` after
inverse cabinet rotation. Cabinet rotations use Three.js Euler order YXZ.

| Quantity/path | Convention |
| --- | --- |
| Native pressure, current, velocity, and propagation | `exp(-iωt)`; outgoing radial phase `exp(+ikr)` |
| Positive drive delay τ | Multiplier `exp(+iωτ)` in the native convention |
| Electrical impedance phase display | Native `arg(V/I)`; no conjugation in the current renderer |
| Acoustic resistance/reactance display | Standard-audio `exp(+iωt)`: native impedance is conjugated |
| Stored diaphragm differential pressure | Complex RMS pressure converted to `exp(+iωt)` |

This phase-display difference is current behavior, not a common-sign convention
across all plots. Magnitudes and real power are unaffected by conjugation.

Channel and speaker levels/delays add; polarities multiply. Channel mute zeros
the drive. EQ filters are stored UI/project structures but are not applied.
For effective level L, polarity σ, and delay τ:

`g(f) = σ · 10^(L/20) · exp(+i2πfτ)`

Pattern/Boundary multiply their stored package responses by g. Pattern and fixed
traces use the selected logical excitation (default first excitation), grouping
symmetry-expanded ports where the package maps them to the same logical source.
The parity-ROM Coupled request instead applies a common `2.83 g` V RMS default
to each input port. This is not an independently configurable multi-port drive
matrix in the UI. Check export excitation provenance before interpreting absolute
differences between methods as loading alone.

## Pattern method

For each source/receiver pair, inverse-rotate the ray into cabinet coordinates
and sample the package's complex directional pressure. Microphones use the
nearest exported direction by dot product. Audience maps use a precomputed
72-azimuth × 37-elevation nearest-direction lookup, then nearest lookup-bin
sampling. This extra angular discretization can produce small microphone/map
differences for directional patterns.

At distance r, using sample reference radius r₀:

`p(r,f) = p(r₀, direction, f) · (r₀/r) · exp[ik(r-r₀)] · g(f)`

Distances are floored at 0.02 m to avoid division by zero. This is a numerical
guard, not a valid near-field source model. The `1/r` propagation extrapolates a
sampled spherical response; its validity depends on the package and observation
distance. Direct and image paths inside the sample radius increment the
near-field diagnostic. Counts are path contributions, not necessarily distinct
receivers or a percentage of the grid.

For mixed packages, microphone frequencies are the sorted union of exported
frequencies inside the common band. Pressure real/imaginary components are
linearly interpolated between bracketing package frequencies. The map builds
interpolated lookups at its selected frequency. Neither operation reconstructs
unresolved spectral features or interpolates a new Boundary/Coupled solution.

### Rigid ground

Let M(x,y,z) = (x,-y,z). For each source, evaluate the free-field pressure at both
the receiver and its reflection, applying inverse source rotation separately:

`pground(x) = Σs [ps(x) + ps(Mx)]`

The image coefficient is +1. Add complex pressures, not SPL values; do not
conjugate the image. This mirrors directional radiation correctly for tilted
and asymmetric cabinets. At y=0 the two terms coincide. Elevated receivers
have different path lengths and launch directions. No additional cabinet/scene
scattering or ground-induced motion feedback is introduced into Pattern.

## Boundary method

Python transforms complete exterior package meshes into a shared scene mesh.
It scales the packaged normal derivative `q = ∂p/∂n` by drive and imposes q on
speaker surfaces. Added rigid objects have q=0. The shared exterior pressure is
then solved; packaged reference pressure is carried for reference/diagnostics,
not simultaneously prescribed as a second boundary condition.

The Julia path uses continuous P1 pressure and facewise DP0 normal derivative
with a Burton–Miller Galerkin exterior formulation. The always-on rigid y=0
half-space kernel supplies positive ground images. Geometry stays in the physical
upper half-space; the ground is not a finite meshed rectangle. Singular and close
interactions, including image pairs, receive specialized quadrature treatment.
Field pressure is evaluated from the solved pressure and prescribed q.

This responds to exterior scattering, but keeps source q fixed; it does not update
interior pressures, coil current, or diaphragm motion. It should not be described
as a full coupled transducer calculation. For operators, sign conventions and
quadrature, use [BEAT Engine Core](../../docs/advanced/beat-engine-core.md) and
[CUDA implementation](../../docs/advanced/beat-engine-CUDA.md).

## Coupled method

Supported Level 3 packages provide frequency-specific reduced operators by
parity sector: reduced system/drive/boundary maps and output maps for velocity,
current, and boundary normal derivative. Sector signs and node/face orbits expand
the reduced response consistently onto each complete cabinet instance. Rank comes
from package metadata; a default export rank is not a guarantee of exactness.

At a given frequency, the cabinet response is linear in its electrical drive and
the exterior pressure applied to its boundary. Schematically:

`q = qdrive + T p`

If the exterior discretization is written `A p = B q`, eliminating the cabinet
response gives `(A - B T)p = B qdrive`. Here T denotes the assembled reduced
feedback map, not a new dense full-interior solve. The actual code applies this
Schur feedback through the ROM. The CUDA exterior system is factorized for
preconditioning, and GMRES solves the feedback problem. Current request defaults
are relative tolerance 1e-4 and at most 30 iterations. Residual/history and other
diagnostics are returned; those tolerances do not bound ROM approximation error.

Final per-transducer complex velocity/current come from the reduced output maps
at the converged exterior pressure. This includes mutual loading and ground/scene
feedback within the packaged linear model. It does not rerun the full interior
FEM model or include thermal/nonlinear/structural-damage physics. See
[Interior FEM Solver](../../docs/Interior%20FEM%20Solver.md) and
[Coupled Solver](../../docs/Coupled%20Solver.md) for model construction.

The Deploy exterior path uses Float32/ComplexF32. Package frequency identifiers
and rounded solver frequencies are matched before acoustic postprocessing.
Boundary uses the exported fixed-trace grid; Coupled sweeps are restricted to
frequencies supported by its ROM. Reduced operators are not interpolated by the UI.

## Derived plots

Let ω=2πf; I and v are a transducer's complex RMS current and velocity. With the
native convention, mechanical impedance is:

`Zm = Rms + i[1/(ω Cms) - ω Mmd]`

Mmd is the model's mechanical moving mass, not an independently substituted
air-loaded Mms. Recover net opposing acoustic force by force balance:

`Fload = Bl I - Zm v`

| Output | Calculation |
| --- | --- |
| Microphone/map SPL | `20 log10(|p| / 20 µPa)` |
| Peak excursion (mm) | `1000 sqrt(2) |v| / ω` |
| Cabinet current | Sum coil-current phasors, with physical-driver orbit multiplicities; then take magnitude |
| Cabinet electrical impedance | `Vcabinet / Icabinet`, under the current common-voltage drive assumption |
| Real input power | `Re(Vcabinet conj(Icabinet))`; RMS convention means no additional factor 1/2 |
| Normalized acoustic impedance | `conj(Fload/v) / (ρ c Sd)`; plot real or imaginary part |
| Differential pressure | `conj(Fload/Sd)` stored as complex RMS Pa; plot magnitude, optionally ×sqrt(2) for sinusoidal peak or ÷1000 for kPa |

Sd is effective area from compiled package normalization metadata, generally
projected area for rigid translation, not the cone's sloping surface area. Acoustic
load includes both sides of the diaphragm, not just exterior radiation. Pressure
is a force-equivalent spatial differential, not a separate front/back measurement
or a local maximum. Opposing spatial loads can cancel in this average.

Acoustic impedance is omitted for invalid normalization or velocity below
`max(1e-12 m/s, 1e-8 × maximum finite cabinet velocity magnitude)`. Pressure is
computed before this division and can remain finite at zero velocity. Missing
data are gaps, not zero; a finite zero differential remains valid. Electrical
impedance is undefined at vanishing summed current. Negative active acoustic
resistance and negative real electrical power can occur in an interacting driven
model and are not standalone equipment-risk classifications.

Spatial average SPL is the arithmetic mean of valid sample dB values; spread is
P90 minus P10 using discrete sample order statistics. Neither is energy averaging,
area integration, audience weighting, or broadband averaging. Below-ground
audience samples are masked; display range/banding does not alter the solution.

## State, comparisons, and reproducibility

Field results are keyed to the scene/frequency and kept separately for Boundary
and Coupled. Live solving debounces edits and follows an in-flight solve with the
latest requested state. Microphone/speaker sweeps are separate operations and
stream results per frequency; scene edits invalidate current matching results.
Solver switching alone can retain results whose keys still match.

Captures clone the available completed sweep data and configuration, preserving
original frequency grids. They do not resimulate or normalize comparisons to equal
power. Single-cabinet comparisons use explicit one-cabinet scene captures, with
the same rigid-ground assumption as other scenes. There is no automatic isolated
free-field reference.

Project schema v7 stores configuration and external asset references; v5/v6 are
accepted. It does not embed result caches or captures. Capture downloads use
`.blabanalysis.json`, serializing arrays/maps and missing numeric values as JSON
nulls. Import is not implemented. For reproducibility, retain the referenced
packages/meshes and their versions as well as configuration and downloaded data.

## Implementation and verification

Primary implementation anchors:

- [Pattern and spatial statistics](../src/model/field.ts), [drive composition](../src/model/channels.ts).
- [Request preparation and package cache](../../src/blab/deploy_solve.py), [streaming worker](../../src/blab/deploy_worker.py).
- [Deploy exterior/Schur solver](https://github.com/JWSound/BEAT_Engine/blob/v0.1.0/src/beat_engine/julia_local/deploy_solver.jl), [ROM construction](https://github.com/JWSound/BEAT_Engine/blob/v0.1.0/src/beat_engine/julia_local/src/BeatEngineSpeakerROM.jl).
- [Acoustic postprocessing](../../src/blab/deploy_acoustic_loading.py), [electrical postprocessing/UI](../src/App.tsx), [captures](../src/model/analysisCapture.ts).

From `deploy/`, use `npm run test:pattern`, `npm run test:analysis`,
`npm run test:channels`, `npm run test:placement`, `npm run test:package`,
`npm run build`, and `npm run test:desktop`. The desktop smoke exercises UI and
transport interactions; it is not a full numerical convergence study.
`npm run test:level2` is the slower CUDA integration smoke.
`npm run benchmark:level2` measures cold/warm field and movement performance;
see the [recorded benchmark](../benchmarks/level2-pipeline-2026-08-26.md).
Python tests for `deploy_solve`, `deploy_worker`, `deploy_acoustic_loading`, and
`speaker_package` exercise the shared headless contracts.

Mesh/ROM convergence, package excitation consistency, angular/frequency sampling,
and scene clearance must be assessed independently of passing software tests.
For export-side validation and reproducible headless runs, use the main
[CLI workflow](../../docs/advanced/cli-workflow.md).
