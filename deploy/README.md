# Boundary Lab Deploy prototype

Boundary Lab Deploy is a desktop prototype for interactive subwoofer placement and analysis. It uses Electron for the desktop shell and a React, TypeScript, and Three.js renderer.

## Run it

From this directory:

```powershell
npm install
npm run dev
```

To run the built desktop renderer:

```powershell
npm run build
npm start
```

## Current vertical slice

- Imports multiple Boundary Lab speaker-package schema v1 `.blabsp` archives into a project library without replacing the scene.
- Reads the package manifest, complex spherical pressure, frequency order, excitation shape, and exterior Gmsh surface.
- Opens new desktop projects with two coarse `S218BP_LOD.blabsp` cabinets separated by a 2 m surface gap.
- Provides source position, yaw, level, delay, and polarity controls without line-array layout concepts.
- Displays eight bounding-box grab points on selected speaker and rigid objects for strictly ground-parallel dragging; only a successful snap to a corner at another height introduces vertical movement.
- Provides W-key XYZ translation and E-key pitch/yaw/roll rotation gizmos with axis-only X/Y/Z rotation wheels and 5-degree snapping; hold Alt for unsnapped rotation. A near-gizmo overlay reports signed movement to 0.001 m or the active rotation to whole degrees while dragging.
- Adds or duplicates package-backed speaker instances while preserving independent placement and DSP settings.
- Imports closed, consistently oriented Gmsh 2.2 ASCII triangle meshes as reusable rigid-boundary assets, using Boundary Lab's default millimetre mesh units (`0.001` mesh-to-metre scale). Rigid objects share cabinet selection, eight-corner handles, W/E movement and rotation, ground enforcement, and 10 mm surface padding, but are ignored by Level 1.
- Complex-sums Level 1 pattern pressure from mixed package types on an editable audience plane, with complex frequency interpolation, using Boundary Lab's `exp(-i omega t)` convention.
- Renders the speaker meshes and SPL surface in an orbitable Three.js scene.
- Treats the audience plane as a scene-list-selectable object with unrestricted position and pitch/yaw/roll, W/E transform gizmos, asymmetrical R-key corner resizing, and sparse above-ground sampling.
- Adds translation-only microphone point probes with one direct-drag handle and a W-key XYZ gizmo.
- Plots every microphone's package-derived SPL response across the exact exported frequency grid.
- Calculates explicit complex microphone pressure across the package grid for both Level 2 exterior BEM and Level 3 parity-ROM coupled solves. ROM sweeps retain exterior geometry while selecting each frequency's reduced operators. Both paths stream progress and turn the Calculate button into a Stop control while active.
- Reuses the Level 3 sweep to plot peak diaphragm excursion (`sqrt(2) |v| / 2πf`) from RMS velocity as one progressively updated line per scene transducer; the excursion sweep does not require a microphone probe.
- Adds a cabinet-level Electrical tab with switchable impedance magnitude/phase, RMS current, and real input-power plots derived from each speaker object's applied complex RMS voltage and summed complex coil current.
- Runs an explicit single-frequency, multi-cabinet Level 2 exterior solve with prescribed speaker Neumann traces, zero-Neumann rigid objects, and an always-on rigid Y=0 half-space Green's function through a persistent BEAT CUDA worker.
- Schur-eliminates Level 3 parity-sector ROMs into the shared exterior BEM solve so cabinet loading and transducer feedback respond to the complete array.
- Provides a play/pause live-solve mode that debounces scene edits and follows an in-flight solve with the newest scene revision.
- Streams solve status back to the renderer and only displays a boundary result while it matches the current scene revision.
- Retains separate current observation-plane frames for Boundary and Coupled fidelity so users can compare solver levels without repeating unchanged solves.
- Keeps speaker and rigid geometry above the ground plane, omits below-ground audience samples, and reserves 10 mm between all boundary-object surfaces for stable close-pair quadrature.
- Uses threshold-oriented triangle-BVH clearance validation with early exit and emits conservative higher-order corrections for close speaker/rigid face pairs and their ground images.
- Saves speaker packages, rigid-mesh assets and instances, microphones, and observation-plane display settings as a schema-v6 `.blabdeploy.json` project (schema v5 remains loadable).
- Includes a deterministic built-in demonstration model when no package is loaded.

Boundary fidelity is available in the desktop app when every active source uses the same Level 2 package loaded from disk and the selected frequency was exported by that package. Coupled fidelity is enabled for a parity Petrov–Galerkin Level 3 package under the same homogeneous-scene constraints. Mixed-package scenes and browser-only sessions currently use the Level 1 preview.

The Level 2 worker uses `BLAB_PYTHON_EXE` and `BLAB_JULIA_EXE` when set; otherwise it resolves `python` and `julia` from `PATH`. The current slice uses a globally reflective rigid ground plane, supports multiple instances of one fixed-source package, and requires an exact exported frequency.

## Verification

```powershell
npm run test:package
npm run build
npm run test:desktop
npm run test:level2
```

`test:package` loads the repository's S218BP LOD package and rigid-stage fixture, verifies their mesh/project contracts, and computes a finite observation-plane preview. `test:desktop` loads the production renderer in a hidden Electron window and checks the application shell, WebGL canvas, rigid-mesh import and manipulation, microphone workflow, and project loading without runtime console errors. `test:level2` is the slower NVIDIA/CUDA integration smoke test; it crosses the Electron IPC, Python combined-boundary preparation, persistent Julia worker, BEAT CUDA solve, and renderer result path with a rigid stage present, then moves the cabinet and verifies that live solving refreshes the result.

For a timestamped cold/warm movement and 200 x 200 plane benchmark, run
`npm run benchmark:level2`. The harness prints one JSON record containing
Julia, Python, Electron IPC, field-frame parsing, and heatmap rasterization
timings. A reference run and interpretation are recorded in
[`benchmarks/level2-pipeline-2026-08-26.md`](benchmarks/level2-pipeline-2026-08-26.md).

## Level 3 validation history

The retired full-order Level 3 baseline and the performance gate that motivated
the parity-ROM implementation are recorded in
[`benchmarks/level3-s218bp-2026-09-01.md`](benchmarks/level3-s218bp-2026-09-01.md).
Exact-system execution remains available only to standalone developer validation
scripts as a ROM accuracy oracle.
