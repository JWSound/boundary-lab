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
- Displays eight bounding-box grab points on the selected cabinet for ground-parallel dragging; snapping a handle to a cabinet corner at another height enables vertical stacks.
- Provides W-key XYZ translation and E-key pitch/yaw/roll rotation gizmos with 5-degree snapping; hold Alt for free rotation.
- Adds or duplicates package-backed speaker instances while preserving independent placement and DSP settings.
- Complex-sums Level 1 pattern pressure from mixed package types on an editable audience plane, with complex frequency interpolation, using Boundary Lab's `exp(-i omega t)` convention.
- Renders the speaker meshes and SPL surface in an orbitable Three.js scene.
- Treats the audience plane as a scene-list-selectable object with unrestricted position and pitch/yaw/roll, W/E transform gizmos, asymmetrical R-key corner resizing, and sparse above-ground sampling.
- Adds translation-only microphone point probes with one direct-drag handle and a W-key XYZ gizmo.
- Plots every microphone's package-derived SPL response across the exact exported frequency grid.
- Calculates explicit Level 2 complex microphone pressure across the full package grid, streams progress, and turns the Calculate button into a Stop control while the sweep is active.
- Runs an explicit single-frequency, multi-cabinet Level 2 fixed-Neumann exterior solve with an always-on rigid Y=0 half-space Green's function through a persistent BEAT CUDA worker.
- Provides a play/pause live-solve mode that debounces scene edits and follows an in-flight solve with the newest scene revision.
- Streams solve status back to the renderer and only displays a boundary result while it matches the current scene revision.
- Keeps cabinet geometry above the ground plane, omits below-ground audience samples, and reserves 10 mm between cabinet surfaces for stable close-pair quadrature.
- Uses a triangle BVH to measure exact inter-cabinet surface spacing before a solve.
- Saves the package registry and per-source package references, including microphones, as a strict schema-v4 `.blabdeploy.json` project.
- Includes a deterministic built-in demonstration model when no package is loaded.

Boundary fidelity is available in the desktop app when every active source uses the same Level 2 package loaded from disk and the selected frequency was exported by that package. Mixed-package scenes currently use the Level 1 preview. Coupled fidelity remains unavailable, and browser-only sessions retain Level 1.

The Level 2 worker uses `BLAB_PYTHON_EXE` and `BLAB_JULIA_EXE` when set; otherwise it resolves `python` and `julia` from `PATH`. The current slice uses a globally reflective rigid ground plane, supports multiple instances of one fixed-source package, and requires an exact exported frequency.

## Verification

```powershell
npm run test:package
npm run build
npm run test:desktop
npm run test:level2
```

`test:package` loads the repository's S218BP LOD package, verifies its real exterior mesh and pattern arrays, and computes a finite observation-plane preview. `test:desktop` loads the production renderer in a hidden Electron window and checks the application shell, WebGL canvas, microphone creation and manipulation, response trace, and project loading without runtime console errors. `test:level2` is the slower NVIDIA/CUDA integration smoke test; it crosses the Electron IPC, Python preparation, persistent Julia worker, BEAT CUDA solve, and renderer result path, then moves the cabinet and verifies that live solving refreshes the result.

For a timestamped cold/warm movement and 200 x 200 plane benchmark, run
`npm run benchmark:level2`. The harness prints one JSON record containing
Julia, Python, Electron IPC, field-frame parsing, and heatmap rasterization
timings. A reference run and interpretation are recorded in
[`benchmarks/level2-pipeline-2026-08-26.md`](benchmarks/level2-pipeline-2026-08-26.md).

## Next solver milestone

Reuse the rigid-ground and close-pair geometry caches across live solves, then route the future Level 3 condensed-interior exterior operator through the same half-space path.
