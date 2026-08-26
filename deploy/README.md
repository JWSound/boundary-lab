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

- Opens Boundary Lab speaker-package schema v1 `.blabsp` archives.
- Reads the package manifest, complex spherical pressure, frequency order, excitation shape, and exterior Gmsh surface.
- Opens new desktop projects with two coarse `S218BP_LOD.blabsp` cabinets separated by a 2 m surface gap.
- Provides source position, yaw, level, delay, and polarity controls without line-array layout concepts.
- Displays eight bounding-box grab points on the selected cabinet for three-dimensional dragging and corner snapping, including vertical cabinet stacks.
- Provides a W-key XYZ translation gizmo and an E-key yaw gizmo with 5-degree snapping; hold Alt for free rotation.
- Complex-sums Level 1 pattern pressure on an editable audience plane using Boundary Lab's `exp(-i omega t)` convention.
- Renders the speaker meshes and SPL surface in an orbitable Three.js scene.
- Treats the audience plane as a scene-list-selectable object with size, position, and inspector resolution controls up to 200 samples and aspect-ratio-aware rows and columns.
- Reports average, peak, and P10-P90 coverage spread while the scene changes.
- Runs an explicit single-frequency, multi-cabinet Level 2 fixed-Neumann exterior solve through a persistent BEAT CUDA worker.
- Provides a play/pause live-solve mode that debounces scene edits and follows an in-flight solve with the newest scene revision.
- Streams solve status back to the renderer and only displays a boundary result while it matches the current scene revision.
- Keeps cabinet geometry and audience planes above the ground plane and reserves 2 mm between cabinet surfaces.
- Uses a triangle BVH to measure exact inter-cabinet surface spacing before a solve.
- Saves the editable scene as a readable `.blabdeploy.json` project.
- Includes a deterministic built-in demonstration model when no package is loaded.

Boundary fidelity is available in the desktop app for Level 2 packages loaded from disk. Coupled fidelity remains unavailable, and browser-only sessions retain the Level 1 preview.

The Level 2 worker uses `BLAB_PYTHON_EXE` and `BLAB_JULIA_EXE` when set; otherwise it resolves `python` and `julia` from `PATH`. The current slice is free-field, supports multiple instances of one fixed-source package, and requires an exact exported frequency.

## Verification

```powershell
npm run test:package
npm run build
npm run test:desktop
npm run test:level2
```

`test:package` loads the repository's S218BP LOD package, verifies its real exterior mesh and pattern arrays, and computes a finite observation-plane preview. `test:desktop` loads the production renderer in a hidden Electron window and checks that the application shell, WebGL canvas, package card, and analysis metrics are present without runtime console errors. `test:level2` is the slower NVIDIA/CUDA integration smoke test; it crosses the Electron IPC, Python preparation, persistent Julia worker, BEAT CUDA solve, and renderer result path, then moves the cabinet and verifies that live solving refreshes the result.

## Next solver milestone

Use the BVH candidates to apply selective close-pair correction quadrature, followed by a rigid half-space ground image.
