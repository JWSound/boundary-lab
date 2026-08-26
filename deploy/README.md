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
- Reads the package manifest, complex spherical pressure, frequency order, excitation shape, and optional exterior Gmsh surface.
- Opens new desktop projects with the repository's S218BP package placed as a single source.
- Provides source position, yaw, level, delay, and polarity controls without line-array layout concepts.
- Complex-sums Level 1 pattern pressure on an editable audience plane using Boundary Lab's `exp(-i omega t)` convention.
- Renders the speaker meshes and SPL surface in an orbitable Three.js scene.
- Reports average, peak, and P10-P90 coverage spread while the scene changes.
- Saves the editable scene as a readable `.blabdeploy.json` project.
- Includes a deterministic built-in demonstration model when no package is loaded.

Boundary and Coupled controls deliberately remain marked as requiring their solve engines. They do not present the Level 1 preview as a higher-fidelity solution.

## Verification

```powershell
npm run test:package
npm run build
npm run test:desktop
```

`test:package` loads the repository's S218BP package, verifies its real exterior mesh and pattern arrays, and computes a finite observation-plane preview. `test:desktop` loads the production renderer in a hidden Electron window and checks that the application shell, WebGL canvas, package card, and analysis metrics are present without runtime console errors.

## Next solver milestone

The next vertical slice is a persistent solve session that accepts the same scene revision and returns a Level 2 single-frequency exterior BEM result. The renderer already distinguishes live Pattern data from unavailable Boundary/Coupled data so integration can remain progressive.
