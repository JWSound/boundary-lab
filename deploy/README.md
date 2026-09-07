# Boundary Lab Deploy

Boundary Lab Deploy is a desktop prototype for loudspeaker-array placement,
coverage prediction, and comparative loading analysis using Boundary Lab speaker
packages. It offers immediate Pattern predictions and higher-detail Boundary and
reduced-order Coupled calculations, all with an infinite rigid ground plane.

## Requirements

- A checkout of the Boundary Lab repository and Node.js with npm (the current
  Vite toolchain accepts Node 18.x or 20+). Electron is installed by npm.
- A graphics environment capable of running Electron/Three.js with WebGL.
- For Boundary/Coupled: Python 3.11+ with Boundary Lab's dependencies installed,
  Julia with the BEAT Engine CUDA environment prepared, and a compatible NVIDIA
  GPU/driver. Follow [Boundary Lab setup](../docs/Installation%20and%20Setup.md)
  and [BEAT Engine CUDA](../docs/advanced/beat-engine-CUDA.md).
- A suitable disk-backed `.blabsp` package for numerical solves. A bundled coarse
  S218BP example is provided for exploring the workflow.

Pattern evaluation does not require the Python/Julia solver runtime. The desktop
currently requests CUDA for Boundary/Coupled; it does not automatically fall back
to CPU. Opening only the browser renderer provides Pattern preview, not desktop
file/solver integration.

## Install and run

From the repository root:

```powershell
cd deploy
npm install
npm run build
npm start
```

For development, use `npm run dev` instead of build/start; it launches the web
development server and Electron together.

For numerical solves, make the installed Python and Julia executables available
on PATH, or set explicit executable paths before launching Deploy. For example,
from `deploy/` in PowerShell after preparing the repository's Python environment:

```powershell
$env:BLAB_PYTHON_EXE = (Resolve-Path ..\.venv\Scripts\python.exe).Path
$env:BLAB_JULIA_EXE = "C:\path\to\julia.exe"
npm start
```

Replace the Julia placeholder with the installed executable. The Python process
runs from the repository root with `src` on its module path; it still needs the
installed dependencies. If unset, the executable defaults are `python` and
`julia`.

## Documentation

- [User Guide](docs/user-guide.md) — scene setup, drive controls, maps, plots,
  comparisons, saving, and practical limitations.
- [System Model](docs/system-model.md) — numerical paths, conventions, derived
  quantities, approximation limits, implementation references, and verification.
