# Boundary Lab

<img src="assets/mainwindow.png" alt="Boundary Lab main window" width="700">

Boundary Lab is a GUI-based multiphysics acoustic simulation tool for loudspeaker design. It generates or imports loudspeaker meshes, infers exterior BEM, interior FEM, or coupled FEM-BEM-LEM solving from the configured physical system, and presents acoustic and electroacoustic results in the desktop application. Ath is the bundled geometry-generator provider.

### [Follow the official development thread on DIYAudio](https://www.diyaudio.com/community/threads/boundary-lab.440847/)

## Features

- Waveguide design editor with one-click geometry generation through the bundled [Ath4](https://at-horns.eu/) generator
- 3D mesh viewport for generated geometry and imported `.msh` files
- Physical-system editor for exterior BEM, interior FEM, and coupled FEM-BEM-LEM models
- Prescribed-velocity and linear electrodynamic transducer components
- Channel controls for level, polarity, delay, and HPF/LPF crossover shaping
- Live horizontal/vertical directivity, on-axis response, spinorama, excursion, maximum-SPL, and impedance plots
- Plot-image, polar-data, on-axis channel-data, and balloon-data export
- 3D balloon viewer built directly from Fibonacci-sphere solve samples
- Project save/load with readable, backward-compatible `.blab.json` files

While not required, if modeling in Autodesk Fusion, the [Fusion2Msh](https://github.com/JWSound/fusiontomsh) add-in is strongly recommended for quick imports of mesh files into Boundary Lab.

## Windows quick start

1. Double-click `01_install_update_boundary-lab.bat` in the repository folder.
2. Follow the guided prompts. The installer creates the Python environment and
   can optionally prepare the Julia BEAT Engine CPU and NVIDIA CUDA solvers.
3. If the installer adds Git, Python, or Julia, close it and run it again when
   instructed so Windows can refresh the available commands.
4. Double-click `02_start_boundary_lab.bat` to launch Boundary Lab.

While not required, if modeling in Autodesk Fusion, the [Fusion2Msh](https://github.com/JWSound/fusiontomsh) add-in is strongly recommended for quick imports of models into Boundary Lab.

## Solver Requirements

Boundary Lab currently has five selectable BEM solver backends in application
preferences: Server, BEAT Engine Nvidia CUDA, BEAT Engine CPU, BEAT Engine AMD
ROCm, and Bempp OpenCL CPU.

### BEAT Engine CUDA GPU Solver Requirements

* NVIDIA Maxwell-generation or newer GPU
* Latest NVIDIA Studio/Game Ready driver recommended
* [Julia](https://julialang.org/downloads/manual-downloads/) installed and available on `PATH`

To prepare the Julia environment, from the repository root run:

```bash
julia --project=src/blab/solvers/julia_cuda -e "using Pkg; Pkg.instantiate()"
```


GPU solving VRAM requirements scale quadratically with mesh element count. Below are estimated VRAM requirements for various element counts:

| Total Elements | Estimated VRAM |
|---:|---:|
| 1,000 | ~50-100 MB |
| 2,000 | ~200-300 MB |
| 3,000 | ~400-600 MB |
| 5,000 | ~1.0-1.5 GB |
| 7,000 | ~2.0-3.0 GB |
| 10,000 | ~4-6 GB |
| 15,000 | ~8-12 GB |
| 20,000 | ~14-20 GB |

### BEAT Engine CPU Solver Requirements

* Intel, AMD, or ARM CPU
* [Julia](https://julialang.org/downloads/manual-downloads/) installed and available on `PATH`

To prepare the Julia environment, from the repository root run:

```bash
julia --project=src/blab/solvers/julia_local -e "using Pkg; Pkg.instantiate()"
```

### Bempp CPU Solver Requirements

* Intel or AMD CPU
* An OpenCL runtime

The [Intel CPU OpenCL runtime](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html) is a practical option even on many non-Intel systems.

## Application Installation

From the repository root run:

```bash
python -m pip install -e ".[gui]"
```

## Run The GUI

```bash
blab gui
```

## Boundary Lab Deploy prototype

The first TypeScript, Three.js, and Electron deployment prototype lives in
[`deploy`](deploy/README.md). It starts with the S218BP subwoofer, opens
`.blabsp` packages, evaluates the Level 1 complex pattern field on an
interactive audience plane, and can run a single-frequency Level 2 fixed-source
exterior solve through a persistent BEAT CUDA worker. New projects use the
coarse `deploy/library/S218BP_LOD.blabsp` package.

Boundary Lab captures Ath blab-mode geometry and writes managed generated
artifacts into:

```text
runs/generated_geometry
```


## Boundary Lab Server

Boundary Lab can also run a local or LAN-accessible job server that accepts solve
jobs and streams per-frequency results back as NDJSON events:

```bash
blab server --host 127.0.0.1 --port 8765 --solver bempp_cpu
blab server --host 127.0.0.1 --port 8765 --solver beat_cpu --julia-threads auto
blab server --host 127.0.0.1 --port 8765 --solver beat_cuda
```

Supported server-side solver IDs are `bempp_cpu` for Bempp OpenCL CPU, `beat_cpu`,
`beat_cuda`, and `beat_rocm`. ROCm supports exterior and coupled FEM-BEM solves,
including symmetry and hybrid FEM static condensation. For BEAT Engine solvers, use
`--julia-executable` and `--julia-threads` to point the server at the intended
Julia installation and thread count.

To use it from the GUI application, open `Edit > Preferences`, set `BEM Solver` to
`Server`, and set `Solve Server URL` to the server address. Use `Check Server` to
query `/health`; the app uses the advertised capabilities, such as mesh
symmetry support for feature availability. For another machine on the LAN, bind
the server to that machine's LAN address or `0.0.0.0` and use
`http://<server-ip>:8765` in the client. The GUI uploads the solver mesh files
with each server job, so the server does not need access to the client's local
paths.

For Docker image deployment with the BEAT Engine CUDA solver, see
[Docker](docs/Docker.md).

## Documentation

- [Installation and Setup](docs/Installation%20and%20Setup.md)
- [User Guide](docs/User%20Guide.md)
- [Physical System Model](docs/Physical%20System%20Model.md)
- [Interior FEM Solver](docs/Interior%20FEM%20Solver.md)
- [Coupled Solver](docs/Coupled%20Solver.md)
- [Boundary Lab Server](docs/Boundary%20Lab%20Server.md)
- [CUDA Server Docker Image](docs/Docker.md)
- [Model Assumptions](docs/Model%20Assumptions.md)
- [Inputs and Outputs](docs/Inputs%20and%20Outputs.md)
- [Advanced CLI workflow](docs/advanced/cli-workflow.md)
- [BEAT Engine Core](docs/advanced/beat-engine-core.md)
- [BEAT Engine CPU](docs/advanced/beat-engine-CPU.md)
- [BEAT Engine AMD ROCm](docs/advanced/beat-engine-rocm.md)
- [BEAT Engine CUDA](docs/advanced/beat-engine-CUDA.md)
- [Forward Beam Shape plot](docs/advanced/forward-beam-shape.md)
