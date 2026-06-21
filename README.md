# Boundary Lab

<img src="assets/mainwindow.png" alt="Script Editor" width="500">

Boundary Lab is a GUI-based Boundary Element Method (BEM) tool for loudspeaker design. It uses Ath to generate loudspeaker surface meshes, runs BEM solves, and shows SPL, directivity, radiation impedance, spinorama-style curves, and 3D balloon plots inside the desktop application.

## Features

- [Ath4](https://at-horns.eu/) `.cfg` editor with one-click geometry generation
- 3D mesh viewport for generated Ath meshes and imported `.msh` files
- Multi-mesh and multi-radiator BEM solves
- Source controls for level, polarity, delay, and HPF/LPF crossover shaping
- Live horizontal/vertical directivity, on-axis response, spinorama, and impedance plots
- dB/phase exporting into .txt files
- 3D balloon plot viewer with spherical sampling
- Project save/load with readable `.blab.json` files

## Base Requirements

- Windows 10/11 64-bit
- Python 3.11 or newer


While not required, if modeling in Autodesk Fusion, the [Fusion2Msh](https://github.com/JWSound/fusiontomsh) add-in is strongly recommended for quick imports of models into Boundary Lab.

## Solver Requirements

Boundary Lab currently has 3 selectable BEM solver backends in the application preferences menu. Solve speed is dependant on hardware, but typically GPU-based solving is the fastest option if available.

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

##

### BEAT Engine CPU Solver Requirements

* Intel, AMD, or ARM CPU
* [Julia](https://julialang.org/downloads/manual-downloads/) installed and available on `PATH`

To prepare the Julia environment, from the repository root run:

```bash
julia --project=src/blab/solvers/julia_local -e "using Pkg; Pkg.instantiate()"
```

##

### Bempp CPU Solver Requirements

* Intel or AMD CPU
* An OpenCL runtime

The [Intel CPU OpenCL runtime](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html) is a practical option even on many non-Intel systems.

##

## Application Installation

From the repository root run:

```bash
python -m pip install -e ".[gui]"
```

## Run The GUI

```bash
blab gui
```

On startup, Boundary Lab updates `ath/ath.cfg` so Ath writes generated files into:

```text
runs/ath_output
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
`beat_cuda`, and `beat_rocm`. ROCm is accepted as a server selector but the ROCm
BEAT Engine implementation is still a placeholder and will report not implemented
until that engine path is completed. For BEAT Engine solvers, use
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

Server API:

- `POST /jobs` submits a solve request with `SimulationConfig` and `frequencies_hz`.
- `GET /jobs/{job_id}` returns job status and artifact links.
- `GET /jobs/{job_id}/events?since=0` streams job events as newline-delimited JSON.
- `POST /jobs/{job_id}/cancel` requests cancellation.
- `GET /jobs/{job_id}/artifacts/result.npz` downloads the completed result bundle.

## Documentation

- [User Guide](docs/User%20Guide.md)
- [Model Assumptions](docs/Model%20Assumptions.md)
- [Inputs and Outputs](docs/Inputs%20and%20Outputs.md)
- [Advanced CLI workflow](docs/advanced/cli-workflow.md)
- [BEAT Engine Core](docs/advanced/beat-engine-core.md)
- [BEAT Engine CPU](docs/advanced/beat-engine-CPU.md)
- [BEAT Engine CUDA](docs/advanced/beat-engine-CUDA.md)
