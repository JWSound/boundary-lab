# Boundary Lab

Boundary Lab is a GUI-based Boundary Element Method (BEM) tool for loudspeaker design. It uses Ath to generate loudspeaker surface meshes, runs BEM solves with `bempp-cl`, and shows SPL, directivity, radiation impedance, spinorama-style curves, and 3D balloon plots inside the desktop application.

## Features

- [Ath4](https://at-horns.eu/) `.cfg` editor with one-click geometry generation
- Mesh preview for generated Ath meshes and imported `.msh` files
- Multi-mesh and multi-radiator BEM solves
- Source controls for level, polarity, delay, and HPF/LPF crossover shaping
- Live horizontal/vertical directivity, on-axis response, spinorama, and impedance plots
- 3D balloon plot viewer with spherical sampling
- Project save/load with readable `.blab.json` files

## Requirements

- Windows 10/11 64-bit
- Python 3.11 or newer
- An OpenCL runtime for `bempp-cl`/`pyopencl`

The [Intel CPU OpenCL runtime](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html) is a practical option even on many non-Intel systems.

While not required, if modeling in Autodesk Fusion, the [Fusion2Msh](https://github.com/JWSound/fusiontomsh) add-in is strongly recommended for quick imports of models into Boundary Lab.

## Install

From the repository root:

```bash
python -m pip install -e ".[gui]"
```

## Run The GUI

```bash
blab gui
```

On startup, Boundary Lab updates `ath/ath.cfg` so Ath uses the bundled Gmsh executable and writes generated files into:

```text
runs/ath_output
```


## Run A Solve Server

Boundary Lab can also run a local or LAN-accessible job server that accepts solve
jobs and streams per-frequency results back as NDJSON events:

```bash
blab server --host 127.0.0.1 --port 8765
```

To use it from the GUI application, open `Edit > Preferences`, set `Solve Backend` to
`Server`, and set `Solve Server URL` to the server address. For another machine
on the LAN, bind the server to that machine's LAN address or `0.0.0.0` and use
`http://<server-ip>:8765` in the client. The GUI uploads the solver mesh files
with each server job, so the server does not need access to the client's local
paths.

API surface:

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
