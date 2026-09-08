# Installation and Setup

This guide covers installing, updating, and preparing Boundary Lab on Windows
and Linux. Only the Python application and GUI are required to design systems,
import meshes, and inspect projects. Geometry generation through Ath and each
solver backend have their own runtime requirements described below.

Boundary Lab requires Python 3.11 or newer.

The guided Windows scripts target 64-bit Windows 10 and 11. The Linux commands
below target an x86-64 Debian or Ubuntu desktop, including Ubuntu under WSL2.
The Python application and BEAT Engine CPU backend may work on other Linux
architectures, but the bundled Windows Ath executable requires an
x86-compatible Wine environment.

## Windows

### Guided installation

The recommended Windows path uses the two batch files in the repository:

1. Double-click `01_install_update_boundary-lab.bat`.
2. Allow the script to install Git or Python if either prerequisite is missing.
   Close the window and run the script again when instructed so Windows can
   refresh the available commands.
3. Choose whether to prepare the optional Julia-based BEAT Engine solvers. If
   an NVIDIA GPU is detected, the installer also offers to prepare and verify
   the CUDA environment.
4. Double-click `02_start_boundary_lab.bat` to launch the application.

The installer creates `.venv`, installs or repairs Boundary Lab and its GUI
dependencies, and validates the `blab` command. When run from an existing Git
checkout, it can optionally pull a fast-forward update from `origin/main`.

The launcher remembers an NVIDIA GPU selection when more than one is
available. To select again, run this from Command Prompt in the repository:

```bat
02_start_boundary_lab.bat /choose
```

### Manual Windows installation

Install Python 3.11 or newer and Git, then run the following from the repository
folder in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
blab gui
```

If PowerShell prevents activation of local scripts, the environment can be
used without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gui]"
.\.venv\Scripts\blab.exe gui
```

Ath runs natively on Windows, and Boundary Lab meshes its geometry through the
cross-platform Python Gmsh library. Wine is not required.

## Linux

The examples below use Debian or Ubuntu package names. Adapt them for the
package manager used by another distribution.

### Application and GUI

Install the base tools and Qt/X11 runtime libraries:

```bash
sudo apt update
sudo apt install \
  git python3 python3-pip python3-venv \
  libegl1 libgl1 \
  libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 \
  libxcb-xkb1 libxkbcommon-x11-0
```

Clone and install Boundary Lab:

```bash
git clone https://github.com/JWSound/boundary-lab.git
cd boundary-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
blab gui
```

### WSLg and Wayland

The embedded VTK mesh preview currently requires Qt and VTK to use compatible
window handles. If startup under WSLg or a Wayland session fails with an X11
`BadWindow` or `X_ConfigureWindow` error, launch Boundary Lab through Qt's xcb
backend:

```bash
QT_QPA_PLATFORM=xcb blab gui
```

The xcb libraries in the Linux prerequisite command are required for this
backend. Avoid setting `QT_QPA_PLATFORM` globally because it would affect every
Qt application in the shell environment.

## Ath geometry generation on Linux

Boundary Lab automatically invokes the bundled `ath.exe` through `wine` on
Linux. The bundled Ath executable is 32-bit; Gmsh runs natively through the
Python package installed with Boundary Lab.

On Debian or Ubuntu:

```bash
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install --install-recommends wine wine64 wine32:i386
```

Wine's default 64-bit/WoW64 prefix supports Ath. An optional dedicated prefix
can be initialized with:

```bash
export WINEPREFIX="$HOME/.wine-boundary-lab"
wineboot -u
```

Launch Boundary Lab from a shell carrying the same `WINEPREFIX`. No .NET
runtime, Wine Mono, Wine Gecko, or Winetricks package is required for Ath mesh
generation. Gnuplot is optional and is not used by Boundary Lab's normal mesh
generation workflow.

Generated Ath GEO, diagnostic logs, and final solve-ready meshes are written
below `runs/generated_geometry`. Gmsh meshing runs in a cancellable child
process, and no intermediate raw mesh is written or reloaded.

## Solver setup

Boundary Lab offers three local BEAT Engine backends. All support exterior
BEM and coupled FEM-BEM systems, including X and XY symmetry.

| Backend | Hardware/runtime | Exterior BEM | Coupled FEM-BEM |
|---|---|:---:|:---:|
| BEAT Engine CPU | Julia and CPU BLAS/LAPACK | Yes | Yes |
| BEAT Engine Nvidia CUDA | Julia and supported NVIDIA GPU | Yes | Yes |
| BEAT Engine AMD ROCm | Julia, AMDGPU.jl, and a functional ROCm SDK | Yes | Yes |

Bempp and the legacy HTTP solve server are retired. Saved backend selections
for either migrate to BEAT CPU. The ROCm path uses GPU-resident regular and Duffy singular operator
assembly, rocBLAS/rocSOLVER dense solves, and GPU exterior field evaluation.
See [BEAT Engine AMD ROCm](advanced/beat-engine-rocm.md) for setup and
validation details.

### BEAT Engine AMD ROCm on Windows

Install an AMD Windows HIP SDK supported by your GPU using AMD's
[Windows HIP SDK installer](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/).
AMD publishes the current GPU and operating-system matrix in the
[Windows system requirements](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/shared/hipsdk/reference/system-requirements.html).
Restart the terminal after installation so updated environment variables are visible.

Run `01_install_update_boundary-lab.bat` and accept the ROCm solver prompt. The
installer detects `HIP_PATH`, `ROCM_PATH`, `ROCM_HOME`, and installed SDK versions
under `%ProgramFiles%\AMD\ROCm`; it then prepares the Julia environment and verifies
AMDGPU.jl, rocBLAS, and rocSOLVER. Boundary Lab deliberately does not download or
silently elevate AMD's SDK installer because it requires separate license acceptance.

Portable TheRock SDK layouts can be selected during installation or configured later:

```powershell
blab rocm configure "$env:USERPROFILE\SDKs\ROCm-TheRock"
blab rocm detect --json
```

This stores a per-user path under `%LOCALAPPDATA%\Boundary Lab`, avoiding a
checkout-specific drive or directory. Use `blab rocm clear` to remove the saved
override.

### BEAT Engine CPU

Install [Julia](https://julialang.org/downloads/) and make `julia` available on
`PATH`. Prepare the CPU project from the repository root:

```bash
python -m beat_engine instantiate --backend cpu
```

The CPU backend supports Intel, AMD, and ARM processors. Runtime depends heavily
on the mesh size, frequency range, symmetry, and the BLAS implementation used by
Julia.

### BEAT Engine CUDA

Install a current NVIDIA driver for a Maxwell-generation or newer NVIDIA GPU,
then install Julia and prepare the CUDA project:

```bash
python -m beat_engine instantiate --backend cuda
julia --project="<CUDA project from python -m beat_engine paths --backend cuda>" -e "using CUDA; CUDA.functional() || error(\"CUDA is not functional\"); CUDA.versioninfo()"
```

On WSL2, use an NVIDIA Windows driver with WSL CUDA support. Do not install a
second Linux display driver inside WSL.

GPU solve memory grows approximately quadratically with the number of mesh
elements. The following values are planning estimates rather than hard limits:

| Total elements | Estimated VRAM |
|---:|---:|
| 1,000 | ~50-100 MB |
| 2,000 | ~200-300 MB |
| 3,000 | ~400-600 MB |
| 5,000 | ~1.0-1.5 GB |
| 7,000 | ~2.0-3.0 GB |
| 10,000 | ~4-6 GB |
| 15,000 | ~8-12 GB |
| 20,000 | ~14-20 GB |

## Updating an installation

On Windows, rerun `01_install_update_boundary-lab.bat` and accept the update
prompt. Automatic updates require a clean checkout on the `main` branch and use
a fast-forward-only pull.

For a manual Windows or Linux checkout:

```bash
git pull --ff-only
python -m pip install -e ".[gui]"
```

Rerun the applicable Julia `Pkg.instantiate()` command after solver dependency
changes.

## Installation diagnostics

Useful checks from an activated environment are:

```bash
python --version
python -m pip check
blab --help
```

If Ath reports that Wine is required, confirm that `wine` is available on the
same `PATH` used to launch Boundary Lab:

```bash
wine --version
```
