# Installation and Setup

This guide covers installing, updating, and preparing Boundary Lab on Windows
and Linux. Only the Python application and GUI are required to design systems,
import meshes, and inspect projects. Geometry generation through Ath and each
solver backend have their own runtime requirements described below.

Boundary Lab requires Python 3.11 or newer.

The guided Windows scripts target 64-bit Windows 10 and 11. The Linux commands
below target an x86-64 Debian or Ubuntu desktop, including Ubuntu under WSL2.
The Python application and BEAT Engine CPU backend may work on other Linux
architectures, but the bundled Windows Ath and Gmsh executables require an
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

Ath and the bundled Windows Gmsh executable run natively on Windows; Wine is
not required.

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
Linux. The bundled Ath executable is 32-bit and the bundled Gmsh executable is
64-bit, so an x86-64 Linux installation needs both Wine architectures.

On Debian or Ubuntu:

```bash
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install --install-recommends wine wine64 wine32:i386
```

Wine's default 64-bit/WoW64 prefix supports both bundled executables. Do not
create a `WINEARCH=win32` prefix because it cannot run the 64-bit Gmsh binary.
An optional dedicated prefix can be initialized with:

```bash
export WINEPREFIX="$HOME/.wine-boundary-lab"
wineboot -u
```

Launch Boundary Lab from a shell carrying the same `WINEPREFIX`. No .NET
runtime, Wine Mono, Wine Gecko, or Winetricks package is required for Ath mesh
generation. Gnuplot is optional and is not used by Boundary Lab's normal mesh
generation workflow.

To verify the bundled 64-bit Gmsh executable before using Ath:

```bash
wine gmsh/gmsh-4.15.2-Windows64/gmsh.exe -version
```

Generated Ath files are written below `runs/generated_geometry` and the mesh
used by Boundary Lab is cleaned before being added to the project.

## Solver setup

Boundary Lab offers four local backends. The ROCm backend currently targets
non-symmetric exterior BEM; the other BEAT backends also support coupled systems.

| Backend | Hardware/runtime | Exterior BEM | Coupled FEM-BEM |
|---|---|:---:|:---:|
| Bempp OpenCL CPU | CPU OpenCL runtime | Yes | No |
| BEAT Engine CPU | Julia and CPU BLAS/LAPACK | Yes | Yes |
| BEAT Engine CUDA | Julia and supported NVIDIA GPU | Yes | Yes |
| BEAT Engine ROCm | Julia, AMDGPU.jl, and a functional ROCm SDK | Yes | No |

The server backend can submit exterior BEM jobs to another Boundary Lab
installation. The ROCm path supports non-symmetric exterior BEM using native
GPU-resident regular and Duffy singular operator assembly plus a rocBLAS/rocSOLVER
GPU solve. Exterior field evaluation remains CPU-resident.
See [BEAT Engine ROCm development](advanced/beat-engine-rocm.md) for setup and
validation details.

### Bempp OpenCL CPU

Bempp-cl and PyOpenCL are installed with Boundary Lab. The operating system
must additionally provide an OpenCL installable client driver (ICD) exposing a
CPU device.

#### Windows on an Intel CPU

Download and run the current Intel 64-bit Windows installer from
[Intel CPU Runtime for OpenCL Applications](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html),
then restart Boundary Lab. Open **About > Diagnostic Info** and run diagnostics
to confirm that an Intel CPU OpenCL device is listed.

Intel does not officially support this CPU runtime on AMD processors. On an AMD
Windows system, use BEAT Engine CPU, a Boundary Lab server, or another OpenCL
CPU runtime known to support that processor.

#### Linux with PoCL

For a portable CPU runtime on Ubuntu, including AMD CPUs and distributions not
listed by Intel's runtime support matrix, install PoCL:

```bash
sudo apt install pocl-opencl-icd clinfo
```

#### Linux on an Intel CPU

For a supported Intel Core or Xeon processor, Intel distributes its CPU runtime
through the oneAPI APT repository. Configure the repository and install only
the runtime package:

```bash
sudo apt install wget gpg

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update
sudo apt install intel-oneapi-runtime-opencl clinfo
```

Intel documents the supported processor families and Linux releases in the
[Intel CPU Runtime for OpenCL Applications Guide](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-guide.html).
Ubuntu's similarly named `intel-opencl-icd` package is the Intel **GPU** compute
runtime, not this CPU runtime.

Verify that at least one CPU OpenCL device is available:

```bash
clinfo -l
python -c 'import pyopencl as cl; print([(p.name, [d.name for d in p.get_devices()]) for p in cl.get_platforms()])'
```

If PyOpenCL reports `PLATFORM_NOT_FOUND_KHR`, the generic OpenCL loader is
present but no usable vendor runtime is registered.

### BEAT Engine CPU

Install [Julia](https://julialang.org/downloads/) and make `julia` available on
`PATH`. Prepare the CPU project from the repository root:

```bash
julia --project=src/blab/solvers/julia_local -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
```

The CPU backend supports Intel, AMD, and ARM processors. Runtime depends heavily
on the mesh size, frequency range, symmetry, and the BLAS implementation used by
Julia.

### BEAT Engine CUDA

Install a current NVIDIA driver for a Maxwell-generation or newer NVIDIA GPU,
then install Julia and prepare the CUDA project:

```bash
julia --project=src/blab/solvers/julia_cuda -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
julia --project=src/blab/solvers/julia_cuda -e "using CUDA; CUDA.functional() || error(\"CUDA is not functional\"); CUDA.versioninfo()"
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

## Boundary Lab server

The installed command can also run a local or LAN server:

```bash
blab server --host 127.0.0.1 --port 8765 --solver bempp_cpu
blab server --host 127.0.0.1 --port 8765 --solver beat_cpu --julia-threads auto
blab server --host 127.0.0.1 --port 8765 --solver beat_cuda
```

Configure the GUI through **Edit > Preferences > BEM Solver > Server** and set
the server URL. See [Boundary Lab Server](Boundary%20Lab%20Server.md) for
capabilities, authentication, and LAN setup, or [Docker](Docker.md) for an
authenticated CUDA deployment.

## Installation diagnostics

Useful checks from an activated environment are:

```bash
python --version
python -m pip check
blab --help
```

If the GUI reports that PyOpenCL's compiled `_cl` extension is missing, repair
the wheel with:

```bash
python -m pip install --force-reinstall --no-cache-dir pyopencl
```

If Ath reports that Wine is required, confirm that `wine` is available on the
same `PATH` used to launch Boundary Lab:

```bash
wine --version
```
