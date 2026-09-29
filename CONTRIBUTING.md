# Contributing to Boundary Lab

Thanks for taking an interest in Boundary Lab. This document covers how to set up a development environment, run tests, and submit changes.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Platform Notes](#platform-notes)

## Development Setup

### Prerequisites

- **Python 3.11 or newer**
- **Git**
- **Julia** (required for BEAT Engine solver backends)
- **Wine** (Linux/macOS only — needed if using Ath for mesh generation)
- **OpenCL runtime** (required for the Bempp solver backend)
- **NVIDIA GPU + CUDA** (optional — required for GPU-accelerated solving)

### 1. Clone the repository

```bash
git clone https://github.com/JWSound/boundary-lab.git
cd boundary-lab
```

### 2. Install Boundary Lab in editable mode

```bash
python -m pip install -e ".[gui]"
```

This installs Boundary Lab and its Python dependencies (PySide6, pyvista, bempp-cl, etc.) in development mode so local changes take effect without reinstalling.

### 3. Install Julia and prepare solver environments

Boundary Lab uses Julia for the BEAT Engine solver backends. Two separate Julia environments are required:

**For GPU solving (BEAT Engine CUDA):**

```bash
julia --project=src/blab/solvers/julia_cuda -e "using Pkg; Pkg.instantiate()"
```

**For CPU solving (BEAT Engine CPU):**

```bash
julia --project=src/blab/solvers/julia_local -e "using Pkg; Pkg.instantiate()"
```

> Julia must be available on your `PATH` for Boundary Lab to locate it at runtime. If you use a non-standard installation path, you can point the solver to it via the application preferences.

### 4. Install an OpenCL runtime (Bempp solver backend)

The Bempp CPU solver requires an OpenCL runtime. On most systems:

- **Windows:** Install the [Intel CPU OpenCL runtime](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html) (compatible with AMD and Intel CPUs).
- **Linux:** `apt install intel-opencl-icd` or the equivalent package for your distribution.
- **macOS:** OpenCL is not available on modern macOS. Use the Julia-based BEAT Engine solvers instead (CPU or CUDA via Docker).

### 5. Install Wine (Linux/macOS only — optional)

If you plan to generate meshes using Ath, Wine is required on non-Windows systems. Without Ath, you can still load existing `.msh` files and run solvers.

## Project Structure

```
boundary-lab/
├── src/
│   └── blab/
│       ├── __init__.py
│       ├── app.py              # PySide6 GUI application entry point
│       ├── widgets/            # GUI panels (viewport, plots, editor)
│       ├── solvers/            # BEM solver backends
│       │   ├── bempp_local.py  # Bempp-cl CPU solver
│       │   ├── julia_cuda/     # BEAT Engine CUDA Julia project
│       │   └── julia_local/    # BEAT Engine CPU Julia project
│       ├── mesh/               # Mesh import/export and processing
│       └── utils/              # Shared utilities
├── docs/                       # User-facing documentation
│   ├── User Guide.md
│   ├── Boundary Lab Server.md
│   ├── Docker.md
│   ├── Model Assumptions.md
│   ├── Inputs and Outputs.md
│   └── advanced/               # Advanced workflow docs
├── assets/                     # Screenshots and images
├── examples/                   # Example project files
├── tests/                      # Test suite
├── pyproject.toml
└── README.md
```

## Running Tests

Boundary Lab uses pytest. From the repository root:

```bash
pytest
```

Tests are automatically run on every push via the project's CI workflow (Windows, latest Python).

## Code Style

Boundary Lab uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. The CI pipeline checks Ruff rules on every pull request. To run the linter locally:

```bash
ruff check src/
ruff format --check src/
```

To auto-format:

```bash
ruff format src/
```

Please ensure your changes pass Ruff checks before submitting a pull request.

## Pull Request Process

1. **Open an issue first** describing what you'd like to change (unless it's a very small fix). This avoids wasted effort if the change isn't a good fit.
2. **Fork the repository** and create a branch off `main` with a descriptive name: `fix/typo-readme`, `docs/contributing-guide`, `feat/description-here`.
3. **Make your changes.** Keep commits small and well-scoped.
4. **Run the tests** and Ruff linter to verify nothing is broken.
5. **Submit a pull request** against `main`. Include a clear description of what your changes do and why.
6. **Respond to review feedback** if any. The project is maintained by one person, so review turnaround depends on their availability.

### What makes a good PR

- Small, focused changes are much easier to review than large rewrites.
- Documentation improvements are always welcome.
- If fixing a bug, include a brief description of the root cause.
- If adding a feature, the discussion in the issue thread should come first.

## Platform Notes

### macOS

- OpenCL is not supported on modern macOS. Use the Julia-based BEAT Engine CPU solver, or the BEAT Engine CUDA solver running on a remote server or Docker container.
- Test coverage on macOS is lower than on Windows. If you encounter issues, please open a bug report with diagnostic info from `About > Diagnostic Info`.

### Linux

- Ath mesh generation requires Wine.
- The BEAT Engine Julia environments and the Bempp OpenCL solver have both been tested on Ubuntu and Debian-based distributions.

### Windows

- This is the primary development and testing platform. Everything should work out of the box.
- The GPU CUDA solver requires an NVIDIA Maxwell-generation or newer GPU.

## License

Boundary Lab is [GPL-3.0](LICENSE). By contributing, you agree that your contributions will be licensed under the same license.
