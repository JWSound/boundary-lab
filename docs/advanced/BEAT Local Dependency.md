# BEAT Engine dependency

Boundary Lab requires the independently released `beat-engine` package. Its
`pyproject.toml` selects the exact engine requirement and must agree with
`packaging/runtime-lock.json`. Releases pin a published wheel URL and SHA-256,
so installation verifies that artifact without a sibling engine checkout.
Release-preparation branches may temporarily pin a Git commit while qualifying
the new engine; the release gate rejects those Git pins.

```text
python -m pip install -e ".[gui,dev]"
python -m beat_engine instantiate --backend cpu
python -m beat_engine doctor --backend cpu --threads 2
python -m beat_engine paths --backend cpu
```

Julia is installed separately; the wheel does not provide Julia or GPU drivers.
Use `--backend cuda` or `--backend rocm` to prepare the corresponding environment.
Hardware availability is a separate qualification from successful installation.

Run these commands with Boundary Lab's Python environment activated. On Windows,
you can instead use `.\.venv\Scripts\python.exe` in place of `python`. Installing
BEAT into another Python environment does not configure Boundary Lab's environment.

## Solver choices

The Preferences solver list comes from the installed engine's `backend_catalog()`
API, with Boundary Lab Server added by the application. Listing and selecting a
backend does not launch Julia, test hardware, install packages or contact GitHub.
There are no availability indicators in Preferences. A selected backend that
cannot run reports its name and the runtime error after Solve is clicked.

The same catalog supplies local backend IDs and Julia project paths to source
solves, physical-system solves, retained-field evaluation, and explicit CLI
backend choices. Legacy settings aliases are preserved. Unknown IDs fail instead
of selecting a different backend. Headless automatic selection retains its
CUDA-then-CPU policy.

## Updating an existing installation

Update the Boundary Lab checkout, then rerun its installer or the normal
`python -m pip install -e ".[gui]"` command. Boundary Lab selects its supported
engine release automatically. After an engine update, prepare each backend you
use again with `python -m beat_engine instantiate --backend cpu` (or `cuda`/`rocm`),
then inspect `doctor` output. The Windows installer's solver prompts perform
environment preparation and include CUDA/ROCm runtime checks.

The wheel supplies Julia source and project files, but not the downloaded Julia
packages. Previous Julia package downloads may be reused from the local depot;
the installed release's own project still needs to be instantiated. Restart
Boundary Lab after updating so existing workers do not retain the old engine.

Engine releases contain wheel and source distributions on
[GitHub Releases](https://github.com/JWSound/BEAT_Engine/releases).
No PyPI publication is configured.
The compiled-system contract, worker negotiation, transport, numerical sources,
and fixtures belong to BEAT Engine. Boundary Lab owns project compilation,
backend/SDK selection policy, GUI/CLI integration, and result models.

There is no bundled engine or distribution switch. Missing or incompatible engine
packages fail instead of falling back. `BLAB_BEAT_ENGINE_DISTRIBUTION` is obsolete
and can be removed from shell configuration.

## Contributor override

Install Boundary Lab normally first, then explicitly replace the engine in that
development environment:

```text
python -m pip install --no-deps -e <path-to-BEAT_Engine>
python -m beat_engine paths
```

Use a unique candidate version and preserve supported contracts. On the application
feature branch, update the dependency pin, supported-version check and associated
test expectations to that candidate version. Never relabel candidate code as an
existing stable version.
Reinstalling Boundary Lab may restore the pinned release; use a separate virtual
environment for engine development. To restore the application's declared
dependencies, run this from the intended Boundary Lab checkout:

```text
python -m pip install --force-reinstall .
```

## Updating the engine

Boundary Lab 0.5.0 targets BEAT 0.6.0. Its projected FEM-BEM interface correction
uses the same DP0 face flux for FEM loading and BEM radiation. It rejects
rank-deficient or poorly conditioned interface coefficient transfers, including
some previously accepted triangulations. Recompute affected results and verify
mesh convergence; refinement alone does not guarantee an admissible interface.
See the engine's
[projected-interface documentation](https://github.com/JWSound/BEAT_Engine/blob/main/docs/Projected%20interface%20flux.md).

The Vented_Sub qualification fixture retains an existing CUDA precision
limitation: 50 Hz voice-coil current differs from production CPU by 0.134%,
exceeding the 0.1% backend-agreement screen (BEAT 0.5.1: 0.135%).
Use CPU when this low-current accuracy matters. See the
[0.5.0 qualification notes](../releases/0.5.0.md#known-cuda-precision-limitation)
for the double-precision diagnostic and scope of the comparison.

BEAT 0.4.0 introduced fixed-order CPU regular quadrature and
coupled reductions with Float64 assembly and refined Float32 LU. Dense-system
memory use can be approximately three times the pre-0.4 default. See the
[engine release notes](https://github.com/JWSound/BEAT_Engine/releases/tag/v0.4.0)
for controls and compatibility details. Boundary Lab continues to emit compiled
contract v1 for its existing model features; its legacy uniform source setting
is translated to the equivalent baseline normal-velocity request.

Qualify the new engine version independently, then update Boundary Lab's dependency
URL/hash and supported-version check together. Run application tests and
`python scripts/check_engine_integration.py`; compare complex result arrays with
the prior release and check engine/runtime provenance. Package installation must
work without an editable engine checkout. GPU changes also require matching
hardware qualification.
