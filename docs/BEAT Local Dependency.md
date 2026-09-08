# BEAT Engine dependency

Boundary Lab requires the independently released `beat-engine` package. Its
`pyproject.toml` pins the `v0.1.0rc1` wheel URL and SHA-256, so ordinary installation
downloads and verifies that exact artifact without a sibling engine checkout:

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

The release is [v0.1.0rc1](https://github.com/JWSound/BEAT_Engine/releases/tag/v0.1.0rc1).
It contains wheel and source distributions. No PyPI publication is configured.
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

The candidate checkout must retain the supported package version and contracts.
Reinstalling Boundary Lab may restore the pinned release; use a separate virtual
environment for engine development. To restore the release explicitly:

```text
python -m pip install --force-reinstall --no-deps "beat-engine @ https://github.com/JWSound/BEAT_Engine/releases/download/v0.1.0rc1/beat_engine-0.1.0rc1-py3-none-any.whl#sha256=10513eff64a9e19d3a5d442887afdcc66a18b621056ee6cd53ca0391493fcb00"
```

## Updating the engine

Qualify the new engine version independently, then update Boundary Lab's dependency
URL/hash and supported-version check together. Run application tests and
`python scripts/check_engine_integration.py`; compare complex result arrays with
the prior release and check engine/runtime provenance. Package installation must
work without an editable engine checkout. GPU changes also require matching
hardware qualification.
