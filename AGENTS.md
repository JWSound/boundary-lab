# Boundary Lab agent guide

Boundary Lab is a free, open-source loudspeaker enclosure design application. It
loads Gmsh geometry plus `.blab.json` physical-system projects and solves exterior
BEM or coupled FEM-BEM-LEM electroacoustic models. Python application code is in
`src/blab`; BEAT Engine numerical code is under `src/blab/solvers/julia_local`.

Use the headless project workflow for automated diagnosis. Start with
`blab project validate <project.blab.json> --json`, then run
`blab project solve`; arbitrary complex-pressure probes and result artifacts are
documented in [the advanced CLI workflow](docs/advanced/cli-workflow.md#headless-project-workflow).
The default selects BEAT CUDA when it is functional and otherwise uses BEAT CPU.
Validate before launching a potentially expensive solve, keep mesh paths relative
to the project, and do not commit generated `runs/` artifacts.

Keep reusable solve preparation free of Qt; the GUI and CLI should call the same
core contracts. Preserve complex per-excitation results and record enough
provenance to reproduce a run. Use Python 3.11+, run focused tests with
`python -m pytest <test-file>`, and run `python -m ruff check src tests` for
changed Python code.