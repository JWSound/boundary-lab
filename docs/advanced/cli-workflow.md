# Advanced CLI Workflow

Boundary Lab still includes command-line tools for mesh cleaning, solving, data preparation, and static plot generation. The GUI is the recommended entry point for normal use.

## Headless Project Workflow

The project CLI loads a complete `.blab.json` physical system and runs the same
BEAT Engine physical-system path used by the desktop application. It supports
exterior BEM and coupled FEM-BEM-LEM projects without opening Qt.

Validate a project before starting an expensive solve:

```bash
blab project validate examples/Simple_Sealed/simple_sealed.blab.json --json
```

Validation migrates the project schema, resolves paths relative to the project,
loads and compiles all meshes and physical groups, checks backend capabilities,
and reports the solve kind, assumptions, mesh hashes, excitations, outputs, and
frequency range. It does not launch Julia or assemble solver matrices.

Run the project using its saved frequency, polar, sphere, symmetry, channel, and
observation-plane settings:

```bash
blab project solve speaker.blab.json --backend beat_cpu --output runs/speaker-check
```

The default `--backend beat_auto` always selects a BEAT Engine backend. It probes
the configured Julia CUDA environment and uses `beat_cuda` when CUDA is
functional; otherwise it falls back to `beat_cpu`. Use an explicit
`--backend beat_cpu`, `--backend beat_cpu_condensed`, `--backend beat_cuda`, or
`--backend beat_rocm` to disable automatic selection. `beat_cpu_condensed` uses
the ordinary CPU path for exterior-only projects and exact FEM interface
condensation for coupled projects.
`--julia-executable` and `--julia-threads` select the Julia installation and
thread count. The output directory must not already exist, which prevents an
agent from accidentally overwriting a previous run.

For machine-readable progress, request newline-delimited JSON events:

```bash
blab project solve speaker.blab.json --request diagnosis.json --events ndjson
```

### Solve request overlays

A request overlay changes one experiment without modifying the project. All
coordinates use the project's meter-based global coordinate frame. For example:

```json
{
  "schema_version": 1,
  "frequencies_hz": [35.5, 40.0, 45.0, 50.0, 56.0, 63.0],
  "excitation_port_ids": ["excitation:woofer"],
  "include_project_observations": false,
  "probes": [
    {
      "id": "farfield_on_axis",
      "coordinate_frame": "project",
      "points_m": [[0.0, 0.0, 3.0]]
    },
    {
      "id": "port_nearfield",
      "coordinate_frame": "project",
      "points_m": [[0.18, -0.12, 0.03]]
    }
  ],
  "retain": ["bem_boundary_traces", "fem_nodal_pressure"],
  "solver_options": {
    "quadrature_order": 4,
    "singular_order": 4
  }
}
```

`frequency_sweep` may replace `frequencies_hz`:

```json
{
  "schema_version": 1,
  "frequency_sweep": {
    "min_hz": 20,
    "max_hz": 500,
    "count": 100,
    "spacing": "log"
  }
}
```

Supported `retain` entries are `bem_boundary_pressure`,
`bem_boundary_neumann`, `bem_boundary_traces` (both BEM traces), and
`fem_nodal_pressure`. Retaining fields can substantially increase solve output.
Complex results remain separated by excitation port; the headless path does not
collapse them into synthesized GUI channels.

Full-matrix `validation_diagnostics` may be enabled for monolithic coupled BEAT
CPU solves. They cannot be combined with CPU, CUDA, or ROCm FEM static
condensation; project validation rejects that combination before Julia starts.

### Result artifact

Each run is a self-describing directory:

```text
speaker-check/
  manifest.json
  project.snapshot.blab.json
  request.json
  compiled-system.json
  domains.json
  domains.npz
  frequencies/
    000000.json
    000000.npz
    ...
```

`manifest.json` records the backend, solver options, phasor convention,
frequency axis, excitation ids, completion mask, and run status. Each frequency
is committed independently, so a failed or interrupted run retains completed
complex quantities and diagnostics. The frequency JSON maps semantic quantity
ids to arrays in the corresponding NPZ file. The result phasor convention is
`exp(-i omega t)`.

The current headless interface evaluates arbitrary exterior probe points declared
before the solve. Retained BEM traces and FEM nodal fields provide the data needed
for a future post-solve point-sampling command without repeating the system solve.

### Speaker package solve and export

Create a complex-pattern level 1 package directly from a project:

```bash
blab project export-speaker speaker.blab.json --output speaker.blabsp --fidelity pattern
```

Create a level 2 package with fixed distributed BEM sources:

```bash
blab project export-speaker speaker.blab.json --output speaker.blabsp --fidelity fixed
```

The command accepts the same `--request`, `--backend`, `--julia-executable`,
`--julia-threads`, and `--events` controls as the headless solve. It forces the
spherical pressure and boundary traces required by the selected fidelity and
publishes the archive atomically only after every requested frequency has
completed.

## Legacy Mesh Workflow

The CLI workflow is:

1. `blab clean`
2. `blab solve`
3. `blab prepare`
4. `blab plot`

## Clean A Mesh

```bash
blab clean input.msh output_clean.msh --merge-tol 1e-9
```

This merges coincident vertices, removes degenerate or duplicate triangles, and writes Gmsh 2.2 format.

## Run A Solve

```bash
blab solve output_clean.msh --output-npz pressure_data_raw.npz --freq-min 200 --freq-max 20000 --freq-count 48 --workers 4
```

Useful options include:

- `--config`
- `--output-npz`
- `--freq-min`
- `--freq-max`
- `--freq-count`
- `--step-size`
- `--min-angle`
- `--max-angle`
- `--axial-offset`
- `--workers`
- `--gmres-tol`
- `--spherical-sampling`
- `--spherical-sampling-points`

The legacy solve CLI accepts a TOML file for multi-mesh, multi-radiator
exterior-BEM jobs. Paths are resolved relative to the TOML file. For example:

```toml
[[meshes]]
name = "cabinet"
file = "cabinet.msh"
scale_factor = 0.001
translation_m = [0.0, 0.0, 0.0]

[[radiators]]
name = "woofer"
mesh = "cabinet"
tag = 2
channel = "main"
velocity_offset_db = 0.0

[radiators.hpf]
filter = "butterworth"
order = 2
frequency_hz = 80.0
```

Each radiator requires `name` and integer physical `tag`; `mesh` is required
when more than one configured mesh could contain that tag. Optional radiator
fields are `channel`, `velocity_offset_db`, `level_db`, `polarity`, `delay_ms`,
`hpf`, and `lpf`.

## Prepare Visualization Data

```bash
blab prepare pressure_data_raw.npz pressure_data_formatted.npz --min-db -30 --max-db 0
```

This applies clipping, interpolation, normalization, and smoothing for the static plot pipeline.

## Generate Static Plots

```bash
blab plot pressure_data_formatted.npz --output-dir .
```

This writes:

- `horizontal_isobar.png`
- `vertical_isobar.png`
- `acoustic_impedance.png`

## Notes

The CLI path is useful for scripted exterior-BEM workflows and regression
checks. It does not load `.blab.json` physical systems, run coupled FEM-BEM
models, or expose the GUI's live plot behavior.
