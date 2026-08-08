# Advanced CLI Workflow

Boundary Lab still includes command-line tools for mesh cleaning, solving, data preparation, and static plot generation. The GUI is the recommended entry point for normal use.

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
