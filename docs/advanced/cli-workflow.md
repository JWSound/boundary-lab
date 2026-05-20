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

For multi-mesh and multi-radiator TOML config files, see [solver-configuration.md](solver-configuration.md).

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

The CLI path is useful for scripted workflows and regression checks. It does not expose the full GUI project workflow or live plot update behavior.
