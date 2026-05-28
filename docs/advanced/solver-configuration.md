# Advanced Solver Configuration

This document covers the TOML-based configuration path for cases that go beyond the default single-mesh, single-radiator workflow.

## Terminology

The current solver does **not** implement a full lumped-element or electro-mechanical driver model.

What it does implement is a **prescribed radiator drive model**:

- one or more radiating surface groups
- relative drive level in dB
- polarity inversion
- delay-based phase offset
- basic crossover transfer functions
- optional composition of multiple meshes into one acoustic solve

These features shape the normal-velocity excitation applied to BEM radiator surfaces. They do not currently derive diaphragm motion from T/S parameters, electrical impedance, suspension compliance, motor force factor, or network loading.

## TOML Config Overview

Use:

```bash
blab solve --config path/to/config.toml
```

Config files may define:

- `[[meshes]]`
- `[[radiators]]`
- `[radiators.hpf]`
- `[radiators.lpf]`

TOML paths are resolved relative to the config file location.

## Multi-Mesh Solves

When `[[meshes]]` entries are present, the solver ignores the positional `mesh_file` argument for geometry loading and instead concatenates all configured meshes into one BEM grid.

Each mesh may define:

```toml
[[meshes]]
name = "waveguide"
file = "waveguide_clean.msh"
scale_factor = 0.001
translation_m = [0.0, 0.0, 0.0]
```

Fields:

- `name`: identifier used by radiators
- `file`: mesh path
- `scale_factor`: optional per-mesh unit conversion
- `translation_m`: optional `[x, y, z]` translation in meters

Meshes should already be positioned consistently and should not unintentionally overlap.

## Radiator Definitions

Each radiator identifies a driven surface group inside one mesh:

```toml
[[radiators]]
name = "HF"
mesh = "waveguide"
tag = 1
level_db = -2.0
polarity = 1
delay_ms = 0.0
```

Fields:

- `name`: label used in logs and impedance plots
- `mesh`: mesh name from `[[meshes]]`
- `tag`: physical surface tag local to that mesh
- `level_db`: relative prescribed velocity magnitude
- `polarity`: `1` or `-1`
- `delay_ms`: propagation or alignment delay expressed in milliseconds

When only one mesh is active, `mesh` may be omitted. With multiple configured meshes, every radiator must specify it.

## Crossover Shaping

Each radiator may optionally define a crossover response:

```toml
[radiators.hpf]
filter = "linkwitz_riley"
order = 4
frequency_hz = 1400.0
```

Supported values:

- `filter`: `butterworth`, `linkwitz_riley`
- `order`: `1`, `2`, `4`, `6`
- `frequency_hz`: positive cutoff frequency

Use `[radiators.hpf]` for high-pass shaping and `[radiators.lpf]` for low-pass shaping. If both are present, Boundary Lab multiplies the two complex responses and applies the result to the radiator drive.

## Example Two-Way Multi-Mesh Setup

```toml
[[meshes]]
name = "waveguide"
file = "waveguide_clean.msh"
scale_factor = 0.001
translation_m = [0.0, 0.0, 0.0]

[[meshes]]
name = "woofer_enclosure"
file = "woofer_enclosure_clean.msh"
scale_factor = 0.001
translation_m = [0.0, 0.0, 0.0]

[[radiators]]
name = "HF"
mesh = "waveguide"
tag = 1
level_db = -2.0
polarity = 1
delay_ms = 0.0

[radiators.hpf]
filter = "linkwitz_riley"
order = 4
frequency_hz = 1400.0

[[radiators]]
name = "LF"
mesh = "woofer_enclosure"
tag = 3
level_db = 0.0
polarity = 1
delay_ms = 0.0

[radiators.lpf]
filter = "linkwitz_riley"
order = 4
frequency_hz = 1400.0
```

See [`docs/examples/2waybookshelf/2wayconfig.toml`](../examples/2waybookshelf/2wayconfig.toml) for a ready-to-edit configured two-way example.

## Output Notes

With multiple radiators active:

- directivity output reflects the combined prescribed excitation
- acoustic impedance is emitted per radiator
- impedance plots label each radiator separately

If a crossover drives a radiator effectively to zero at a frequency, that radiator's impedance value may be stored as `nan` for that point.

