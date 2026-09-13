# Advanced CLI Workflow

### Drive-dependent desktop plots

Real Input Power shows signed RMS terminal power per voltage-only channel and
its total, in watts. It combines the complex current response to all excitations
before computing `Re(V * conj(I))`, including physical driver multiplicities.
Negative channel power is retained. This is electrical input power, not solely
voice-coil heating. Channel voltage and DSP changes update it without a new solve.

Coupled projects with FEM-BEM interfaces request `interface_average_normal_velocity`.
The canonical complex output has axes `(frequency, excitation, interface)` and
units m/s. Metadata identifies each interface, its represented mesh area, and
the bounded-region outward normal. The Interface Particle Velocity plot shows
the magnitude of area-weighted complex normal velocity after channel synthesis,
in RMS m/s. This is net average flow, not maximum local velocity or a port-noise
limit. Area metadata refers to the symmetry-reduced mesh patch; averaging is
unchanged by identical symmetry replication.

Both plots support trace filtering, previous-solve comparisons, image export,
and tabular export. Older results without the new quantity leave interface
velocity unavailable. This output requires a BEAT worker advertising
`interface_average_normal_velocity` in `optional_output_quantities`; use the
updated engine checkout until a release containing this capability is pinned.
Headless artifacts retain independent complex bases without GUI processing.

Boundary Lab still includes command-line tools for mesh cleaning, solving, data preparation, and static plot generation. The GUI is the recommended entry point for normal use.

## Headless Project Workflow

The project CLI loads a complete `.blab.json` physical system and runs the same
BEAT Engine physical-system path used by the desktop application. It supports
exterior BEM, interior FEM, and coupled FEM-BEM-LEM projects without opening Qt.

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

When `stitch_exterior_meshes` is enabled, GUI preview, Build/Identify Interfaces,
mesh reload, and GUI/headless solve preparation use the same exterior-region
assembly. Assign every part to be stitched (including generated waveguides) to
the exterior region in System. Enabled assets outside that region are not
implicitly included, and FEM volume meshes are never stitched.

Preparation transforms exterior parts to project coordinates, stitches them
using `project_preferences.stitch_tolerance_mm`, and then conforms configured
interfaces to the authoritative FEM facets. Closure is checked on the assembled
exterior, allowing edges on active symmetry planes. Intentional cutouts may be
open in source assets but must be closed by the assembly. Missing parts or
unmatched seams still fail validation.

Derived meshes and preparation manifests are written under `runs/`; source
assets, FEM connectivity, component IDs, excitation ports, and channel assignments
remain intact. Manifests record source hashes, transforms, stitch tolerance,
symmetry, and physical-group mappings. The old all-enabled-mesh stitching and
conversion through legacy radiator inputs have been retired. Saved legacy
radiator assignments remain readable for project migration.

The default `--backend beat_auto` always selects a BEAT Engine backend. It probes
the configured Julia CUDA environment and uses `beat_cuda` when CUDA is
functional; otherwise it falls back to `beat_cpu`. Use an explicit
`--backend beat_cpu`, `--backend beat_cuda`, or `--backend beat_rocm` to disable
automatic selection. `beat_cpu` uses the ordinary CPU path for exterior-only
projects, sparse UMFPACK pressure FEM for interior-only projects, and exact FEM
interface condensation for coupled production projects.
The former `beat_cpu_condensed` ID remains a compatibility alias for `beat_cpu`.
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

Exterior-only CUDA solves select direct Burton–Miller system assembly. The
engine assembles the dense system and excitation right-hand sides without
materializing the four boundary-operator matrices, and shares one pivoted LU
factorization across all excitations at each frequency. CPU and ROCm retain
operator-matrix assembly. Result diagnostics record `burton_miller_assembly`
and `factorization_count`.

For numerical or performance comparisons, select the previous CUDA path in a
request overlay with `"solver_options": {"burton_miller_assembly":
"operator_matrices"}`. This option applies to exterior-only solving.

Coupled CUDA solves without full-matrix diagnostics default to combined
Burton–Miller A/C assembly, sparse interface projection, and fused symmetry
images. To compare with individual operators, use `"solver_options":
{"coupled_bem_assembly": "operators"}`. `coupled_bem_image_fusion: false`
keeps separate image launches. `coupled_bem_max_registers` defaults to `0`
(compiler default); values from 32 through 255 cap the fused kernel. A cap of
160 was beneficial on the RTX 2080 Ti benchmark, but is not a universal default.
Frequency diagnostics record the effective assembly mode, fusion, and cap.
Full-matrix diagnostics use individual operators; CPU and ROCm retain their
existing assembly paths. See [Coupled Solver](../Coupled%20Solver.md).

Interior-only projects do not accept exterior point probes or retained BEM
traces. Their FEM nodal pressure is retained automatically and can drive
Interior observation planes.

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
`exp(+i omega t)`.

The current headless interface evaluates arbitrary exterior probe points declared
before the solve. Retained BEM traces and FEM nodal fields provide the data needed
for a future post-solve point-sampling command without repeating the system solve.

Completed interior FEM runs can also be evaluated directly on named tube exits:

```bash
blab project evaluate-fem runs/manifold-check
```

By default the command selects compiled plane-wave termination groups matching
`exit_*` and writes `fem-validation.json` into the run. It reconstructs the
tagged Gmsh triangles and their adjacent tetrahedra, then reports integrated P1
or P2 surface-mean pressure, plane-mode fraction, within-exit phase RMS,
inter-exit phase spread, normal particle velocity, impedance, and local
forward/backward wave estimates. For P2 results, pressure is integrated with
quadratic face quadrature and velocity uses the quadratic tetrahedron gradient
at each face centroid. The default gates are configurable with
`--max-within-phase-rms-deg`, `--max-inter-phase-rms-deg`,
`--max-inter-phase-deg`, and `--min-plane-mode-fraction`; repeat
`--surface-pattern` to select a different set of termination groups.
Use `--split-surface-entities` when one physical exit group contains separate
Gmsh faces, such as a 2x2 septated bundle. The report then evaluates each child
face independently while preserving its parent exit name as a prefix. Four-face
groups are named by geometric row and then column, avoiding mesh-dependent
left/right swaps from microscopic centroid noise.

Interior quadratic FEM accepts matching Gmsh `tetra10` volume cells and
`triangle6` boundary faces. The result artifact preserves both the corner
topology and full ten-node connectivity, and records `element_order` in the FEM
domain metadata. Mixed P1/P2 bounded regions and mismatched volume/boundary
orders are rejected. The current coupled FEM-BEM path remains P1-only; P2 is
supported for pure interior FEM solves.

The evaluator verifies the source mesh SHA-256 before reconstructing named
surfaces. Keep the exact `.msh` used for the solve available at its recorded
path; replacing it makes the run intentionally unevaluable. The report also
records tetrahedron-edge statistics and applies a default minimum of eight
points per wavelength at the 95th-percentile edge and four at the maximum edge.
Override those screening gates with `--min-points-per-wavelength-p95` and
`--min-points-per-wavelength-maximum-edge` only when documenting a deliberate
mesh-convergence policy.

Compare two reports with common frequencies, excitations, and surface names
before accepting a mesh. Frequencies present in only one report are listed but
do not prevent comparison of the shared subset:

```bash
blab project compare-fem runs/coarse/fem-validation-children.json runs/fine/fem-validation-children.json
```

The comparison removes common complex gain and phase, then gates the
area-weighted surface-phase delta, normalized amplitude delta, and plane-mode
fraction delta. This catches accumulated P1 dispersion that a local
points-per-wavelength count can miss in long routed passages.

Interior P1 FEM accepts an experimental `fem_consistent_mass_weight` solver
option between zero (fully row-sum lumped) and one (standard consistent mass,
the default). Any non-default value must be justified by a convergence report;
mass blending changes numerical dispersion and is not itself evidence of
accuracy.

### Speaker package solve and export

Create a complex-pattern level 1 package directly from a project:

```bash
blab project export-speaker speaker.blab.json --output speaker.blabsp --fidelity pattern
```

Create a level 2 package with fixed distributed BEM sources:

```bash
blab project export-speaker speaker.blab.json --output speaker.blabsp --fidelity fixed
```

Estimate Level 3 storage and per-frequency working sets before solving:

```bash
blab project speaker-preflight speaker.blab.json --json
```

Create a level 3 package with the rank-reduced interior model used for mutual
coupling:

```bash
blab project export-speaker speaker.blab.json --output speaker.blabsp --fidelity coupled
```

Level 3 requires a coupled FEM-BEM physical system and exports a parity
Petrov–Galerkin ROM. It temporarily expands X/XY symmetry to a full-domain
system before compilation; the project file is not changed. The ROM preserves the source project's
symmetry: full-domain projects use one general sector, X-symmetric projects use
even-X and odd-X sectors, and XY-symmetric projects use four parity sectors.
Exact-system models remain internal validation oracles and are not offered as
user-facing speaker-package exports.

The command accepts the same `--request`, `--backend`, `--julia-executable`,
`--julia-threads`, and `--events` controls as the headless solve. It forces the
spherical pressure and boundary traces required by the selected fidelity and
publishes the archive atomically only after every requested frequency has
completed.

## Legacy Mesh Workflow

The source-model `blab solve` command and legacy server protocol are retired. Use the
headless project workflow above for new solves. Mesh cleaning and the
`blab prepare` / `blab plot` commands remain available for historical result files.

The physical-system server is available through `blab server` and
`blab project solve --server-url http://127.0.0.1:8765`. The server chooses its BEAT
backend; remote observation planes remain unsupported. Private-network mode
permits LAN HTTP, while hosted mode requires an access key behind provider HTTPS,
a VPN, or SSH. Localhost remains the default. See the
[server setup guide](../Boundary%20Lab%20Server.md) for connection instructions and
the [server developer reference](boundary-lab-server.md) for the remote job contract.

## Clean A Mesh

```bash
blab clean input.msh output_clean.msh --merge-tol 1e-9
```

This merges coincident vertices, removes degenerate or duplicate triangles, and writes Gmsh 2.2 format.

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
