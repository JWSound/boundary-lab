# Boundary Lab CLI workflow

## Headless project workflow

**Starting from a saved run?** Go directly to [inspect and query saved results](#inspect-and-query-saved-results). Starting from a `.blab.json` project? Choose the outputs you will need, validate the request, then solve.

The following example runs the bundled sealed box at three frequencies. From the repository root, save this as `request.json`:

```json
{"schema_version": 1, "frequencies_hz": [400, 500, 630]}
```

```bash
blab project validate examples/Simple_Sealed/simple_sealed.blab.json --request request.json --backend beat_cpu --json
blab project solve examples/Simple_Sealed/simple_sealed.blab.json --request request.json --backend beat_cpu --events ndjson --output runs/sealed-check
blab results inspect runs/sealed-check
blab results report runs/sealed-check on-axis-frequency-response
```

Check validation's `solve_kind`, `backend_id`, frequency range, excitation IDs and requested outputs before solving. Use the same request and backend for both commands. Mesh paths resolve relative to the project. The output directory must be new. NDJSON emits one event per line, including completed frequencies and terminal status.

Omit `--request` to use the project's saved sweep and observations. Omit `--backend` for automatic selection, which can launch Julia to probe CUDA and falls back to CPU. Explicit `beat_cpu` skips that probe.

### Solve request overlays

Choose inputs **before** the solve according to the report you want afterward. Overlay coordinates are in project metres (+Z forward).

| Analysis needed | What to request |
| --- | --- |
| On-axis response, isobars, group delay, Spinorama | Keep project observations (default). Group delay needs at least three distinct frequencies. |
| Balloon or spherical Spinorama | Also enable spherical sampling in the project. |
| Saved channel or system response | Include every excitation port assigned to that response. Keeping all ports is the default. Saved flat-target correction needs horizontal polar pressure. |
| Pressure at known exterior points | Add named `probes`. For example, `"probes": [{"id":"front","coordinate_frame":"project","points_m":[[0,0,1]]}]`. |
| Plane positioned after solving | Set `retain` to `["fem_nodal_pressure"]` for interior, `["bem_boundary_traces"]` for exterior, or both for combined. Interior-only solves retain nodal pressure automatically. Fields increase artifact size. |
| On-axis contribution from each coupled interface | Keep project observations, or explicitly retain `interface_radiated_pressure` when `include_project_observations` is false. |

For a sweep, replace `frequencies_hz` with `"frequency_sweep": {"min_hz":20,"max_hz":500,"count":100,"spacing":"log"}`. Spacing may also be `linear`. To restrict the basis, supply `excitation_port_ids` from validation. For a probe-only exterior experiment, `"include_project_observations": false` omits the saved polar and sphere samples.

### Inspect and query saved results

`blab results inspect RUN` lists status, completion mask, manifest frequencies, excitation-to-channel mapping, quantity IDs, axes, units, domains and saved planes. Use those IDs for queries. Frequencies are exact manifest samples. An interrupted run can still supply its committed frequencies.

| Question | Command for the example run |
| --- | --- |
| What happened at one frequency? | `blab results inspect runs/sealed-check --diagnostics 500` |
| Which index is the angle I want? | `blab results domain runs/sealed-check observation:horizontal-polar angle_deg` |
| What is the complex pressure for each excitation there? | `blab results query runs/sealed-check acoustic:pressure:horizontal-polar --frequency 500 --index observation 180` |
| What is the saved system SPL at that index across the sweep? | `blab results query runs/sealed-check acoustic:pressure:horizontal-polar --response system --index observation 180 --view spl --format csv` |

Index 180 is zero degrees on this example's 1° grid. Look up the index for other projects. Raw queries preserve complex excitation bases. Use `--excitation PORT_ID` for one basis or `--response channel:NAME` for a saved drive. Synthesis applies voltage where applicable, gain, polarity, delay and crossover before calculating SPL. If horizontal polar data is missing, `--uncorrected` explicitly omits flat-target correction and still applies channel DSP. JSON returns parallel real and imaginary arrays and uses null for undefined values.

### Extract desktop plots and observation planes

Use `blab results report RUN METRIC` for desktop quantities. JSON is the default. Add `--format svg` for a figure. `blab results report --help` lists the full catalog. [Desktop plot definitions](../User%20Guide.md#plot-views)

- **Drive and limits.** `peak-excursion` gives one-way peak mm. `maximum-spl` needs on-axis pressure, diaphragm velocity and saved Xmax/Pmax ratings on a voltage-only channel. `normalized-impedance` reports complex acoustic load divided by ρcSd. For terminal quantities use `electrical-impedance` or `real-input-power`.
- **Directivity.** Use `spinorama --spherical` for spherical sound power. Select one frequency for a `balloon` SVG. Isobar frequency coordinates describe the interpolated plot grid.
- **Interfaces.** `interface-velocity` gives RMS area-averaged normal velocity. For `interface-radiation`, sum complex source pressures and compare with the independent Combined trace.

For a plane, use an ID from `inspect` (`--plane-id ID`) or supply geometry in a JSON file. For example, save the following as `plane.json` and use a run with the corresponding retained fields:

```json
{"id":"cross-section","type":"combined","center_m":[0,0,-0.1],"orientation_wxyz":[1,0,0,0],"width_m":0.5,"height_m":0.3,"resolution_m":0.025,"display":"spl"}
```

```bash
blab results plane RUN --plane-file plane.json --frequency 500
```

JSON returns project coordinates, region masks, complex pressure and particle velocity, SPL, phase and in-plane intensity. Use `shape_yx` to reshape the grid (X varies fastest) and `valid_mask` to exclude undefined points. Exterior and combined sampling require PyVista (`fields` or `gui` extra) and a local BEAT worker. CPU is the default. Use `--pressure-only` to skip exterior gradients. [Plane geometry and display conventions](../User%20Guide.md#observation-planes)

### Result artifact

For provenance or a custom reader, open `manifest.json` first. It records project and mesh hashes, backend, solver options, phasor convention, completion mask and per-frequency file references. The run also contains the project snapshot, request, compiled system, domain coordinates/topology, and per-frequency JSON descriptors and NPZ arrays. [Run provenance](BEAT%20Run%20Provenance.md) · [Phasor convention](Phasor%20Convention.md)

### Task-specific references

| If you need to… | Start here |
| --- | --- |
| Diagnose invalid mesh assignments or FEM–BEM interfaces | [Physical system model](../Physical%20System%20Model.md) · [Interface preparation](../Inputs%20and%20Outputs.md#conforming-fem-and-bem-interfaces) |
| Set numerical options or interpret solver diagnostics | [CPU operator assembly and quadrature](beat-engine-CPU.md#cpu-operator-assembly) · [CUDA assembly](beat-engine-CUDA.md) · [Coupled solver](../Coupled%20Solver.md) |
| Evaluate interior exits or compare meshes | `blab project evaluate-fem RUN` needs the original mesh at its recorded path. Use `project compare-fem COARSE_REPORT FINE_REPORT` and read its JSON `passed` field. [Interior FEM](Interior%20FEM%20Solver.md) |
| Submit to a server | `project solve --server-url URL` delegates backend selection to the server and omits saved observation planes. [Server setup](../Boundary%20Lab%20Server.md) |
| Export a speaker package | `project speaker-preflight PROJECT --json` estimates Level 3 cost. `project export-speaker PROJECT --output speaker.blabsp --fidelity pattern` also accepts `fixed` or `coupled` fidelity. [Speaker packages](../Inputs%20and%20Outputs.md#exporting-speaker-packages) |
| Clean a mesh or plot historical NPZ files | See `blab clean --help`, `blab prepare --help` and `blab plot --help`. |
