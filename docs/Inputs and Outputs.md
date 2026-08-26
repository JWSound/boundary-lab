# Project Files

Boundary Lab project files use the `.blab.json` extension.

Project files store:

- waveguide design documents, including generator provider ID and provider-owned source
- generated mesh enabled state, scale, and XYZ offset
- imported mesh rows, including absolute `.msh` paths
- whether the exterior region's mesh parts should be stitched into one solve mesh
- an editable physical-system graph for exterior BEM and coupled BEM/FEM models
- application-side component-to-channel routing for excitation responses
- authored observation planes, including their transform, size, sampling resolution, and display settings

Project files do not store:

- solved BEM results
- exported plots
- solver backend, GMRES tolerance, and Burton-Miller preferences
- generated geometry file contents

## Example Shape

```json
{
  "schema_version": 8,
  "active_generator_document_id": "design1",
  "generator_documents": [
    {
      "id": "design1",
      "name": "Waveguide",
      "provider_id": "ath",
      "provider_schema_version": 1,
      "source": {"format": "ath_cfg", "text": "..."},
      "mesh_enabled": true,
      "mesh_scale_factor": 0.001,
      "mesh_translation_mm": [0, 0, 0],
      "artifact": null
    }
  ],
  "imported_meshes": [],
  "stitch_exterior_meshes": false,
  "physical_system": {},
  "component_channel_by_id": {},
  "observation_planes": []
}
```

## Loading Projects

Loading a project updates the design editor, mesh, channel, and physical-system
configuration. It does not automatically run a geometry provider or start a
solve. Older source-config projects are converted to an exterior physical
system as soon as their mesh artifacts are available. Pending compatibility
assignments are retained when a generated artifact must be rebuilt first.
Schema-6 projects that stored a global FEM bulk-loss preference are migrated by
copying that value to every bounded acoustic region.

If the project references imported mesh files, those paths are expected to exist on the local machine.
Relative mesh and generated-output paths are resolved from the project file's directory, which keeps bundled samples portable.

The application uses one shared last-used directory for open, save, import,
export, and directory-selection dialogs. The directory is stored in QSettings
between sessions, is updated only after the user accepts a selection, and
falls back to an existing home or working directory if the remembered path is
no longer available.

Enabled imported `.msh` sources are checked when the application regains
focus. If an external tool has changed a BEM or FEM file, Boundary Lab reloads
it. Configured interfaces that depend on the changed mesh are then verified
and, when necessary, rebuilt from the refreshed FEM boundary facets. The
**Replace .msh** action provides an explicit alternative for imported rows and
preserves downstream references when physical-group names remain compatible.

## Conforming FEM and BEM Interfaces

Coupled FEM–BEM models require both meshes to use the same nodes and triangle
connectivity on their shared interface. Boundary Lab includes a command-line
utility for the common case where a tagged planar BEM patch is surrounded by
one planar surface:

```console
blab conform-interface femvolume.msh exterior.msh exterior_conforming.msh
```

The FEM tetrahedron boundary facets are authoritative. The utility copies those
facets into the output BEM mesh, remeshes the surrounding BEM surface to meet
their perimeter, preserves the BEM physical groups, and verifies that the
resulting exterior mesh is watertight. Input physical-group names default to
`Interface` and can be changed with `--fem-interface` and `--bem-interface`.
The original input files are not modified.

See [Physical System Model](Physical%20System%20Model.md) for how mesh groups,
regions, boundaries, interfaces, components, and excitation ports relate to
one another. See [Coupled Solver](Coupled%20Solver.md)
for the initial direct FEM–BEM backend and its supported outputs.

## Exporting Plot Images

Use the save button in any plot panel's title bar to export its current figure
as a `.png` image. Isobar exports are prepared at final interpolation quality.

## Exporting Polar Data

Users can export simulated polar data as individual .txt files per angle sampled for horizontal and vertical axes. Channel-basis solves export three tab-separated columns: frequency in Hz, normalized SPL in dB, and relative phase in degrees. The .txt files can be directly imported into tools such as REW and VituixCAD for external analysis.

## Exporting On-Axis Data

Users can export each solved channel's on-axis response as a tab-separated .txt file containing frequency in Hz, SPL in dB, and phase in degrees. Single-channel solves use a save-file dialog. Multi-channel solves use a directory picker and write one file per channel; the combined system response is not exported. The files use the original solved frequency samples and can be imported into tools such as REW and VituixCAD.

BEAT and canonical solved-result artifacts use the `exp(-i omega t)` phasor
convention. Boundary Lab converts phase shown in plots, observation planes, and
REW/VituixCAD-compatible text exports to the standard audio
`exp(+i omega t)` convention. Channel delays and crossover filters are converted
to BEAT's convention before they are combined with solver-native pressures.

## Exporting Balloon Data

The balloon viewer exports a schema-versioned directory containing:

- `metadata.json`, including coordinate conventions, array shapes, and radius
  reconstruction rules;
- `topology.npz`, containing frequency, angular and Cartesian sample
  directions, and the shared triangular topology;
- `spl_db.npy`, with shape `(frequency, point)`;
- `radius_norm.npy`, with the same shape and values clipped to the configured
  balloon dB range.

The point order is the original solver Fibonacci-sample order. The export does
not contain contour actors, axes, protractor geometry, or radar/isobar slice
plots.

## Exporting Speaker Packages

Use **File > Export Speaker Package...** to configure a package and select
**Solve and Export**. Boundary Lab starts a new physical-system solve with the
raw complex quantities required by the selected fidelity; it does not reuse a
plot-normalized result.

Speaker packages use the `.blabsp` extension and are versioned ZIP64 archives.
They preserve the independent excitation-port basis and the
`exp(-i omega t)` phasor convention.

- **Level 1 — Pattern superposition** contains complex pressure on a spherical
  Fibonacci sampling surface.
- **Level 2 — Fixed distributed sources** contains level 1 data plus the
  exterior BEM surface, continuous P1 pressure, and facewise DP0 pressure
  normal derivative. These traces define the fixed equivalent source
  `D[p] - S[q]`; they are not two simultaneously imposed boundary conditions.
- **Level 3 — Condensed interior coupling** contains levels 1 and 2 plus a
  frequency-indexed boundary macro-model. Its matrices satisfy
  `K z + C p = B u` and `q = D z + E u`, where `p` and `q` use the exported
  exterior mesh's P1-node and DP0-face ordering. The retained state `z`
  includes condensed FEM pressure, interface flux, diaphragm velocity, and
  voice-coil current. Inputs `u` are normalized to 1 V or 1 m/s by port kind.
  An array solver can assemble one exterior BEM problem around multiple
  packages and couple their boundary traces through these equations.
- **Level 3 — Condensed interior coupling** contains levels 1 and 2 plus a
  frequency-indexed boundary macro-model. Its matrices satisfy
  `K z + C p = B u` and `q = D z + E u`, where `p` and `q` use the exported
  exterior mesh's P1-node and DP0-face ordering. The retained state `z`
  includes condensed FEM pressure, interface flux, diaphragm velocity, and
  voice-coil current. Inputs `u` are normalized to 1 V or 1 m/s by port kind.
  An array solver can assemble one exterior BEM problem around multiple
  packages and couple their boundary traces through these equations.

Boundary Lab uses +Z as forward. Exported speaker packages use the array-tool
frame with +Y as forward by applying the proper right-handed rotation
`(x, y, z) -> (x, z, -y)`. Point and direction coordinates are rotated without
reordering samples or changing complex trace values.

For Level 2 solves using X or XY symmetry, Boundary Lab automatically mirrors
the reduced BEM surface and traces into a complete physical boundary before
rotating it into the package frame. Symmetry-plane nodes are shared, coincident
face images are omitted, and reflected triangles have their winding corrected
to preserve outward normals. The scalar P1 pressure and DP0 outward-normal
derivative values are copied to each image. The archive records the source
symmetry, image transforms, and source indices for every exported node and face.
The package manifest records medium properties, units, coordinate conventions,
frequency and excitation order, provenance, payload semantics, and checksums.

Level 3 cannot use an even-symmetry sector because incident fields from other
array elements are generally asymmetric. Boundary Lab therefore recompiles an
export-only full-domain system with symmetry off. It prefers retained generated
full-domain meshes; otherwise it reflects the reduced FEM and BEM meshes,
welds cut-plane nodes, corrects element orientation, removes FEM cut facets,
and rebuilds the physical interfaces. A driver divided by a cut plane remains
one mechanical/electrical state, while complete off-axis drivers are cloned as
independent components and excitation ports. The temporary system and meshes
are discarded after export and are never written back to the project.

Level 3 cannot use an even-symmetry sector because incident fields from other
array elements are generally asymmetric. Boundary Lab therefore recompiles an
export-only full-domain system with symmetry off. It prefers retained generated
full-domain meshes; otherwise it reflects the reduced FEM and BEM meshes,
welds cut-plane nodes, corrects element orientation, removes FEM cut facets,
and rebuilds the physical interfaces. A driver divided by a cut plane remains
one mechanical/electrical state, while complete off-axis drivers are cloned as
independent components and excitation ports. The temporary system and meshes
are discarded after export and are never written back to the project.
