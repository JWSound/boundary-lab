# Project Files

Boundary Lab project files use the `.blab.json` extension.

Project files store:

- waveguide design documents, including generator provider ID and provider-owned source
- generated mesh enabled state, scale, and XYZ offset
- imported mesh rows, including absolute `.msh` paths
- whether the exterior region's mesh parts should be stitched into one solve mesh
- an editable physical-system graph for exterior BEM and coupled BEM/FEM models
- application-side component-to-channel routing for excitation responses

Project files do not store:

- solved BEM results
- exported plots
- solver backend, GMRES tolerance, and Burton-Miller preferences
- generated geometry file contents

## Example Shape

```json
{
  "schema_version": 7,
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
  "component_channel_by_id": {}
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
