# Project Files

Boundary Lab project files use readable JSON and the `.blab.json` extension.

Project files store:

- waveguide design documents, including generator provider ID and provider-owned source
- generated mesh enabled state, scale, and XYZ offset
- imported mesh rows, including absolute `.msh` paths
- whether imported meshes should be stitched into a single solve mesh
- source configuration by surface name
- an optional editable physical-system graph for coupled models
- application-side component-to-channel routing for coupled excitation responses

Project files do not store:

- solved BEM results
- exported plots
- solver backend, GMRES tolerance, and Burton-Miller preferences
- generated geometry file contents

## Example Shape

```json
{
  "schema_version": 5,
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
  "stitch_imported_meshes": false,
  "source_config_by_name": {},
  "component_channel_by_id": {}
}
```

## Loading Projects

Loading a project updates the design editor, mesh, channel, and physical-system
configuration. It does not automatically run a geometry provider or start a
solve. Older project schemas are migrated without inventing a coupled physical
system.

If the project references imported mesh files, those paths are expected to exist on the local machine.
Relative mesh and generated-output paths are resolved from the project file's directory, which keeps bundled samples portable.

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
one another. See [Coupled Reference Solver](Coupled%20Reference%20Solver.md)
for the initial direct FEM–BEM backend and its supported outputs.

## Exporting Plot Images

Users can export any of the rendered plots as .png images.

## Exporting Polar Data

Users can export simulated polar data as individual .txt files per angle sampled for horizontal and vertical axes. Channel-basis solves export three tab-separated columns: frequency in Hz, normalized SPL in dB, and relative phase in degrees. The .txt files can be directly imported into tools such as REW and VituixCAD for external analysis.
