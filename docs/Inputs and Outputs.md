# Project Files

Boundary Lab project files use readable JSON and the `.blab.json` extension.

Project files store:

- waveguide design documents, including generator provider ID and provider-owned source
- generated mesh enabled state, scale, and XYZ offset
- imported mesh rows, including absolute `.msh` paths
- whether imported meshes should be stitched into a single solve mesh
- source configuration by surface name

Project files do not store:

- solved BEM results
- exported plots
- solver backend, GMRES tolerance, and Burton-Miller preferences
- generated geometry file contents

## Example Shape

```json
{
  "schema_version": 3,
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
  "source_config_by_name": {}
}
```

## Loading Projects

Loading a project updates the design editor, mesh config, and source config. It does not automatically run a geometry provider or start a solve. Schema v1 and v2 project files are migrated to Ath-backed generator documents when loaded.

If the project references imported mesh files, those paths are expected to exist on the local machine.
Relative mesh and generated-output paths are resolved from the project file's directory, which keeps bundled samples portable.

## Exporting Plot Images

Users can export any of the rendered plots as .png images.

## Exporting Polar Data

Users can export simulated polar data as individual .txt files per angle sampled for horizontal and vertical axes. Channel-basis solves export three tab-separated columns: frequency in Hz, normalized SPL in dB, and relative phase in degrees. The .txt files can be directly imported into tools such as REW and VituixCAD for external analysis.
