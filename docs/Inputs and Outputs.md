# Project Files

Boundary Lab project files use readable JSON and the `.blab.json` extension.

Project files store:

- Ath `.cfg` editor text
- generated Ath mesh enabled state and XYZ offset
- imported mesh rows, including absolute `.msh` paths
- whether imported meshes should be stitched into a single solve mesh
- source configuration by surface name

Project files do not store:

- solved BEM results
- exported plots
- global preferences
- generated Ath output files

## Example Shape

```json
{
  "schema_version": 1,
  "ath_config_text": "...",
  "ath_mesh": {
    "name": "ath",
    "source_file": "E:/AthGUI/boundary-lab/runs/ath_output/case/ABEC_FreeStanding/case_clean.msh",
    "cleaned_file": null,
    "translation_mm": [0, 0, 0],
    "enabled": true
  },
  "imported_meshes": [],
  "stitch_imported_meshes": false,
  "source_config_by_name": {}
}
```

## Loading Projects

Loading a project updates the editor, mesh config, and source config. It does not automatically run Ath or start a solve.

If the project references imported mesh files, those paths are expected to exist on the local machine.
Relative mesh and generated-output paths are resolved from the project file's directory, which keeps bundled samples portable.

## Exporting Plot Images

Users can export any of the rendered plots as .png images.

## Exporting Polar Data

Users can export simulated polar data as individual .txt files per angle sampled for horizontal and vertical axes. Channel-basis solves export three tab-separated columns: frequency in Hz, normalized SPL in dB, and relative phase in degrees. The .txt files can be directly imported into tools such as REW and VituixCAD for external analysis.
