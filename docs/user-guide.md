# GUI User Guide

This guide covers the Boundary Lab desktop GUI. Run it from the repository root with:

```bash
blab gui
```

## Main Window

The main window has three working areas:

- the Ath `.cfg` editor on the left
- the mesh preview in the middle
- live plot panels on the right

The bottom control bar contains geometry generation, solve controls, mesh/source configuration, and frequency range settings.

## File Menu

- `New Project`: clears the editor, mesh setup, source setup, preview, and solved data.
- `Save Project`: saves the current project to the active `.blab.json` file.
- `Save Project As`: chooses a new `.blab.json` path.
- `Load Project`: loads editor text, mesh config, and source config.
- `Import .cfg`: imports only Ath config text into the editor.
- `Export .cfg`: exports only the editor contents.
- `Export Plot`: exports generated plot panels as PNG files.
- `Export Polar Data`: exports solved horizontal and vertical polar text files.

Project files do not store solved results or global preferences.

## Generate

Click `Generate` to run the bundled `ath/ath.exe` against the editor text. Boundary Lab writes the temporary `.cfg`, lets Ath generate geometry, cleans the generated mesh, and loads it into the preview.

Ath output is written under:

```text
runs/ath_output
```

## Mesh Config

`Mesh Config` lists the generated Ath mesh and any imported `.msh` files.

The `ath` row is the default generated mesh. It cannot be renamed or removed, but it can be:

- enabled or disabled
- translated in X/Y/Z

Imported `.msh` rows can be enabled/disabled, renamed, removed, and translated.

When imported meshes are enabled, they are included in the preview and solve. If mesh stitching is enabled in Preferences, Boundary Lab can stitch the active generated/imported meshes before solving.

## Source Config

`Source Config` lists mesh surface groups discovered from the active meshes. Use it to mark driven radiator surfaces and set:

- relative level in dB
- polarity
- delay in milliseconds
- high-pass filter
- low-pass filter

Only driven surfaces are excited during the BEM solve. All other surfaces are treated as rigid/unassigned boundaries.

## Preferences

Preferences are global app settings, not project settings.

Useful controls include:

- GMRES tolerance
- worker count
- polar angle step
- smoothing and SPL display range
- mesh stitching settings
- spherical sampling for the balloon plot

Enable spherical sampling before solving if you want `View > Balloon Plot` to be available afterward.

## Solving

Click `Solve` to start the BEM sweep. Plots update as each frequency completes. Click `Stop` to stop after the current in-flight frequency finishes; completed frequencies remain available for plotting/export.

The GUI currently displays:

- horizontal isobar
- vertical isobar
- acoustic impedance
- on-axis frequency response
- spinorama-style curves

## Balloon Plot

After a solve with spherical sampling enabled, open:

```text
View > Balloon Plot
```

The viewer includes:

- rotatable/zoomable 3D directivity balloon
- frequency picker
- SPL color legend
- 6 dB contour lines
- horizontal, vertical, and on-axis guide lines

The balloon viewer uses the frequencies that completed before the solve ended.
