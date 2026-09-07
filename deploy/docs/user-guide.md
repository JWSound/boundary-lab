# Boundary Lab Deploy — User Guide

[Setup](../README.md) · [System Model](system-model.md)

Deploy is a desktop prototype for arranging packaged loudspeaker cabinets and
comparing their coverage and loading. It is intended for array design studies,
not rigging approval, amplifier protection, or a prediction of equipment damage.
You do not need to prepare a numerical mesh to use an existing speaker package.

## First study

1. Start Deploy using the [installation instructions](../README.md). The bundled
   S218BP example provides a starting scene; its coarse model is a demonstration,
   not a substitute for a suitably resolved production package.
2. Import the required `.blabsp` speaker package into the library. Importing a
   package does not replace the scene. Choose the active package and add cabinets.
3. Set cabinet positions and orientations, then assign output channels and drive
   settings. Start with Pattern for immediate placement feedback.
4. Position the audience plane and add microphones at representative listening
   positions. Set the frequency and display range for the coverage map.
5. Where supported, select Boundary or Coupled and solve the field. Run a separate
   frequency sweep in the analysis drawer for microphone or speaker responses.
6. Name and capture a completed comparison before changing the layout. Save the
   project configuration separately.

Create new speaker packages in the main Boundary Lab application; see its
[User Guide](../../docs/User%20Guide.md) and
[package documentation](../../docs/Inputs%20and%20Outputs.md).

## Scene and controls

The library contains reusable speaker packages and rigid-mesh assets. The scene
contains their placed instances, microphones, and the audience plane. Selecting
an item exposes its properties. Positions and distances in Deploy are in metres;
angles are in degrees. Scene Y is height, with the ground at Y = 0.

- Add or duplicate cabinets to build the array; each instance has its own pose,
  channel assignment, level, delay, and polarity. Ctrl+D duplicates selected
  boundary objects.
- Use **W** for translation and **E** for rotation. Rotation normally snaps in
  5° increments; hold Alt to rotate without snapping. Microphones translate but
  do not rotate.
- Cabinet and rigid-object corner handles support placement and snapping. Direct
  dragging is ground-parallel; snapping to another height can change elevation.
- Select the audience plane to move or rotate it. **R** enables corner resizing
  of the plane, not scaling of cabinets.
- Use the viewport camera controls to inspect the arrangement; camera movement
  does not change source orientation or acoustic geometry.

Cabinets and rigid objects must remain above ground and separated by 10 mm of
surface clearance. This is a numerical placement restriction, not a rigging or
physical-safety specification. Do not force intersecting or coincident meshes.

Rigid objects represent sound-hard boundaries such as an idealized stage. Import
closed, consistently oriented Gmsh 2.2 ASCII triangle meshes. The default import
scale is 0.001 m per mesh unit (millimetres); check the resulting dimensions.
Pattern ignores these objects. Boundary and Coupled include them, but do not
model material absorption or transmission through them.

## Drive settings

Output-channel processing is combined with each speaker object's settings:
levels and delays add, polarities multiply, and channel mute silences its
assigned speakers. Keep drive settings consistent when comparing layouts.
Changing the number of cabinets at the same per-cabinet level also changes the
total electrical input; it is not automatically an equal-power comparison.

EQ/filter controls are placeholders. Do not assume that stored filter settings
are applied to predictions. The current Coupled drive defaults to 2.83 V RMS per
input port at 0 dB. Pattern and Boundary scale the responses stored in the package;
check package excitation provenance when making absolute cross-method comparisons.

## Choosing a calculation method

Fidelity describes how the cabinet and its surroundings are represented, not a
guaranteed accuracy rating. Package frequency coverage, geometry, and numerical
resolution still matter.

| Method | Includes | Does not provide |
| --- | --- | --- |
| **Pattern — Level 1** | Packaged directional responses, distance/phase propagation, coherent array summation, rigid-ground reflections | Additional scattering from cabinets or scene objects; changes to driver motion from array loading |
| **Boundary — Level 2** | Exterior boundary-element calculation: ground and scene scattering with fixed source normal derivatives | Updated cabinet interiors or transducer response |
| **Coupled — Level 3** | Exterior calculation coupled to reduced cabinet interiors and transducers; loading feedback between cabinets | A fresh full interior FEM solve, nonlinear driver behavior, or guaranteed accuracy independent of the package |

All three methods assume an infinite perfectly rigid ground plane at Y = 0.
Pattern is **not free field** in Deploy: it adds reflected pressure with the
appropriate direction and path delay. Ground cannot currently be disabled or
assigned a material. Ground-level receivers can gain approximately 6 dB relative
to the same nonzero free-field prediction; elevated receivers show interference,
not a constant boost.

Pattern supports mixed speaker packages over their overlapping frequency range.
Boundary and Coupled currently require the Electron desktop app, disk-backed
assets, and all placed speakers using the same package. Boundary requires Level 2
data; Coupled requires a supported parity-ROM Level 3 package. The selected
frequency must be present in the relevant exported data. A disabled method's
tooltip explains the restriction. The desktop currently requests CUDA; it does
not offer the main application's automatic CPU fallback.

## Audience-plane analysis

The frequency selector controls the map's single analysis frequency. Adjust plane
position, size, orientation, and sample resolution to suit the audience area.
Samples below ground are omitted. Increasing sample density improves display
sampling, not the underlying cabinet model or boundary mesh resolution.

Pattern updates immediately. For Boundary/Coupled, use **Solve field** for a
single calculation, or enable live solving to update after scene edits. Live
solving waits around edits and can queue the latest arrangement after an active
solve. It is not an automatic full-frequency sweep.

SPL displays magnitude in dB relative to 20 µPa. The real/imaginary pressure views
show signed components of the complex harmonic pressure. Their animation is a
visualization of a single-frequency solution, not a transient simulation.
Colour limits and banding change presentation, not drive level.

Where reported, **average SPL** is the arithmetic mean of valid sample dB values,
not an energy average. **Spread** is the 90th-minus-10th percentile in dB. These
are spatial sample statistics, not audience-weighted or broadband measures.

## Microphones and speaker plots

The analysis drawer has **Microphones** and **Speakers** sections. The comparison
panel on the left holds named captures; trace controls toggle individual lines.
All response plots use a white background and grey grids. Click and drag inside
a plot to read frequency and vertical-axis coordinates; release to keep the
crosshair and double-click to clear it. This reads axis coordinates, not a snapped
trace value. Electrical impedance also shows the right-axis phase coordinate.

### Microphones

Place point probes at the listening positions of interest. Pattern responses are
available immediately. Select Boundary or Coupled and run the corresponding
pressure sweep to add that method's traces. Sweeps update as frequencies finish;
the Calculate control becomes Stop while running.

Colour identifies a microphone; line style identifies the method: **Pattern
dotted, Boundary dashed, Coupled solid**. Switching the audience-map fidelity
does not discard matching microphone results from another method. A scene edit
can invalidate current results; capture them first if you want a fixed comparison.

For mixed packages, Pattern uses frequencies within the shared band, interpolating
complex package responses as needed. It is not a continuous-frequency measurement.
Near-field warnings indicate extrapolation inside a package's reference radius;
they are not an error bound or a guarantee of accuracy outside that radius.

### Speakers

These quantities require a Coupled sweep. A microphone is not required. Choose
**All Speakers**, **Selection** (follows scene selection), or a named cabinet.
Cabinet-level electrical quantities aggregate its drivers; mechanical/acoustic
quantities show individual transducers.

| Quantity | Interpretation |
| --- | --- |
| Driver excursion | Peak sinusoidal displacement in mm, not peak-to-peak |
| Electrical impedance | Cabinet input impedance magnitude and phase, from applied voltage and summed coil current |
| RMS current | Cabinet input-current magnitude in A |
| Real input power | Electrical real power in W for the current drive |
| Acoustic loading | Dimensionless resistance or reactance of the net opposing acoustic load, normalized by density × sound speed × effective diaphragm area |
| Diaphragm pressure differential | Force-equivalent average differential pressure; RMS/peak and Pa/kPa controls |

Negative active acoustic resistance is possible in a driven array; do not treat
it alone as a failure condition. Pressure differential is not separate front
and rear pressure, local cone stress, or a safe/unsafe threshold. Its Peak option
is sinusoidal amplitude, not a transient or broadband peak. Read loading alongside
excursion, current, and power. Pressure remains meaningful at zero velocity;
impedance may have missing samples there.

Electrical impedance phase currently follows the native solver convention,
whereas acoustic reactance uses the opposite, standard-audio convention. Their
reactive signs must not be compared directly. See the
[convention table](system-model.md#conventions-and-drive).

## Comparisons, captures, and saving

1. Finish the required sweep(s), enter a descriptive comparison name, and choose
   **Capture results**. Captures freeze the available completed sweeps and Pattern
   response, together with configuration and raw sweep data.
2. Edit the scene and calculate the next case. Toggle captures to overlay them;
   each capture retains its own frequency grid.
3. For a single-cabinet baseline, use a one-cabinet scene at the intended height
   and drive, sweep and capture it, then build the comparison array. Remove rigid
   objects if unwanted. This remains a rigid-ground scene, not free field.

Stopping a sweep can leave partial curves visible, but those incomplete sweeps
are not frozen as completed captured results. There is no built-in isolated-cabinet
reference overlay.

**Save project** writes `.blabdeploy.json`: scene configuration, channels, asset
references, and display settings. It does not embed the speaker/mesh files or
save calculated result caches and captures. Keep referenced assets available.
**Download** on a capture writes `.blabanalysis.json` for external retention;
importing that file is not currently supported. Captures are session-only—do not
close the application assuming a project save preserved them.

## Limits and troubleshooting

- If Boundary/Coupled is unavailable, check package type, homogeneous package
  use, disk paths, exported frequency, and CUDA/backend setup.
- If a solve fails, inspect the reported error and verify geometry clearance and
  runtime setup. Moving an object or changing drive invalidates a current solve.
- If speaker plots are empty, select an appropriate subject and run a new Coupled
  sweep. Acoustic quantities also need valid driver parameters and effective area.
- Do not interpret missing samples as zero response. Near-zero denominators,
  unavailable data, or incomplete sweeps can produce gaps.
- Higher fidelity does not include limiter action, thermal compression, excursion
  dependent parameters, structural cone failure, or porous-ground acoustics.
  Broad claims about maximum safe output require information outside these plots.

For numerical assumptions and validation, continue to the [System Model](system-model.md).
