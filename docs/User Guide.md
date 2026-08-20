# Main Window

The main window is a dockable workspace containing:

- the waveguide design editor;
- the 3D mesh preview;
- horizontal and vertical isobars, acoustic impedance, on-axis response, and
  spinorama plots;
- the command strip and status readout.

<img src="../assets/mainwindow.png" alt="Boundary Lab main window" width="700">

Panels can be resized, rearranged, floated, or closed. Reopen a closed panel
from the **View** menu.

## Waveguide Design Editor

The waveguide design panel contains the editor supplied by the active geometry
provider. The bundled Ath provider edits Ath `.cfg` text. Use **File > Import
Waveguide Design...** and **Export Waveguide Design...** to exchange the active
design with other tools.

<img src="../assets/scripteditor.png" alt="Ath waveguide design editor" width="300">

Ath scripts can be added, removed, and renamed using the tab controls at the top of the editor pane. To rename a script, double click its tab. Multi-script workflows can be useful for complex multiway designs (see /examples/MultiAth+Mesh_3WayIntegrated), or workflows where you might be comparing outputs on the same script with different values by copy/pasting it into multiple script tabs and enabling/disabling them in the `Mesh Config` window.

## 3D Mesh Preview

The preview displays every enabled generated and imported mesh. Rigid surfaces
are gray, moving surfaces are blue, and FEM-BEM interface surfaces are green.
Mirrored images are shaded darker when X or XY symmetry is active. Hover over a
surface to see its mesh name, physical-group name and tag, and element count.
The lower-right readout reports the active element count and model dimensions.

<img src="../assets/3dviewport.png" alt="3D mesh preview" width="500">

The red, blue, and yellow axes represent X, Y, and Z. The acoustic reference
axis is +Z, and observation points are positioned relative to the global
origin. Use **Meshes** to change a mesh's scale or XYZ translation.

For a coupled system, the two buttons in the preview title bar can isolate the
bounded interior or unbounded exterior region. They are disabled when the
project does not contain the corresponding region type.

## Plot Views

Plots are enabled from the **View** menu and update as frequencies complete
when live plot streaming is enabled. Use the save button in a plot's title bar
to export that plot as a PNG. Horizontal and vertical isobars also provide
buttons to capture and clear contour overlays.

<img src="../assets/plotviewer.png" alt="Boundary Lab plot panel" width="400">

All five plot types share the same mouse interactions:

- Press and drag the left mouse button to position a crosshair. The crosshair
  remains after release; double-click the plot to remove it.
- Hold the right mouse button to show the previous completed solve. Releasing
  the button or moving the pointer outside the plot restores the current solve.

The comparison gesture becomes available after a new solve has replaced a
previous completed result. Multi-curve plots report the raw cursor coordinates
rather than snapping to one curve.

## Menu Bar

### File Menu

<img src="../assets/filemenu.png" alt="File menu" width="250">

- **New Project** clears the current design, mesh, physical-system, channel,
  and solved-result state after prompting for unsaved changes.
- **Save Project** writes to the active `.blab.json` path.
- **Save Project As** selects a new project path.
- **Open Project** loads a `.blab.json` project.
- **Open Recent** lists recently opened projects and can clear that history.
- **Import/Export Waveguide Design** reads or writes the active provider's
  editable design source.
- **Export Polar Data** writes horizontal and vertical response text files.
- **Export On-Axis Data** writes SPL and phase for each solved channel.

The **On-Axis Frequency Response** dock provides a **Traces** menu for showing
or hiding the summed response and individual channels. The phi icon button adds
matching dotted phase traces, wrapped between -180 and 180 degrees, on a right
axis with a fixed -180 to 600 degree display range. Magnitude traces remain solid,
and the summed response is always black. Phase is available when the solve retains
complex channel-basis pressure. On-axis phase removes the propagation time from
the acoustic origin to the configured polar observation distance. On-axis text
exports use the same reference, including any channel delay changes made after
the solve.

The **Transducer Excursion** dock is available for systems containing
electrodynamic transducers. It plots synthesized excursion magnitude in
millimetres for each transducer, with a **Traces** menu for hiding individual
components. Excursion uses the same channel gain, polarity, delay, crossover,
and normalized channel correction as the acoustic response.

All application file and directory pickers share one last-used directory. It
is remembered between application sessions, falls back to an existing folder
if the saved location disappears, and changes only after a selection is
accepted.

Project files do not contain solved results or solver-backend choices. They do
contain reproducibility-related observation and visualization preferences; on
open, Boundary Lab asks before applying values that differ from the current
application settings.

### View Menu

<img src="../assets/viewmenu.png" alt="View menu" width="260">

The View menu shows or hides the design editor, mesh preview, and five plot
panels. **Balloon Plot** opens its own window and is enabled only when the
current solve contains spherical samples.

### Edit and About Menus

**Edit > Preferences** opens application and solve preferences. **About**
contains diagnostic information, donation information, and the bundled help
guide.

## Preferences

<img src="../assets/preferences.png" alt="Preferences window" width="700">

### Solver Config

- **BEM Solver** selects Server, BEAT Engine Nvidia CUDA, BEAT Engine CPU, BEAT
  Engine AMD ROCm, or Bempp OpenCL CPU. All BEAT Engine backends support exterior
  and coupled FEM-BEM systems, including X and XY symmetry. The ROCm backend uses
  GPU-resident operator assembly, dense solve, and exterior field evaluation.
- **Solve Server URL** and **Check Server** configure and query a remote
  exterior-BEM server. A successful health check also updates advertised
  capabilities such as symmetry support.
- **Server access token** is an optional bearer token for an authenticated
  server. Generate or paste it, copy it into the deployment's secret store,
  and keep it safe. Boundary Lab retains it only for the current application
  session.
- **Balloon Sampling** requests Fibonacci-sphere observation points during the
  solve. Without these samples, the Balloon Plot action remains unavailable.
- **Balloon Angle Precision** controls the approximate angular spacing and,
  consequently, the number of solved balloon vertices. The default 2.5-degree
  spacing produces approximately 6,600 points.

### Observation Config

- **Polar Angle Step** controls the solved horizontal and vertical observation
  spacing. Spinorama processing requires 10-degree spacing or finer.
- **Polar Observation Distance** sets the observation radius from the origin.
- **Normalized Channel Correction** applies a per-channel reference-axis
  magnitude correction before channel gain, delay, polarity, and crossover
  filtering.
- **Horizontal/Vertical Normalization Angle** chooses the reference angle used
  for directivity normalization in each plane.
- **Spin Horizontal/Vertical Ref Angle** chooses the reference axes for the
  spinorama on-axis and listening-window curves without changing the early
  reflections or sound-power data.
- **Polar Smoothing** applies fractional-octave smoothing to directivity,
  spinorama, and balloon presentation data.
- **SPL Min/Max** set the displayed and exported directivity clipping range.
- **Isobar Contour Step** selects stepped isobar colors. Set it to 0 dB for a
  smoothly interpolated color map.

### Mesh Config

**Stitch Tolerance** is the maximum distance used when joining adjacent mesh
parts for an exterior region whose stitching option is enabled in **System >
Regions**.

### Application

- **Theme** selects the system, light, or dark appearance.
- **Live Plot Streaming** enables plot refreshes while a solve is running.
- **Live Plot Quality** selects the interpolation density used for those live
  updates. Completed solves are rendered at final quality.

## Command Strip

The command strip contains geometry generation and solve controls, entry points
for project configuration, logarithmic frequency-range controls, and the
current status message.

<img src="../assets/commandstrip.png" alt="Command strip" width="800">

### Generate

**Generate (F7)** runs the active design through its geometry provider. With
Ath, Boundary Lab stages the design script, captures Gmsh geometry from Ath's
blab mode, meshes and cleans it in a cancellable worker, and loads the final
surface mesh into the project and preview. **Stop (Shift+F5)** terminates either
the active Ath process or a Gmsh meshing operation that cannot complete.
Managed generated artifacts are stored below `runs/generated_geometry`.

### Solve and Stop

**Solve (F5)** infers the numerical path from **System > Regions**:

- one unbounded exterior and no bounded regions uses exterior BEM;
- one or more bounded FEM regions and no unbounded region uses interior FEM;
- one or more bounded FEM regions plus one unbounded exterior uses the coupled
  FEM-BEM path.

**Stop (Shift+F5)** requests cooperative cancellation. A backend may finish
the in-flight frequency before stopping; completed frequencies remain
available for plotting and export. BEAT Engine CPU and CUDA keep a persistent
Julia worker between solves, including after ordinary cancellation, so later
solves can reuse the initialized process.

## Meshes

The **Meshes** window lists generated geometry and imported `.msh` files.

<img src="../assets/mesheswindow.png" alt="Meshes window" width="700">

Generated rows are locked to their design documents. Imported rows can be
enabled, disabled, renamed, removed, scaled, translated, or replaced. **Replace
.msh** is available for one selected imported row and preserves that row's
identity so region, boundary, interface, and component references continue to
work when the replacement retains the same physical-group names. `.msh` files
may also be dragged into the table.

Choose **Off**, **X**, or **XY** symmetry here. Reduced meshes must lie in the
positive-X, or positive-X/positive-Y, fundamental domain. Symmetry is available
for local BEAT Engine CPU/CUDA and for servers that advertise support.

When the application regains focus, it checks enabled imported source files
for external changes. Changed BEM and FEM meshes are reloaded automatically.
If a configured interface depends on a changed mesh, Boundary Lab verifies the
pair and rebuilds the derived conforming BEM interface mesh when necessary.

## Channels

Channels apply post-solve synthesis to independently solved component bases.
They are useful for multiway interference and crossover studies.

<img src="../assets/channelconfig.png" alt="Channels window" width="600">

- **Name** identifies the channel used by components.
- **Voltage** sets the nominal voltage shared by electrodynamic transducers on
  the channel. It is unavailable when the channel contains a
  prescribed-velocity source.
- **Trim dB**, **Polarity**, and **Delay ms** apply complex channel weights.
- **HPF/LPF Type** and **Frequency** define idealized analog crossover transfer
  functions.

Applying channel changes resynthesizes existing basis results without running
the acoustic solver again. The ordinary plots refresh immediately; balloon
data is resynthesized only while its window is open.

## System

The **System** window is the physical-model editor for exterior BEM, interior
FEM, and coupled FEM-BEM projects. Every active surface defaults to **Rigid**.
The UI has no unassigned or unused surface state.

### Regions

<img src="../assets/regionswindow.png" alt="System Regions tab" width="700">

For exterior or coupled work, create exactly one **Unbounded Exterior** region
and assign its BEM surface mesh or meshes. Add a **Bounded Interior** for each
FEM chamber, choose its tetrahedral mesh and physical volume group, and
optionally select a homogeneous FEM bulk-loss factor. A project containing
bounded regions but no unbounded region is a valid interior-only FEM project.
If the exterior uses adjoining parts of one continuous surface, enable
**Stitch exterior region meshes**; leave it off for disconnected closed bodies.

An exterior-only system supports prescribed-velocity components. Interior and
coupled systems also support linear electrodynamic components. The Interfaces
tab is enabled only when both bounded and unbounded regions exist.

### Boundaries

<img src="../assets/boundarieswindow.png" alt="System Boundaries tab" width="700">

Classify each region surface as **Rigid**, **Moving**, or—where applicable—
**Interface** or **Plane-wave tube termination**. A bounded rigid surface may
additionally use a rigid-backed porous lining via **Wall Impedance**. The Miki
model accepts lining thickness and airflow resistivity; disabling it restores
the hard-wall condition. A plane-wave termination is a bounded-only anechoic
Robin boundary intended for a locally one-dimensional tube mode.

Moving boundaries must be owned by exactly one component. An FEM and BEM port
mouth uses two interface boundary assignments, one in each acoustic region.

### Interfaces

<img src="../assets/interfaceswindow.png" alt="System Interfaces tab" width="700">

**Build/Identify Interfaces** pairs configured bounded and unbounded interface
surfaces, makes the BEM side conform to the authoritative FEM boundary facets
when needed, and validates node, face, and normal-orientation mappings. The
original imported files are not overwritten. Multiple tagged openings may
share the same surrounding BEM surface. This tab is enabled only when both
bounded FEM and unbounded BEM regions exist.

### Components

<img src="../assets/componentswindow.png" alt="System Components tab" width="700">

Components own one or more moving boundaries and route their solved reference
response to an application channel. The editor supports **Prescribed Velocity**
and **Electrodynamic Transducer** components.

<img src="../assets/componenteditorwindow.png" alt="Component editor" width="500">

Each selected surface has a **Relative Velocity** in dB, allowing a dome and
surround to share a component while using different motion amplitudes.
Ath-generated driver groups are initially seeded as prescribed-velocity
components.

Electrodynamic components use direct Re, Le, Bl, Mmd, Cms, and Rms parameters.
The solver retains a 2.83 V reference basis, while the channel Voltage control
scales that basis after the solve. Their rigid-translation motion axis can be
inferred from the selected surface normals or entered manually. In a symmetry
model, Boundary Lab also infers whether moving surfaces are cut by the active
planes and reports how many distinct components exist in the fully mirrored
system; there is no manual component-symmetry multiplier.

See [Physical System Model](Physical%20System%20Model.md) for the object model
and [Coupled Solver](Coupled%20Solver.md) for numerical requirements and
limitations.

## Observation Planes

Observation-plane properties control the plane geometry, sampling resolution,
result response, frequency, and viewport display metric. Real, imaginary, and
animated instantaneous pressure use a diverging color map centered on zero.

Interior planes can also display **Particle Velocity Magnitude**. Boundary Lab
derives the complex particle-velocity vector from the P1 FEM pressure gradient
using the bounded region density, then colors the plane by its magnitude in
metres per second. Phase animation shows instantaneous particle speed. The
option is unavailable for Exterior and Combined planes because their exterior
BEM field evaluator currently returns pressure only.

The **Pressure color range** is automatic by default and uses a symmetric range
based on the current pressure field. To keep the scale fixed while changing or
sweeping frequency, clear **Automatic** and enter the positive limit in Pa. The
viewport then uses `-limit` to `+limit`; values outside that interval saturate
at the end colors. Reducing the limit makes low-amplitude spatial variation
more visible when a localized pressure peak would otherwise dominate the
automatic scale. The selected manual limit is stored with the observation
plane in the project file.

With a plane selected in the viewport, use **W** for Move, **E** for Rotate,
and **R** for Scale. The manual pressure color limit accepts whole pascal
values and starts at 10 Pa.

## Frequency Controls

Set **Min Hz**, **Max Hz**, and **Frequencies** before solving. Boundary Lab
uses logarithmic spacing between the normalized minimum and maximum values.

## Balloon Plot

After a solve with spherical sampling enabled, choose **View > Balloon Plot**.

<img src="../assets/balloon.png" alt="3D directivity balloon" width="500">

The viewer includes:

- a rotatable and zoomable 3D directivity balloon whose vertices are the
  original Fibonacci-sphere solve samples;
- a frequency slider, SPL color scale, and 6 dB surface contours that update
  while the slider is dragged;
- horizontal, vertical, and on-axis guides;
- a rotatable polar protractor with 30-degree spokes and 6 dB rings;
- radar and isobar slice plots for the current frequency and slice angle;
- the Forward Beam Shape diagnostic plot.

The viewer includes every frequency completed before the solve ended. Use its
**File > Export Balloon Data** action to write `metadata.json`, `topology.npz`,
`spl_db.npy`, and `radius_norm.npy`. The schema preserves the original sample
directions, shared triangle topology, frequency axis, normalized SPL, and
normalized radius values; XYZ surface positions can be reconstructed from the
documented direction and radius mapping. Guide geometry, contours, and slice
plots are not included.

<img src="../assets/forwardbeamshape.png" alt="Forward Beam Shape plot" width="500">

See [Forward Beam Shape Plot](advanced/forward-beam-shape.md) for the meaning
and limitations of that diagnostic.
