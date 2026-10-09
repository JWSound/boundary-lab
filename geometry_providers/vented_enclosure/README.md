# Vented Enclosure

A bundled Generator Plugin for a rigid rectangular cabinet, one front driver
and one straight-axis circular flared port. It produces a connected interior
air FEM volume, a closed exterior BEM surface and a voltage-driven LEM
transducer. Units in the source and mesh files are metres.

## Use

The first-party plugin is enabled once when introduced. It appears in
**Edit > Generator Plugins...**; existing default choices and subsequent
disabling are preserved. Select **Vented Enclosure** as the default there,
then add a new design (or start a new project). Ath remains available and
existing designs keep their generators.

1. Set **Symmetry Off** in the mesh settings.
2. Set the enclosure, driver and port dimensions in the Generator dock.
   Sliders have matching numeric inputs. The front/profile sketch updates
   immediately; it does not launch meshing.
3. Press **Generate**. Clearance errors appear before meshing starts.
4. Open **System > Components** and edit the driver electrical/mechanical
   parameters. Initial values are illustrative: Re 5.8 Ω, Le 0.5 mH,
   Bl 8 N/A, **dry Mmd 35 g**, Cms 0.4 mm/N, Rms 1.5 N·s/m.
   Mmd excludes air loading; do not substitute measured Mms unchanged.
5. Set an appropriate low-frequency band using the application's frequency
   controls, then solve. Refine both mesh sizes to check convergence.

Regeneration preserves component parameters, channel settings and region
materials. It reasserts the generated wall/driver/interface topology. The
plugin joins an existing physical system's exterior region and otherwise
creates one. Mesh scale and translation apply to the whole assembly. Each
generation uses a new output folder, so cancellation or failure cannot
overwrite the last accepted mesh. Saved projects reopen from their cached
meshes even if the plugin has subsequently been disabled.

## Geometry and physical assumptions

- Width, height and depth are **external** dimensions. All cabinet walls
  share one thickness. The front baffle is at Z=0; the cabinet extends along
  negative Z. X is horizontal and Y is vertical. Both apertures are centred
  at X=0 and have independently adjustable Y positions.
- Moving diameter includes the surround. The surround is a flat annulus,
  the cone a conical frustum, and the dust cap a spherical dome with height
  one quarter of its diameter. They form one zero-thickness diaphragm with
  rigid axial translation along +Z, loaded by both acoustic regions. There
  is no basket, motor obstruction, surround compliance profile or breakup.
- The FEM air volume includes the cavity and entire port passage. The port
  has a cylindrical outer envelope with the specified wall thickness at the
  lips; the curved inner wall therefore becomes thicker towards the throat.
  The internal lip opens into the cavity. The external lip meets the baffle.
- The port mouth is an acoustic interface cap, not a wall. It is meshed once
  and exported into both resources with the same triangles. Driver surfaces
  are likewise shared geometrically. The exterior shell is checked for
  watertightness and oriented outward.
- The initial model uses rigid, lossless walls and linear pressure acoustics.
  Existing System controls can add supported material/loss models. The plugin
  does not predict turbulence, chuffing, nonlinear compression, cabinet
  structural vibration or manufacturing strength.

## Port profile and spreadsheet provenance

The supplied `250911-stvs_port_optimizer_03 (2).xlsx` (STV Beta03,
revision 2025-09-11) informed the circular-arc geometry. In
`iterations+details+grafic`, columns B/D/E/G define length, curvature radius,
NFR and minimum diameter. The main flare follows:

```
R = L / (2 NFR)
r(z) = r_throat + R - sqrt(R² - (z + L/2)²)
```

For NFR=0 the throat is cylindrical. `L` is the total mouth-to-mouth length,
including the lips. With nonzero lip radius `q`, the main arc is trimmed and
joined to circles tangent to both the main arc and the mouth planes. The
roundover centre lies at Z=-q with radial position
`r_throat + R - sqrt((R-q)² - (L/2-q)²)` (mirrored at the inner end).
This uses exact tangent CAD arcs rather than the workbook graph's sampled
roundover approximation. Zero roundover is also supported.

The regression fixture uses the workbook's unrounded result:
L=486.2486466 mm, throat diameter=64.0096353 mm, NFR=0.1310718825,
curvature radius=1854.8930461 mm and exit diameter=96.0144529 mm.

The workbook's separate empirical tuning corrections, Strouhal limits and
iterative SPL/volume optimizer are **not** implemented here. Enter dimensions
from that calculator if desired, then evaluate them with the coupled solver.
No workbook contents are executed and the spreadsheet is not redistributed.

Reference discussion:
https://www.diyaudio.com/community/threads/investigating-port-resonance-absorbers-and-port-geometries.388264/page-54#post-8102675

## Implementation and validation

The plugin lives entirely in `geometry_providers/vented_enclosure/` and uses
the public assembly API. The standard Windows build/restage scripts already
copy this folder into application resources. The backend is Qt-free; a
cancellable subprocess owns Gmsh/OpenCASCADE. It emits Gmsh 4.1 ASCII FEM
and Gmsh 2.2 ASCII BEM resources, source JSON, mesh statistics, a BREP and a
meshing log. Cancellation terminates and reaps only this worker process.

Tests exercise real Gmsh generation, straight/flared and rounded/unrounded
profiles, interface conformity, closed-shell orientation, clearance failures,
cancellation, repeated generation, shared exterior regions, save/reopen and
host editor interactions. A default-mesh CPU smoke solve completed at 30, 50
and 100 Hz with 2.83 V excitation, retaining complex pressure/current/motion
and FEM nodal pressure. This is workflow validation, not mesh convergence or
measurement validation.

`tests/fixtures/geometry_providers/example_box/` is the minimal developer API
fixture. It is excluded from the packaged user examples.
