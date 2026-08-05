# Physical System Model

Boundary Lab represents a loudspeaker as one physical system rather than as a
collection of independent BEM, FEM, and lumped-element simulations. The model
describes the air domains, the surfaces around those domains, the devices that
move some of those surfaces, and the connections that allow sound to pass
between domains.

The central concepts are:

- **Regions** contain acoustic media.
- **Boundaries** assign physical behavior to region surfaces.
- **Interfaces** connect boundary surfaces belonging to different regions.
- **Components** represent devices that drive or respond to moving boundaries.
- **Excitation ports** identify the independent physical inputs used to compute
  unit-response transfer functions.

These concepts form a graph:

```text
Excitation port
      |
      v
Component ---------> Moving boundary
                       |
                       v
                 Acoustic region
                       |
                 Interface boundary
                       |
                       v
                    Interface
                       |
                 Interface boundary
                       |
                       v
                 Acoustic region
```

The editable graph is stored in a Boundary Lab project. Before solving, the
physical-system compiler resolves its named mesh groups, validates its
relationships, and produces an immutable solver contract.

## Application workflow

The main application exposes the physical model through a tabbed **System**
window:

- **Regions** assigns a name, bounded-interior or unbounded-exterior type,
  mesh, and (for bounded regions) physical volume group.
- **Boundaries** assigns each tagged surface as rigid, moving, interface, or
  unused.
- **Interfaces** uses **Build/Identify Interfaces** to match bounded and
  unbounded interface sides. When an imported BEM interface is meshed
  differently, Boundary Lab writes a derived BEM mesh whose interface nodes and
  faces come from the FEM side, then validates conformity and orientation.
  Curved interface interiors are supported when the FEM and BEM perimeters
  describe the same planar opening; the surrounding planar BEM surface may be
  split across multiple Gmsh geometrical entities.
- **Components** attaches prescribed-velocity components to moving boundaries
  and assigns their application channel.

The existing **Meshes** window remains responsible for mesh import, scale, and
translation. Tetrahedral imports bypass the legacy surface cleaner so their
volume connectivity and physical groups are preserved.

Component-to-channel routing is stored as application project state, not in the
physical system or compiled solver contract. The coupled solver returns one
complex reference response per excitation port; responses routed to the same channel
are combined before the existing channel gain, polarity, delay, and filter
settings are applied.

## Regions

A region represents a connected acoustic domain with material properties such
as density, sound speed, and an optional loss model. Region-specific loss
models are reserved by the schema but are not yet supported by the production
solver. A single homogeneous bulk-loss factor for every bounded FEM region is
available separately in application Preferences.

The initial model supports two region kinds:

- `bounded_air`: a finite air volume solved with FEM. Its mesh contains
  tetrahedra and one or more tagged physical volume groups.
- `unbounded_air`: the exterior acoustic domain solved with BEM. Its mesh
  contains the triangular surface bounding the exterior problem.

For example, a vented enclosure may contain:

```text
region:enclosure     bounded_air      FEM tetrahedral mesh
region:exterior      unbounded_air    BEM triangular mesh
```

A region owns the interpretation of its boundaries. The same physical device
may consequently have one boundary facing the enclosure region and another
facing the exterior region.

Regions do not describe how individual surfaces behave. That is the role of
boundary assignments.

## Boundaries

A boundary assigns a physical role to one tagged surface group on one mesh. It
always belongs to exactly one region.

The initial boundary roles are:

- `rigid`: zero normal surface velocity.
- `moving`: motion is supplied by a component.
- `interface`: acoustic state is coupled to another region through an
  interface.
- `impedance`: pressure and normal velocity obey a surface-impedance model.
- `unused`: the surface is deliberately excluded from the active physical
  model.

These are model-level roles. A solver backend must declare which roles it
supports; including a future-facing role in the schema does not imply that the
current numerical backend can solve it.

An example boundary assignment is:

```json
{
  "id": "boundary:rear-diaphragm",
  "name": "Driver rear diaphragm",
  "region_id": "region:enclosure",
  "group": {
    "mesh_id": "mesh:enclosure",
    "dimension": 2,
    "name": "Radiator"
  },
  "kind": "moving",
  "parameters": {}
}
```

This says that the `Radiator` surface group is part of the enclosure region and
that its normal motion comes from a physical component.

Boundaries are deliberately region-local. A surface on an FEM mesh and its
matching surface on a BEM mesh are separate boundary assignments even when
they occupy the same position.

The compiler resolves each group name to its Gmsh physical tag and element
count. Every tagged physical surface used by a region must receive exactly one
boundary role. This prevents an unassigned surface from silently becoming an
unintended opening or rigid wall.

## Interfaces

An interface connects two boundary assignments and transfers acoustic state
between their regions. It represents a relationship between surfaces rather
than another physical surface.

Each FEM-BEM interface has:

- one boundary belonging to a bounded FEM air region;
- one boundary belonging to the unbounded BEM exterior;
- both boundaries assigned the `interface` role;
- a coordinate tolerance used during conformity validation;
- compiled vertex, face, and normal-orientation mappings.

For example:

```json
{
  "id": "interface:port",
  "name": "Enclosure port",
  "bounded_boundary_id": "boundary:fem-port",
  "unbounded_boundary_id": "boundary:bem-port",
  "coordinate_tolerance_m": 1e-8
}
```

The two referenced boundaries may use different mesh tags and different global
vertex numbers. The compiled interface records their explicit correspondence.

At the numerical level, the interface enforces pressure continuity:

$$
p_\mathrm{FEM}=p_\mathrm{BEM}
$$

and conservation of normal velocity or acoustic flux. When both normals point
outward from their respective regions, this is commonly written as:

$$
v_{n,\mathrm{FEM}}+v_{n,\mathrm{BEM}}=0
$$

Coordinate matching alone is not sufficient. The coupled solver must also
transfer quantities correctly between the FEM and BEM basis spaces.

An ordinary open port mouth is therefore an interface, not a component. It
connects two air domains but does not introduce an independent mechanical
device.

Several bounded FEM chambers may connect to different tagged openings in the
same unbounded BEM mesh. When coplanar openings require rebuilding, Boundary
Lab preserves the other interface groups while conforming each FEM-BEM pair.

## Components

A component represents a physical device with behavior beyond the air itself.
It acts through one or more `moving` boundaries.

The initial component kinds are:

- `ideal_velocity_source`: prescribes motion directly for acoustic validation.
- `electrodynamic_transducer`: couples electrical, mechanical, and acoustic
  behavior.
- `passive_radiator`: has mechanical dynamics and acoustic loading but no
  electrical drive.

The coupled backend supports ideal velocity sources and a first linear
electrodynamic model. Passive-radiator behavior remains deferred. The System
editor exposes both supported active component types.

An ideal source might be authored as:

```json
{
  "id": "component:driver",
  "name": "Ideal driver",
  "kind": "ideal_velocity_source",
  "boundary_ids": ["boundary:rear-diaphragm"],
  "parameters": {
    "motion_profile": "uniform"
  }
}
```

The component owns the behavior of its referenced moving boundaries. A moving
boundary must be owned by exactly one component so that two devices cannot
silently prescribe incompatible motion on the same surface.

Each moving boundary may also have a positive relative motion weight. The
System editor displays this value in dB while the physical model stores its
linear amplitude in `boundary_motion_weights`. A dome, inner surround, and
outer surround can therefore share one component while using progressively
lower surface velocity. Electrodynamic coupling applies the same weight to
boundary velocity and generalized pressure force so the coupling remains
reciprocal.

A component may reference multiple boundaries, including moving groups from
different FEM meshes. A full electrodynamic driver, for example, may have
independently triangulated front- and rear-chamber diaphragm boundaries:

```text
Front FEM chamber --- Front diaphragm boundary
                                  |
Excitation port -------- Driver component
                                  |
Rear FEM chamber ---- Rear diaphragm boundary
```

Both acoustic pressures load the same mechanical diaphragm degree of freedom.
The diaphragm surfaces are not acoustic interfaces and do not require matching
nodes: their pressures remain independent and couple only through the
component.
The component converts the resulting pressure difference, motor force, moving
mass, suspension, and damping into boundary motion.

The initial electrodynamic backend model is a single-axis rigid translation
with direct
`re_ohm`, `le_h`, `bl_n_per_a`, `mmd_kg`, `cms_m_per_n`, and
`rms_n_s_per_m` parameters plus a three-component `motion_axis`. `mmd_kg` is
dry moving mass; `Mms` is not accepted. Normal velocity and generalized force
both use each face's projection onto the motion axis. Diaphragm projected area
and pressure force are therefore integrated from the attached moving meshes,
including shaped rigid cones.

Reduced electrodynamic models keep full physical-driver T/S parameters. The
compiler determines whether a driver is cut by each active symmetry plane by
examining perimeter edges in the union of its selected moving surface groups.
It then emits these derived component parameters into the solver contract:

- `symmetry_role`;
- `fractional_symmetry_axes`;
- `surface_completion_factor`;
- `physical_driver_orbit_count`.

The completion factor scales only the recovered full-diaphragm pressure force.
The orbit count describes distinct identically driven physical drivers and is
used when aggregating electrical current or power. These fields may exist in
older project files, but current compilation re-infers and replaces them from
the active symmetry mode and reduced mesh topology.

A passive radiator uses the same general pattern without an excitation port. Its
motion is driven by the pressure difference across its front and rear
boundaries.

## Excitation Ports and Application Synthesis

An excitation port identifies an independently driven physical component
input. It does not contain gain, polarity, delay, crossover filters, or channel
routing.

For an ideal velocity source, the port represents a canonical normal-velocity
input:

```json
{
  "id": "excitation:woofer",
  "name": "Woofer unit normal velocity",
  "component_id": "component:woofer",
  "kind": "normal_velocity"
}
```

For an electrodynamic transducer, the port represents a canonical voltage
input. The current reference amplitudes are `1 m/s` for `normal_velocity` and
`2.83 V` for `voltage`. Passive radiators have no excitation port.

The coupled solver computes one complex reference response for each requested port.
For a linear system, the pressure at an observation point is then:

$$
p(\mathbf{x},f)=\sum_j H_j(\mathbf{x},f)u_j(f)
$$

where $H_j$ is the solved response for excitation port $j$ and $u_j$ is an
application-side complex weight. Gain, polarity, delay, ideal crossover
filters, equalization, and channel mixing are all included in $u_j$ after the
physical solve.

The separation is:

```text
Physical system                 Application synthesis
---------------                 ---------------------
excitation:woofer  <----------  channel assignment
      |                         gain / polarity / delay
component:woofer               HPF / LPF / EQ
      |
boundary:woofer-diaphragm
```

Application synthesis presets remain editable project state, but they are not
part of the compiled physical system or numerical solve request. Changing an
ordinary DSP setting therefore does not invalidate a completed basis solve.

Passive electrical crossover networks, amplifier source-impedance interaction,
feedback control, and nonlinear or level-dependent processing would alter the
physical system rather than merely its excitation weights. They are outside
the intended Boundary Lab coupled-model scope.

## Complete Vented-Enclosure Example

Consider an enclosure represented by a tetrahedral FEM mesh and an exterior
surface represented by a BEM mesh:

```text
excitation:woofer
      |
      v
component:woofer
      |
      +---- boundary:front-diaphragm (moving) ---- region:exterior
      |
      +---- boundary:rear-diaphragm  (moving)
                         |
                         v
                  region:enclosure
                         |
                  boundary:fem-port
                         |
                    interface:port
                         |
                  boundary:bem-port
                         |
                         v
                  region:exterior
```

The remaining enclosure surfaces receive rigid boundary assignments. The
exterior cabinet surfaces also receive rigid assignments.

During a coupled solve:

1. The solver applies the excitation port's canonical reference input.
2. The component determines or prescribes diaphragm motion for that basis.
3. The front diaphragm excites the exterior while the rear diaphragm excites
   the enclosure FEM region.
4. Enclosure pressure and velocity reach the FEM side of the port.
5. The port interface transfers pressure and flux to the BEM side.
6. The BEM region computes the combined exterior loading and radiation.
7. In a dynamic transducer model, the front and rear acoustic loads feed back into the
   component and the complete system is solved simultaneously.
8. Boundary Lab applies channel routing and DSP to the stored complex basis
   responses in the application.

## Common Classification Questions

### Is a diaphragm a component?

The diaphragm surface is a boundary. The driver or passive radiator that
determines its motion is the component.

### Is a port a component?

An explicitly meshed air opening between FEM and BEM regions is an interface.
A reduced-order port model with inertance, resistance, or nonlinear behavior
could later be represented as a component or acoustic multiport.

### Is an enclosure wall a component?

An acoustically rigid wall is a boundary. A flexible enclosure panel would
require a structural component or structural region that is outside the
initial pressure-acoustics model.

### Does an interface own a mesh?

No. Its two boundary assignments refer to mesh surface groups. The interface
owns only their connection and compiled topology mapping.

### Can a component connect two regions?

Yes, indirectly. It references a moving boundary in each region and supplies
the shared device equations that relate their motion and acoustic loading.

## Authoring Model and Compiled Model

The saved project contains human-oriented references:

```text
mesh group name: "Interface"
boundary role:   "interface"
connected to:    "boundary:bem-port"
```

The compiler converts these into solver-oriented data:

```text
physical tag:       3
element count:      180
vertex map:         FEM vertex -> BEM vertex
face map:           FEM facet -> BEM facet
orientation signs: +1 or -1
```

The solver receives the compiled model together with frequencies, requested
unit-basis excitations, output quantities, and numerical options. Each
basis-dependent result declares its excitation IDs and an `excitation` array
axis. The compiled model contains no application DSP settings.

Keeping authoring, compiled, and synthesis models separate allows project
files, numerical backends, and future physics features to evolve independently.

## Model Invariants

The compiler should reject a system when:

- an object identifier is missing or duplicated;
- a region references a mesh of the wrong purpose;
- a bounded region has no tetrahedral physical volume;
- a tagged surface has no boundary assignment or multiple assignments;
- a moving boundary is not owned by exactly one component;
- a component references a non-moving boundary;
- an active component does not have exactly one compatible excitation port;
- a passive component has an excitation port;
- an interface references a boundary that is not marked as an interface;
- the two sides of a FEM-BEM interface are not conforming;
- mesh scale, placement, material properties, or excitation-port kinds are invalid.

These checks keep geometry mistakes and product-model mistakes out of the
numerical backend, where they would otherwise be harder to diagnose.
