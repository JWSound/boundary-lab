# Boundary Lab

Loudspeaker acoustic simulation: a physical system of air regions, their
boundaries, and the devices that drive them, solved with BEM/FEM.

## Physical system

**Region**:
A connected acoustic domain: bounded (interior air, FEM) or unbounded (exterior air, BEM).
_Avoid_: Domain, volume (when meaning the region)

**Boundary**:
The physical role (rigid, moving, interface, …) assigned to one face group of one region.
_Avoid_: Surface condition, BC

**Interface**:
A connection between a bounded region's boundary and the unbounded region's boundary through which sound passes.
_Avoid_: Opening, port (a port is an air body)

**Component**:
A device (ideal velocity source, electrodynamic transducer) that drives one or more moving boundaries.
_Avoid_: Driver (when meaning the model element)

**Source**:
Informal name for a moving boundary owned by a component; not a model element of its own.
_Avoid_: Radiator (ambiguous with the legacy group name)

## Geometry import

**CAD import**:
A geometry design that turns one STEP geometry, its face groups, and sizing into a mesh assembly with physical assignments.
_Avoid_: STEP import (names only the file step), fusion2msh

**CAD bridge**:
A plugin inside a CAD tool that exports the chosen bodies as geometry for CAD import, together with the CAD tool's own identity of each face, so face groups survive CAD edits.
_Avoid_: CAD plugin, exporter, Fusion2Msh

**Air body**:
A solid in imported CAD geometry that represents an air volume (rear chamber, port, waveguide throat) and becomes a bounded region.
_Avoid_: Solid, cavity, chamber body

**Exterior surface body**:
A surface body in imported CAD geometry representing the outside of a cabinet or waveguide; it bounds the unbounded region.
_Avoid_: Shell, skin, BEM body

**Closed shell**:
Free faces in imported CAD geometry that close up among themselves without being a solid; it is guessed to be an air body or exterior surface, and the user can change that.
_Avoid_: Surface solid, watertight body

**Face group**:
A named set of geometry faces that becomes one physical surface group in a mesh. Faces in no other face group form the **walls** face group; each detected interface has its own face group on each side, whose faces detection decides. A face group **needs review** when the current STEP file no longer has all of its stored faces; generation is rejected until it is re-picked or deleted.
_Avoid_: Tag, patch, selection

**Unmeshable face**:
A face of imported CAD geometry that Gmsh's 2D mesher rejects, usually a face malformed in the STEP file. Reading still shows the other faces; generation is rejected until the face is fixed in CAD.
_Avoid_: Bad face, broken face, failed face

**Placement**:
The rotation and translation that put imported CAD geometry into Boundary Lab coordinates.
_Avoid_: Transform, offset, pose, Forward/Up

**On axis**:
The Z direction of Boundary Lab coordinates, the loudspeaker's main radiation axis; **Horizontal** is X and **Vertical** is Y, whose planes through the origin are the symmetry planes.
_Avoid_: Forward, depth axis

**Target element size**:
The default mesh edge length, sound speed ÷ maximum frequency ÷ the global mesh fineness, one per method: the **FEM target element size** sizes air-body volumes and faces, the **BEM target element size** sizes exterior surface faces, and a face on both takes the finer.
_Avoid_: Mesh resolution, density

**Mesh fineness**:
The choice of elements per wavelength at the maximum frequency, as a named level (FEM Fine, Medium, Coarse for uniform interior meshes; BEM Fine, Medium, Coarse for graded surface meshes) or a custom number.
_Avoid_: Mesh density, mesh quality, resolution

**Size override**:
A mesh fineness or edge length set on a face, or on the volume (interior) of an air body, that replaces the target element size there. Only faces and volumes carry one: setting a size on a body or face group sets it on each of its faces (and a body's volume), and a body or face group shows their shared size or "Mixed". A face or volume without one follows its target element size ("Global").
_Avoid_: Local refinement, size field
