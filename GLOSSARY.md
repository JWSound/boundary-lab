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

**Air body**:
A solid in imported CAD geometry that represents an air volume (rear chamber, port, waveguide throat) and becomes a bounded region.
_Avoid_: Solid, cavity, chamber body

**Exterior surface body**:
A surface body in imported CAD geometry representing the outside of a cabinet or waveguide; it bounds the unbounded region.
_Avoid_: Shell, skin, BEM body

**Closed shell**:
Free faces in imported CAD geometry that close up among themselves without being a solid; the user chooses whether it is an air body or exterior surface.
_Avoid_: Surface solid, watertight body

**Face group**:
A named set of geometry faces that becomes one physical surface group in a mesh.
_Avoid_: Tag, patch, selection

**Target element size**:
The default mesh edge length, sound speed ÷ maximum frequency ÷ elements per wavelength.
_Avoid_: Mesh resolution, density
