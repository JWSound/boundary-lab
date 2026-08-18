# Interior FEM Solver

An authored physical system containing one or more bounded-air regions and no
unbounded-air region is solved as interior-only pressure FEM. No placeholder
exterior region or FEM-BEM interface is required. The local BEAT Engine CPU
path assembles continuous first-order pressure degrees of freedom on the
selected tetrahedral volume groups and solves the sparse system with UMFPACK.

Rigid walls use the natural zero-normal-velocity condition. Moving walls may
be driven by a prescribed normal-velocity component or by the same linear
electrodynamic transducer model used in coupled solves. One electrodynamic
component may own moving surfaces in multiple disconnected FEM regions; its
single mechanical and electrical degrees of freedom then couple those acoustic
domains. Use `boundary_motion_signs` only when a surface must be intentionally
reversed relative to the normal projection of the shared motion axis; ordinary
front/rear faces need no manual sign.

## Plane-wave tube termination

Assign **Plane-wave tube termination** to the end surface of a locally uniform
tube when only the outgoing plane mode should remain. With Boundary Lab's
`exp(-i omega t)` convention, the boundary condition is

$$
\frac{\partial p}{\partial n}=i k p,
\qquad k=\frac{\omega}{c}.
$$

Equivalently, the local normal velocity obeys the characteristic impedance
$p/v_n=\rho c$. In the FEM weak form this contributes $-i k M_\Gamma$, where
$M_\Gamma$ is the P1 surface mass matrix. This is an absorbing boundary
condition, not a connection to a hidden exterior domain.

The approximation is exact for a normally incident plane wave in a uniform
tube. It does not capture higher-order modes, reflections from geometry beyond
the cut plane, radiation impedance of a finite opening, or evanescent near
fields. Place the termination on a straight, uniform section where higher
modes have decayed. Use a real exterior BEM region and interface when those
effects matter.

## Current contract

- Every bounded region references one tetrahedral FEM mesh and all regions use
  the same sound speed and density.
- Supported boundaries are rigid, moving, and plane-wave tube termination;
  rigid walls may also carry the existing Miki lining treatment.
- Interfaces and exterior observation points are invalid without an exterior
  region.
- FEM nodal pressure is retained automatically. Diaphragm velocity and
  voice-coil current are returned for electrodynamic components.
- X and XY symmetry use the same reduced-domain and component multiplicity
  rules as the coupled solver.

The compression-driver fixture at
`examples/compression_driver/compression_driver.blab.json` demonstrates two
disconnected FEM chambers coupled by one electrodynamic diaphragm, with a
plane-wave termination on the front tube.
