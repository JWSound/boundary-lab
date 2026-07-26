# Coupled Reference Solver

Boundary Lab includes a double-precision Julia CPU backend for validating the
initial coupled FEM–BEM formulation. It is a correctness reference, not a
production-scale solver.

## Numerical formulation

The bounded acoustic region uses first-order tetrahedral pressure FEM. For each
frequency, it assembles:

$$
A_F = K-k^2M
$$

using analytic P1 tetrahedron stiffness and consistent mass matrices.
Prescribed normal velocity is converted to the pressure normal derivative

$$
q = i\rho\omega v_n
$$

and integrated with the triangular boundary basis.

The unbounded region uses BEAT Engine's double-precision CPU
Burton–Miller operators. A conforming interface has:

- P1 pressure traces on the FEM and BEM interface vertices;
- a P1 interface normal-derivative unknown;
- a consistent FEM boundary-mass operator;
- a face-average projection into BEAT Engine's DP0 Neumann space;
- per-face normal-orientation signs.

The direct block system is:

$$
\begin{bmatrix}
A_F & 0 & -G_F \\
0 & A_B & -R_BQ_B \\
T_F & -T_B & 0
\end{bmatrix}
\begin{bmatrix}
p_F \\ p_B \\ q_I
\end{bmatrix}
=
\begin{bmatrix}
f_v \\ 0 \\ 0
\end{bmatrix}
$$

The matrix is assembled and factored once per frequency. Every requested
excitation port is then solved as a separate right-hand side, producing reusable
complex transfer-function data.

## Supported input

The initial backend intentionally supports a narrow reference configuration:

- one Gmsh 4.1 ASCII tetrahedral FEM mesh;
- one Gmsh 2.2 ASCII triangular BEM mesh;
- one bounded acoustic region and one unbounded acoustic region;
- one conforming FEM–BEM interface;
- first-order tetrahedra and triangles;
- ideal `normal_velocity` excitation ports;
- rigid walls outside the moving and interface groups;
- lossless, linear pressure acoustics;
- `Float64` and `ComplexF64` throughout;
- dense LU for the complete coupled block matrix.

Voltage-driven electrodynamic components, losses, impedance boundaries,
multiple coupled regions, iterative methods, and GPU execution are not enabled
in this reference backend.

## Output quantities

The backend currently recognizes:

- `fem_nodal_pressure`;
- `bem_boundary_pressure`;
- `interface_normal_derivative`;
- `exterior_pressure`, with observation coordinates supplied through
  `options.points_m`.

Every basis-dependent array declares `excitation` as its first axis and returns
the corresponding physical excitation-port IDs. Application gain, polarity,
delay, channel routing, and ideal DSP are applied after solving.

Each frequency also reports:

- coupled-system relative residual;
- pressure-continuity error;
- integrated interface-flux conservation error;
- error against a BEAT-only replay using the solved interface Neumann data.

## Validation

The Julia smoke suite imports the supplied fixtures, checks FEM matrix
properties, solves a prescribed-velocity interior case, compares a generated
sealed cube against its analytic first cavity mode, validates the compiled
interface operators, and optionally runs the complete dense coupled solve.

```powershell
$env:BLAB_RUN_COUPLED_REFERENCE = "1"
julia --project=src/blab/solvers/julia_local `
  src/blab/solvers/julia_local/scripts/smoke_coupled_reference.jl
```

The full fixture validation is opt-in because dense BEM assembly and LU are
deliberately more expensive than ordinary unit tests.
