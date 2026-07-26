# Coupled Solver

Boundary Lab includes two execution modes for the coupled FEM–BEM formulation:

- application solves use `Float32/ComplexF32` with the selected BEAT Engine CPU
  or CUDA backend;
- the backend-only Julia CPU reference uses `Float64/ComplexF64` for numerical
  validation and correctness comparisons.

Both modes currently use a direct dense coupled system and are not intended for
large production-scale models.

## Numerical formulation

The bounded acoustic region uses first-order tetrahedral pressure FEM. For each
frequency, it assembles:

$$
A_F = K-k^2M
$$

using analytic P1 tetrahedron stiffness and consistent mass matrices.
Only tetrahedra belonging to the bounded region's selected physical volume
groups are assembled. The selected submesh is compacted and its surface and
interface indices are remapped before matrix construction.
Prescribed normal velocity is converted to the pressure normal derivative

$$
q = i\rho\omega v_n
$$

and integrated with the triangular boundary basis.

The unbounded region uses BEAT Engine Burton–Miller operators. CPU application
solves assemble them on the host. CUDA application solves assemble the BEM
operators, contract the BEM/interface block, and evaluate the exterior field on
the GPU. FEM assembly and the final monolithic coupled LU remain on the CPU. A
conforming interface has:

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

The initial backend intentionally supports a narrow configuration:

- one Gmsh 4.1 ASCII tetrahedral FEM mesh;
- one Gmsh 2.2 ASCII triangular BEM mesh;
- one bounded acoustic region and one unbounded acoustic region;
- one or more tagged tetrahedral volume groups belonging to the bounded region;
- one conforming FEM–BEM interface;
- first-order tetrahedra and triangles;
- ideal `normal_velocity` excitation ports;
- rigid walls outside the moving and interface groups;
- lossless, linear pressure acoustics;
- `Float32/ComplexF32` for application CPU and CUDA solves;
- `Float64/ComplexF64` for the backend correctness reference;
- dense LU for the complete coupled block matrix.

Voltage-driven electrodynamic components, losses, impedance boundaries,
multiple coupled regions and iterative methods are not enabled. CUDA currently
accelerates only the BEM portion of the production coupled path.
Although the physical-system schema reserves several of those roles for future
backends, the coupled backend rejects them before starting Julia rather than
silently treating them as rigid or lossless.

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

The final two validation solves are enabled by default for reference/validation
requests. The application solve path disables the extra BEAT-only replay and
reports the interface diagnostics needed for interactive use.

## Validation

The Julia smoke suite imports the supplied fixtures, checks FEM matrix
properties, solves a prescribed-velocity interior case, compares a generated
sealed cube against its analytic first cavity mode, validates the compiled
interface operators, and optionally runs the complete dense coupled solve.

```powershell
$env:BLAB_RUN_COUPLED_REFERENCE = "1"
julia --project=src/blab/solvers/julia_local `
  src/blab/solvers/julia_local/scripts/smoke_coupled_solver.jl
```

The full fixture validation is opt-in because dense BEM assembly and LU are
deliberately more expensive than ordinary unit tests.

## Performance and benchmarking

The application keeps a Julia worker alive across coupled solves, using eight
threads for production CPU solves and four for CUDA/reference solves. The first
solve still includes Julia compilation, while later solves reuse the loaded and
compiled backend. FEM matrices, interface operators, BEM spaces,
quadrature geometry, singular-correction data, and field geometry are cached
within a frequency sweep. Only frequency-dependent FEM/BEM matrices and the
dense coupled factorization are rebuilt at each frequency.

The end-to-end benchmark uses the same compiled request and result path as the
application:

```powershell
python scripts/benchmark_coupled_solver.py `
  --julia C:\path\to\julia.exe `
  --mode interactive `
  --precision float32 `
  --bem-backend cuda `
  --persistent `
  --repeat 2
```

It separates mesh/cache setup, operator and block assembly, factorization,
right-hand-side solve, exterior-field evaluation, and unreported process/JIT
overhead. A project file containing a physical system can be supplied with
`--project`.

The backend remains a dense direct solver. BEM operator assembly is
quadratic in boundary size and the coupled dense LU is cubic in total unknown
count, so the current improvements reduce overhead but do not make large
production models scalable.

### Precision comparison

The coupled formulation can be exercised independently in `Float64/ComplexF64`
and `Float32/ComplexF32` using identical meshes, quadrature, excitations, and
observation points:

```powershell
julia -t 4 --project=src/blab/solvers/julia_local `
  src/blab/solvers/julia_local/scripts/compare_coupled_precision.jl
```

The comparison reports relative complex-vector errors plus magnitude and phase
deltas for FEM pressure, BEM pressure, interface flux, and exterior pressure.
Magnitude and phase statistics exclude values more than 80 dB below each
quantity's peak by default, avoiding meaningless phase errors at response
nulls. Frequencies, meshes, observation count, and response floor are
configurable from the command line.

## Application integration

When a project contains a configured physical system, the main **Solve** button
compiles it and dispatches it according to the application solver preference:

- **BEAT Engine (CPU)** uses FP32 FEM, CPU BEM assembly, CPU field evaluation,
  and CPU coupled LU;
- **BEAT Engine (CUDA)** uses FP32 FEM, CUDA BEM assembly and field evaluation,
  and CPU coupled LU.

Other solver preferences are rejected for coupled systems rather than silently
falling back to a different backend. Exterior pressure is requested at the same
horizontal, vertical, and optional spherical observation points used by the
existing live plots. Excitation-port responses are routed and combined by
application channel before ordinary channel DSP is synthesized.

Projects without a physical system continue to use the legacy exterior-BEM
workflow. The coupled path currently requires symmetry to be off and
reports unsupported physical roles when a project exceeds the backend's narrow
reference scope.
