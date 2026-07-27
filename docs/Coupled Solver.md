# Coupled Solver

Boundary Lab's coupled solver models one or more bounded air volumes and the
surrounding unbounded air in one frequency-domain acoustic solve. Each bounded
region is solved with tetrahedral finite elements (FEM), the exterior is solved
with the BEAT Engine boundary element method (BEM), and conforming port
interfaces transfer pressure and normal derivative between them.

The application uses this path whenever the project contains a configured
physical system. Projects without a physical system continue to use the legacy
exterior-BEM workflow.

For the concepts used in the System window, see
[Physical System Model](Physical%20System%20Model.md). For mesh and project-file
details, see [Inputs and Outputs](Inputs%20and%20Outputs.md).

## Production paths at a glance

Coupled application solves require **BEAT Engine (CPU)** or
**BEAT Engine (CUDA)**. Both use `Float32/ComplexF32`, solve all configured
excitation ports as independent reference bases, and stream one result per
frequency.

| | BEAT Engine CPU | BEAT Engine CUDA |
|---|---|---|
| FEM matrices | Sparse assembly on CPU | Sparse assembly on CPU, copied to GPU |
| BEM operators | Assembled on CPU | Assembled on GPU |
| Coupled system | Full monolithic dense system on CPU | Schur-condensed acoustic/electromechanical system on GPU |
| Factorization | CPU dense LU | cuDSS plus GPU dense LU when condensed; GPU dense LU when monolithic |
| Exterior field | Evaluated on CPU | Evaluated on GPU |
| Default Julia threads | 8 | 4 |

CUDA therefore accelerates more than BEM assembly. FEM volume-interior
unknowns are eliminated with an exact Schur complement, the reduced coupled
matrix is assembled and factored on the GPU, and eliminated FEM pressure is
reconstructed after the coupled solve. Electrodynamic models retain the FEM
nodes on their diaphragm surfaces in addition to the port-interface nodes, so
the condensed coupling remains exact.

A separate backend-only **reference path** uses `Float64/ComplexF64` on the CPU.
It retains the full monolithic matrix and enables additional residual and
BEM-replay checks. It exists for correctness testing and is not selectable as
an application solver.

## Model requirements

The production backend currently accepts a deliberately focused physical
system:

- one or more bounded-air regions, currently with one FEM mesh each;
- exactly one unbounded-air region with one BEM mesh;
- one or more FEM-BEM interfaces into that exterior mesh;
- one or more selected physical volume groups in each bounded region;
- ideal prescribed-velocity components, linear electrodynamic transducers, or
  both;
- one `normal_velocity` port per ideal source or one `voltage` port per
  electrodynamic transducer;
- single-axis rigid-body piston motion with an explicit `motion_axis`;
- one or more FEM or BEM moving boundaries per electrodynamic transducer;
- direct `Re`, `Le`, `Bl`, `Mmd`, `Cms`, and `Rms` transducer parameters;
- rigid boundaries everywhere else except configured interface boundaries;
- lossless, linear pressure acoustics;
- `off`, `x`, or `xy` symmetry, with explicit completion/orbit semantics for
  electrodynamic components.

The FEM input must be a Gmsh 4.1 ASCII mesh containing first-order tetrahedra
and triangular boundary facets. The BEM input must be a Gmsh 2.2 ASCII mesh
containing first-order triangles. Boundary Lab applies each mesh's scale and
translation before checking the interface and solving.

Every tagged surface belonging to an active region must have one boundary
assignment. The application rejects `unused` surfaces, and the coupled backend
supports only `rigid`, `moving`, and `interface` assignments. A source used by
an ideal prescribed-velocity component must act on a moving FEM boundary. An
electrodynamic component may couple the same rigid-body degree of freedom to
moving boundaries in several FEM regions, as well as to BEM moving boundaries.
The independently meshed front and rear diaphragm surfaces do not need a
node-to-node map because they communicate through the shared mechanical degree
of freedom.

### Current medium-property rule

The current backend requires the same sound speed and density in every bounded
and unbounded acoustic region. The shared sound speed sets the FEM and BEM
wavenumber, and the shared density converts velocity to pressure normal
derivative. Differing fluids are rejected.

## Setting up a coupled solve

1. Import the tetrahedral FEM mesh or meshes and the triangular BEM mesh.
   Confirm their scale, translation, and physical-group names.
2. In **System > Regions**, create a bounded-air region for each FEM chamber
   and one unbounded-air region. Select each active FEM volume group.
3. In **Boundaries**, classify every active physical surface. Use `moving` for
   prescribed FEM radiators, `interface` for both sides of the opening, and
   `rigid` for the remaining walls.
4. In **Interfaces**, use **Build/Identify Interfaces** to create and validate
   every FEM-to-BEM port connection.
5. In **Components**, attach an ideal prescribed-velocity component to each
   moving FEM boundary and assign its application channel.
6. Select **BEAT Engine (CPU)** or **BEAT Engine (CUDA)** in Preferences. Choose
   full, X-half, or XY-quarter symmetry in Mesh Config.
7. Run the normal application solve.

The current System editor exposes only prescribed-velocity components.
Electrodynamic component validation, serialization, and CPU/CUDA solving are
implemented in the backend contract, but application-side T/S entry controls
are deferred to the next UI phase.

The physical-system compiler resolves group names to Gmsh tags, checks boundary
coverage and model relationships, and records explicit interface vertex, face,
and orientation maps before Julia starts. Unsupported boundary roles and
component models are rejected rather than silently treated as rigid or
lossless.

## Interface requirements

The FEM and BEM sides of the interface must have:

- the same vertex coordinates within the configured tolerance;
- the same triangle connectivity;
- a one-to-one vertex and face correspondence;
- FEM triangles that are actual boundary facets of the selected tetrahedra;
- a recorded sign for the relative FEM and BEM face-normal orientation.

The FEM boundary facets are authoritative. If an imported BEM interface does
not conform, **Build/Identify Interfaces** writes a derived Gmsh 2.2 BEM mesh
and stores that derived mesh in project state. It does not modify the original
mesh or retetrahedralize the FEM volume.

There are two conformance cases:

- If the FEM and BEM interface seams already have matching discrete vertices
  and edges, the interface may be fully curved. Boundary Lab replaces the BEM
  interface directly and leaves its surrounding surface unchanged.
- Otherwise, the two seams must describe the same planar opening. Boundary Lab
  copies the FEM interface facets and remeshes the surrounding planar BEM
  surface to meet their perimeter. That surrounding surface may span multiple
  Gmsh geometrical entities.

When several openings occupy the same planar BEM surface, interface construction
protects the other tagged openings while rebuilding each pair. This allows,
for example, independently meshed front- and rear-chamber ports to connect to
one exterior boundary without consuming one another's physical groups.

Without symmetry, the completed BEM surface must be watertight. With X or XY
symmetry, open BEM boundary edges are allowed only on the active symmetry
planes.

## Numerical formulation

### FEM regions

Each bounded region uses continuous first-order pressure basis functions on
tetrahedra. The disconnected FEM meshes are assembled into a block-diagonal
aggregate pressure system. Boundary Lab assembles analytic stiffness and
consistent mass matrices once for the frequency sweep. At angular frequency
\(\omega=2\pi f\), the FEM Helmholtz matrix is

$$
A_F = K-k^2M, \qquad k=\frac{\omega}{c}.
$$

Only tetrahedra in the selected physical volume groups are retained. Each
domain's vertices and facets are compacted, and boundary tags are remapped into
a collision-free aggregate namespace before assembly.

A prescribed normal velocity \(v_n\) on a moving FEM surface is converted to
the implemented pressure normal derivative

$$
q_v=i\rho\omega v_n
$$

and integrated against the triangular P1 boundary basis. Each excitation port
uses \(v_n=1\ \mathrm{m/s}\) as its canonical basis input.

### BEM region

The exterior uses the BEAT Engine Galerkin Burton-Miller formulation. Exterior
boundary pressure \(p_B\) is represented in a continuous P1 space. The BEM
Neumann data is represented facewise in a discontinuous DP0 space.

The interface introduces a P1 normal-derivative unknown \(q_I\), stored at the
FEM interface vertices. Four sparse transfer operators connect the domains:

- \(G_F\), a consistent FEM boundary-mass operator that loads the FEM equation;
- \(Q_B\), which averages interface nodal derivatives onto BEM DP0 faces and
  applies the recorded normal-orientation signs;
- \(T_F\), which restricts FEM pressure to interface vertices;
- \(T_B\), which restricts BEM pressure to the corresponding interface
  vertices.

### Electrodynamic transducer

Each electrodynamic component adds one rigid-piston velocity \(v_d\) and one
voice-coil current \(I\) to the coupled unknown vector. The required direct
parameters are:

- `re_ohm`;
- `le_h`;
- `bl_n_per_a`;
- `mmd_kg`, explicitly the dry moving mass;
- `cms_m_per_n`;
- `rms_n_s_per_m`;
- `motion_axis`, a three-component translation direction.

`Mms` is deliberately not accepted. Diaphragm area is integrated from the
attached moving mesh surfaces, so a separate `Sd` is not required by the
backend. The axis is normalized by the backend. For a triangle with acoustic
domain outward normal \(\mathbf n\) and normalized motion axis \(\mathbf d\),
the normal velocity and generalized acoustic force use the same projection:

$$
v_n=(\mathbf n\mathbin{\cdot}\mathbf d)v_d,
\qquad
F_\mathrm{ac}=c_d\left[
\sum_{\mathrm{FEM}}\int_{\Gamma_d}p(\mathbf n\mathbin{\cdot}\mathbf d)\,dS
-
\sum_{\mathrm{BEM}}\int_{\Gamma_d}p(\mathbf n\mathbin{\cdot}\mathbf d)\,dS
\right].
$$

This reciprocal projection represents a shaped but rigidly translating cone
correctly. It also makes independently meshed front and rear FEM surfaces load
the same degree of freedom with opposite pressure directions. Optional
per-boundary signs are orientation corrections, not substitutes for the
motion axis. The completion factor \(c_d\) is described under
[Symmetry](#symmetry).

With the solver's time convention, the electrical and mechanical impedances
are

$$
Z_e=R_e-i\omega L_e,
$$

$$
Z_m=R_\mathrm{ms}
+i\left(\frac{1}{\omega C_\mathrm{ms}}-\omega M_\mathrm{md}\right).
$$

The voltage and force equations are

$$
Z_e I+Bl\,v_d=V,
$$

$$
Z_m v_d-Bl\,I-F_\mathrm{ac}=0.
$$

The reference voltage is currently fixed by the backend at \(2.83\ \mathrm{V}\)
with zero source impedance. The same solved \(v_d\) generates every attached
surface's projected normal derivative, so pressure loading, diaphragm motion,
coil current, back-EMF, and radiation are solved bidirectionally.

### Coupled block system

For the CPU production path and the FP64 reference path, each frequency uses
the monolithic system

$$
\begin{bmatrix}
A_F & 0 & -G_F \\
0 & A_B & -R_BQ_B \\
T_F & -T_B & 0
\end{bmatrix}
\begin{bmatrix}
p_F \\
p_B \\
q_I
\end{bmatrix}
=
\begin{bmatrix}
f_v \\
0 \\
0
\end{bmatrix}.
$$

Here \(A_B\) is the Burton-Miller pressure operator and \(R_B\) is its Neumann
right-hand-side operator. The third block row enforces nodal pressure
continuity. Flux continuity is built into the shared \(q_I\) unknown and its
orientation-aware transfer to the BEM faces.

With \(N_T\) electrodynamic transducers, the monolithic system extends to

$$
\begin{bmatrix}
A_F & 0 & -G_F & -i\rho\omega D_F & 0 \\
0 & A_B & -R_BQ_B & -R_B(i\rho\omega D_B) & 0 \\
T_F & -T_B & 0 & 0 & 0 \\
-B_F^\mathsf{T} & B_B^\mathsf{T} & 0 & Z_m & -Bl \\
0 & 0 & 0 & Bl & Z_e
\end{bmatrix}
\begin{bmatrix}
p_F \\ p_B \\ q_I \\ v_d \\ I
\end{bmatrix}
=
\begin{bmatrix}
f_v \\ 0 \\ 0 \\ 0 \\ V
\end{bmatrix}.
$$

\(D_F\) maps piston velocity into projected FEM surface loads and \(D_B\) maps
it to BEM DP0 Neumann data. \(B_F\) and \(B_B\) contain the reciprocal
projected pressure-force integrals, including any reduced-driver completion
factor. \(A_F\) is block diagonal when several bounded FEM domains are present.
Opposite acoustic-domain force conventions make front and rear pressure act on
the same mechanical degree of freedom.

If the FEM mesh has \(N_F\) vertices, the BEM mesh has \(N_B\) vertices, and the
interface has \(N_I\) vertices, the monolithic matrix order is
\(N_F+N_B+N_I+2N_T\).

The CUDA production path retains the union of FEM-BEM interface nodes and
electrodynamic diaphragm nodes. All other FEM nodes are eliminated with an
exact sparse Schur complement. If that retained set has \(N_R\) nodes, the
dense coupled matrix has order
\(N_R+N_B+N_I+2N_T\). For a prescribed-velocity-only model,
\(N_R=N_I\), giving \(N_B+2N_I\). After the reduced system is solved, a
backward sparse solve reconstructs every FEM domain's pressure. Static
condensation changes the work and memory requirements, not the mathematical
solution.

CPU production and FP64 reference solves remain monolithic. CUDA
electrodynamic solves use the retained-surface condensed formulation.

The matrix is assembled and factored once per frequency. All requested
excitation ports are then solved together as multiple right-hand sides, so an
additional requested source adds a response column rather than another
factorization.

## Symmetry

Both production backends support:

- `off`: full model;
- `x`: positive-X half model, mirrored across \(X=0\);
- `xy`: positive-X/positive-Y quarter model, mirrored across \(X=0\) and
  \(Y=0\).

Electrodynamic components use explicit symmetry semantics while retaining
full-driver T/S parameters:

- `symmetry_role` is `complete_representative` when the fundamental domain
  contains one complete representative driver and `fractional_driver` when a
  symmetry plane cuts the same physical driver;
- `fractional_symmetry_axes` records which active planes cut that driver;
- `surface_completion_factor` is \(2^n\), where \(n\) is the number of those
  axes, and multiplies only the generalized pressure-force integral;
- `physical_driver_orbit_count` records the number of distinct identical
  physical drivers represented by symmetry images.

The completion factor times the orbit count must equal 1, 2, or 4 for off, X,
or XY symmetry respectively. A motion axis must lie in every symmetry plane
that cuts the same physical driver. Velocity and Neumann data are not
multiplied: BEM symmetry images already reconstruct their acoustic effect.
Velocity and current outputs are per physical driver; aggregate current for an
orbit is the reported current times `physical_driver_orbit_count`.

Both FEM and BEM meshes must lie in the selected positive fundamental domain.
The FEM cut faces have the natural zero-normal-derivative condition represented
by a rigid symmetry surface. BEM operator assembly includes the reflected
source contributions, and exterior-field evaluation includes all symmetry
images. Observation points can therefore cover the full field even though the
solved mesh is reduced.

Interface construction is symmetry-aware: a reduced interface may meet a
symmetry plane, and a reduced BEM surface may remain open on that plane. Open
edges away from an active symmetry plane are rejected.

Symmetry reduces the FEM, BEM, and interface unknown counts before the coupled
matrix is built. This is especially valuable because the BEM operators and the
final coupled factorization are dense.

## Excitations, channels, and application results

For each frequency, the backend computes one complex response for every
requested excitation-port ID. The application currently requests exterior
pressure at:

- the horizontal polar observation points;
- the vertical polar observation points;
- optional Fibonacci-sphere points used by balloon plots.

Responses assigned to the same application channel are summed before ordinary
channel synthesis. Gain, polarity, delay, high-pass and low-pass filters,
and plot normalization are applied after the physical solve. Changing those
settings does not change the coupled matrices.

The backend protocol also recognizes these quantities for validation and
specialized callers:

| Quantity | Unit | Array axes |
|---|---:|---|
| `fem_nodal_pressure` | Pa | excitation, FEM node |
| `bem_boundary_pressure` | Pa | excitation, BEM node |
| `interface_normal_derivative` | Pa/m | excitation, interface node |
| `diaphragm_velocity` | m/s | excitation, transducer |
| `voice_coil_current` | A | excitation, transducer |
| `exterior_pressure` | Pa | excitation, observation |

The main application path currently requests only `exterior_pressure` and does
not yet project transducer velocity, current, or derived electrical impedance
into its plots. Impedance fields in the legacy live-result shape remain
unavailable.

## Diagnostics

Every production frequency result identifies its precision, BEM backend,
linear backend, symmetry mode, formulation, linear solver, full system order,
solved system order, and detailed stage timings. It also reports the maximum
across excitation bases of:

- relative pressure-continuity error at the interface;
- relative integrated interface-flux conservation error.

For multi-port models these are the worst values over the individual
interfaces, not one potentially cancelling aggregate. Per-interface IDs and
error arrays are included in the diagnostics.

Production solves deliberately do not retain the complete coupled matrix after
factorization. Consequently, they do not report a full coupled-system residual.
The FP64 reference path keeps the matrices and additionally reports:

- the monolithic relative residual;
- error against a separate BEM-only replay driven by the solved interface
  Neumann data.

This distinction matters when reading status messages or benchmark output: the
interactive application normally shows interface continuity, while validation
runs can show a full residual.

## Performance and practical limits

The application keeps a Julia worker alive between solves. The first solve in a
session includes Julia loading and compilation; later solves reuse the worker.
Within one frequency sweep, Boundary Lab caches frequency-invariant data,
including:

- FEM stiffness and mass matrices;
- interface transfer operators;
- BEM P1 and DP0 spaces;
- quadrature and singular-correction geometry;
- symmetry-image and field-evaluation geometry;
- CPU or GPU assembly support data.

The frequency-dependent FEM Helmholtz matrix, BEM operators, coupled blocks, and
factorizations are rebuilt at every frequency.

The solver is still limited by dense BEM and coupled algebra. BEM assembly grows
approximately quadratically with boundary size. CPU monolithic LU grows
approximately cubically with \(N_F+N_B+N_I+2N_T\). CUDA condensation removes FEM
interior nodes from the dense block, but the remaining interface, diaphragm,
BEM, and transducer system is still dense and requires substantial GPU memory.
Symmetry can reduce these costs dramatically for both prescribed-velocity and
electrodynamic models. The current backend is not an iterative or large-scale
fast-multipole solver.

Before committing to a fine sweep, test a few representative frequencies and
watch the reported system order, assembly time, factorization time, and
available host or GPU memory.

## Unsupported physics

The physical-system schema includes roles intended for future solvers. The
following are not implemented by the current coupled backend and are normally
rejected during application preparation or backend validation:

- impedance or parameterized boundaries;
- acoustic region loss models;
- `Mms` input or automatic conversion from conventional T/S parameter sets;
- passive radiators;
- nonideal amplifier/source impedance;
- cone breakup, nonlinear or asymmetric `Bl`, thermal effects, and lossy or
  frequency-dependent voice-coil inductance;
- nonuniform prescribed-motion profiles;
- more than one unbounded exterior region;
- different fluid properties among coupled acoustic regions;
- iterative, fast-multipole, or distributed coupled solution methods;
- Bempp, ROCm, server, or legacy-local execution for a physical system.

## Validation and profiling

The Julia smoke suite checks FEM matrices, a prescribed-velocity interior
solve, a sealed-cavity mode, interface operators, and—when enabled—the full
coupled fixture:

```powershell
$env:BLAB_RUN_COUPLED_REFERENCE = "1"
julia --project=src/blab/solvers/julia_local `
  src/blab/solvers/julia_local/scripts/smoke_coupled_solver.jl
```

The end-to-end benchmark exercises the compiled request and streamed-result
path. This CPU example uses the same FP32 monolithic formulation as an
interactive CPU application solve:

```powershell
python scripts/benchmark_coupled_solver.py `
  --julia C:\path\to\julia.exe `
  --mode interactive `
  --precision float32 `
  --bem-backend cpu `
  --persistent `
  --repeat 2
```

For CPU precision studies, the dedicated comparison runs identical meshes,
quadrature, excitations, and field points through `Float64/ComplexF64` and
`Float32/ComplexF32`:

```powershell
julia -t 4 --project=src/blab/solvers/julia_local `
  src/blab/solvers/julia_local/scripts/compare_coupled_precision.jl
```

The comparison reports relative complex-vector errors plus magnitude and phase
deltas for FEM pressure, BEM pressure, interface derivative, and exterior
pressure. Values far below each quantity's peak are excluded from magnitude and
phase summaries so response nulls do not dominate the statistics.
