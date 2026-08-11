# Coupled Solver

Boundary Lab's coupled solver models one or more bounded air volumes and the
surrounding unbounded air in one frequency-domain acoustic solve. Each bounded
region is solved with tetrahedral finite elements (FEM), the exterior is solved
with the BEAT Engine boundary element method (BEM), and conforming port
interfaces transfer pressure and normal derivative between them.

The application infers this path when a physical system contains one or more
bounded regions. A system containing only its unbounded exterior instead uses
the established exterior-BEM solver.

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
- exactly one unbounded-air region with one or more BEM meshes;
- zero or more FEM-BEM interfaces into those exterior meshes;
- one or more selected physical volume groups in each bounded region;
- ideal prescribed-velocity components, linear electrodynamic transducers, or
  both;
- one `normal_velocity` port per ideal source or one `voltage` port per
  electrodynamic transducer;
- single-axis rigid-body piston motion with an explicit `motion_axis`;
- one or more FEM or BEM moving boundaries per electrodynamic transducer;
- direct `Re`, `Le`, `Bl`, `Mmd`, `Cms`, and `Rms` transducer parameters;
- rigid boundaries everywhere else except configured interface boundaries;
- linear pressure acoustics, with optional homogeneous bulk loss in every
  bounded FEM air volume;
- `off`, `x`, or `xy` symmetry, with explicit completion/orbit semantics for
  electrodynamic components.

The FEM input must be a Gmsh 4.1 ASCII mesh containing first-order tetrahedra
and triangular boundary facets. The BEM input must be a Gmsh 2.2 ASCII mesh
containing first-order triangles. Boundary Lab applies each mesh's scale and
translation before checking the interface and solving.

When the unbounded region contains several BEM mesh resources, the backend
combines them into one logical BEM discretization without welding their
vertices. Disconnected parts therefore remain topologically independent while
still participating in the same acoustic boundary-integral solve.

Every tagged surface belonging to an active region must have one boundary
assignment. The System editor offers only `rigid`, `moving`, and `interface`;
new surfaces and legacy `unused` assignments default to `rigid`. The coupled
backend supports those same three assignments. An ideal prescribed-velocity
component may drive one or more moving boundaries in bounded FEM regions, the
exterior BEM region, or both; per-boundary motion weights are applied to the
component's canonical velocity. An electrodynamic component may likewise
couple the same rigid-body degree of freedom to moving boundaries in several
FEM regions, as well as to BEM moving boundaries. The independently meshed
front and rear diaphragm surfaces do not need a node-to-node map because they
communicate through the shared mechanical degree of freedom.

A sealed enclosure therefore has no FEM-BEM acoustic interface: its rear
diaphragm is a moving FEM boundary, its front diaphragm is a moving BEM
boundary, and both belong to the same electrodynamic component. The interior
and exterior acoustic domains communicate only through that component's
mechanical degree of freedom. A ported enclosure adds a conforming FEM-BEM
interface at each port mouth; its diaphragm boundaries remain moving
boundaries rather than interface boundaries.

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
   prescribed FEM or BEM radiators, `interface` for both sides of the opening,
   and `rigid` for the remaining walls.
4. In **Interfaces**, use **Build/Identify Interfaces** to create and validate
   every FEM-to-BEM port connection.
5. In **Components**, attach prescribed-velocity or electrodynamic components
   to the appropriate moving boundaries, enter any relative surface-velocity
   offsets and transducer parameters, and assign application channels.
6. Set **FEM Bulk Loss Factor** on any bounded region that requires volume
   damping, and optionally configure **Wall Impedance** on bounded rigid
   surfaces. Select **BEAT Engine (CPU)** or **BEAT Engine (CUDA)** in
   Preferences, then choose full, X-half, or XY-quarter symmetry in **Meshes**.
7. Run the normal application solve.

The component editor accepts direct Re, Le, Bl, Mmd, Cms, and Rms values for an
electrodynamic transducer, plus an automatic or manual motion axis. Component
symmetry completion and physical-driver count are inferred from moving-surface
adjacency to the active symmetry planes rather than selected manually.

The physical-system compiler resolves group names to Gmsh tags, checks boundary
coverage and model relationships, and records explicit interface vertex, face,
and orientation maps before Julia starts. Unsupported boundary roles and
component models are rejected rather than silently substituted.

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

Boundary Lab also monitors enabled imported source meshes when the application
regains focus. If either side of a configured interface changes, it verifies
the dependency and rebuilds the derived conforming BEM mesh when required.

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
A_F = K-k^2M-i k^2M_\eta, \qquad k=\frac{\omega}{c},
$$

where \(M_\eta\) is the consistent FEM mass matrix weighted by the loss factor
of each bounded region. **FEM Bulk Loss Factor** is configured per region in
the System window using the presets `0`, `0.002`, `0.005`, `0.01`, `0.02`, and
`0.05`. `0` reproduces the lossless formulation. With the solver's
\(\exp(-i\omega t)\) convention, the negative imaginary mass term is passive.
Near an isolated lightly damped cavity mode, \(Q\) is approximately
\(1/\eta\).

This remains a deliberately simple frequency-independent volume-loss
approximation. It does not damp the exterior BEM domain or reproduce
frequency-dependent thermoviscous boundary-layer losses. The region values
used by a solve are included in its result diagnostics.

Rigid surfaces of a bounded region may also carry a rigid-backed porous wall
treatment. For layer thickness \(d\), airflow resistivity \(\sigma\), and
\(X=f/\sigma\), Boundary Lab evaluates the Miki characteristic impedance and
complex wavenumber,

$$
\frac{Z_c}{\rho c}=1+0.070X^{-0.632}-i\,0.107X^{-0.632},
\qquad
\frac{k_c}{k}=1+0.109X^{-0.618}-i\,0.160X^{-0.618},
$$

forms the rigid-backed surface impedance
\(Z_s=-iZ_c\cot(k_cd)\) in the conventional \(e^{+i\omega t}\) convention,
and conjugates it for the solver's \(e^{-i\omega t}\) convention. Its surface
admittance \(Y_s=1/Z_s\) contributes

$$
A_F \leftarrow A_F-i\rho\omega Y_s B_\Gamma,
$$

where \(B_\Gamma\) is the consistent P1 surface mass matrix. This is a
locally reacting approximation: it captures frequency-dependent absorption by
a porous lining but not its lateral propagation, mounting gaps, compression,
or frame elasticity. The UI defaults of 30 mm and 5,000 Pa·s/m² are a useful
starting approximation for typical loose loudspeaker cabinet lining, not a
material specification.

Only tetrahedra in the selected physical volume groups are retained. Each
domain's vertices and facets are compacted, and boundary tags are remapped into
a collision-free aggregate namespace before assembly.

A prescribed normal velocity \(v_n\) on a moving FEM or BEM surface is
converted to the implemented pressure normal derivative

$$
q_v=i\rho\omega v_n
$$

and integrated against the triangular P1 boundary basis for FEM surfaces or
inserted directly into the facewise DP0 Neumann data for BEM surfaces. Each
excitation port uses \(v_n=1\ \mathrm{m/s}\) as its canonical basis input;
per-surface velocity weights scale that basis before assembly.

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

Electrodynamic components retain full-driver T/S parameters. Boundary Lab
infers their symmetry representation from the perimeter topology of the
selected moving surfaces on the reduced solve meshes. An active symmetry plane
cuts a driver when the union of its selected surface groups has perimeter edges
on that plane. Adjacent groups are unioned first, so internal seams between
parts such as a dome and surround do not affect the result.

The inferred solver contract contains:

- `symmetry_role` is `complete_representative` when the fundamental domain
  contains one complete representative driver and `fractional_driver` when a
  symmetry plane cuts the same physical driver;
- `fractional_symmetry_axes` records which active planes cut that driver;
- `surface_completion_factor` is \(2^n\), where \(n\) is the number of those
  axes, and multiplies only the generalized pressure-force integral;
- `physical_driver_orbit_count` records the number of distinct identical
  physical drivers represented by symmetry images.

The component editor displays this inference instead of asking the user to
choose a representation. The compiler repeats the inference, replacing any
legacy persisted values so a change in global symmetry cannot leave stale
component scaling. Disconnected selected patches that imply different cuts are
rejected as ambiguous.

The completion factor times the orbit count equals 1, 2, or 4 for off, X, or
XY symmetry respectively. A motion axis must lie in every symmetry plane that
cuts the same physical driver. Velocity and Neumann data are not multiplied:
BEM symmetry images already reconstruct their acoustic effect. Velocity and
current outputs are per physical driver; aggregate current for an orbit is the
reported current times `physical_driver_orbit_count`.

When the component editor infers a motion axis automatically, it first
reconstructs the selected diaphragm's normal-orientation tensor across every
inferred fractional symmetry axis. This removes the lateral bias that can
otherwise arise from averaging only a reduced half or quarter of a curved
diaphragm and guarantees that the inferred rigid-translation axis lies in each
plane that cuts the physical driver.

Both FEM and BEM meshes must lie in the selected positive fundamental domain.
The FEM cut faces have the natural zero-normal-derivative condition represented
by a rigid symmetry surface. BEM operator assembly includes the reflected
source contributions, and exterior-field evaluation includes all symmetry
images. Observation points can therefore cover the full field even though the
solved mesh is reduced.

Interface construction is symmetry-aware: a reduced interface may meet a
symmetry plane, and a reduced BEM surface may remain open on that plane. Open
edges away from an active symmetry plane are rejected.

Symmetry participation is determined separately for each interface. An
interface whose perimeter contains edges on an active symmetry plane is
treated as an open reduced patch on those planes; a complete port mouth away
from the planes retains its ordinary closed perimeter even when the project
uses X or XY symmetry. This allows one reduced model to combine central,
symmetry-cut openings with complete side or top openings.

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
| `bem_boundary_neumann` | Pa/m | excitation, BEM face |
| `interface_normal_derivative` | Pa/m | excitation, interface node |
| `diaphragm_velocity` | m/s | excitation, transducer |
| `voice_coil_current` | A | excitation, transducer |
| `exterior_pressure` | Pa | excitation, observation |

The main application path always requests `exterior_pressure`. For systems
containing electrodynamic transducers it also retains `diaphragm_velocity` and
`voice_coil_current`, from which diaphragm excursion and electrical input
impedance can be derived. These raw complex quantities are assembled into the
in-memory solved-system model even though dedicated reporting plots have not
yet been added. Impedance fields in the legacy live-plot shape remain
unavailable.

When an exterior or combined observation plane is declared, the application
also retains the BEM P1 boundary pressure and DP0 boundary normal derivative.
Together with the frequency-invariant boundary mesh domain, these traces are
sufficient to evaluate additional exterior field points after the solve
without retaining the coupled matrices or factorization.

The solved-system model keeps the original excitation-port basis and stores
frequency-invariant observation coordinates separately from the quantity
arrays. Channel grouping and plot normalization remain presentation steps, so
they do not destroy component-level physical results. Solved-system data is
currently session-only and is not written into `.blab.json` project files.

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

- general impedance boundary kinds or arbitrary impedance functions (the Miki
  rigid-backed treatment on bounded rigid walls is supported);
- spatially varying acoustic material loss within one FEM region (one
  homogeneous bulk-loss factor per bounded region is supported);
- `Mms` input or automatic conversion from conventional T/S parameter sets;
- passive radiators;
- nonideal amplifier/source impedance;
- cone breakup, nonlinear or asymmetric `Bl`, thermal effects, and lossy or
  frequency-dependent voice-coil inductance;
- nonuniform prescribed-motion profiles;
- more than one unbounded exterior region;
- different fluid properties among coupled acoustic regions;
- iterative, fast-multipole, or distributed coupled solution methods;
- Bempp, ROCm, server, or exterior-local execution for a coupled physical system.

## Validation and profiling

The Julia smoke suite checks FEM matrices, a prescribed-velocity interior
solve, a sealed-cavity mode, interface operators, and—when enabled—the full
coupled fixture:

```powershell
$env:BLAB_RUN_COUPLED_REFERENCE = "1"
julia --project=src/blab/solvers/julia_local `
  src/blab/solvers/julia_local/scripts/smoke_coupled_solver.jl
```

The dedicated noncubic-cavity correctness test compares the sparse P1 FEM
matrices with the analytic rigid-wall modes of a 470 x 330 x 220 mm cavity.
Centered patches on the X, Y, and Z walls verify first-axial-mode source
selectivity at nominal 20, 14, and 10 mm element sizes. A sparse block
shift-invert solve checks that the first twelve nonzero modes match the analytic
rectangular-cavity sequence and converge monotonically under refinement. The
test also checks a homogeneous passive bulk-loss factor in the interior helper,
including the solver sign convention, resonant amplitude scaling, half-power
bandwidth, and modal Q for all three axes:

```powershell
julia --project=src/blab/solvers/julia_local `
  src/blab/solvers/julia_local/scripts/test_noncubic_cavity_loss.jl
```

The companion high-frequency dispersion diagnostic samples exact axial and
oblique modes from about 2 to 10 kHz on the same three mesh densities:

```powershell
julia --project=src/blab/solvers/julia_local `
  src/blab/solvers/julia_local/scripts/analyze_noncubic_ppw.jl
```

For these P1 tetrahedral fixtures, conservative sampled limits are about 17
nominal elements per wavelength for 1% modal-frequency error, 12 for 2%, 8 for
5%, and 6 for 10%. The nominal 14 mm mesh is therefore near the 5% boundary at
3 kHz and is not a reliable 5--10 kHz reference. Bulk loss can reduce resonance
Q, but it cannot correct this spatial-dispersion error.

The curved-interface production fixture has a separate targeted convergence
diagnostic. It compares the baseline and independently exported detailed FEM
meshes using overlapping block shift-invert slices, nearest-node transferred
modal assurance, and HF, MF, interface, and wall participation:

```powershell
julia --project=src/blab/solvers/julia_local `
  src/blab/solvers/julia_local/scripts/analyze_curved_fem_convergence.jl
```

Pass `--stats-only` to validate geometry, physical groups, edge lengths, and
tetrahedron quality without running the eigensolves. For the current fixtures,
the detailed mesh reduces median edge length from 3.18 to 1.96 mm while changing
enclosed volume by 0.060%. Targeted modes from 3 to approximately 10.2 kHz pair
across the meshes. Median baseline-to-detailed frequency shifts are 0.20% from
3--5 kHz, 0.41% from 5--8 kHz, and 0.70% above 8 kHz; the corresponding maximum
sampled shifts are 0.44%, 0.59%, and 0.86%. Closely spaced modes can rotate
within their shared modal subspace, so low individual MAC or participation
agreement in a cluster must be assessed at the subspace level before declaring
a mode unmatched.

For the solver's `exp(-i omega t)` convention, this diagnostic loss uses

$$
A_F(eta)=K-k^2(1+i eta)M=K-k^2M-i eta k^2M.
$$

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
