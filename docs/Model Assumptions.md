# Model Assumptions

Boundary Lab is a frequency-domain linear-acoustics application for predicting
loudspeaker radiation and internal acoustic loading. It supports three
physical-system topologies:

- an unbounded exterior region only, solved with BEM;
- one or more bounded tetrahedral air regions only, solved with sparse FEM;
- one or more bounded tetrahedral air regions coupled to an unbounded exterior
  BEM region, with optional lumped electrodynamic transducer equations.

The application infers the path from the regions configured in the **System**
window. It is intended to answer questions about acoustic response,
directivity, source interaction, enclosure and port behavior, and linear
transducer loading. It is not a general structural, thermal, or nonlinear
finite-element package.

## Physical System and Boundary Conditions

Every active mesh surface belongs to an acoustic region and is classified as:

1. **Rigid**: zero normal surface velocity. A bounded rigid wall may optionally
   use the supported locally reacting Miki porous-lining treatment.
2. **Moving**: motion is supplied by one prescribed-velocity or
   electrodynamic component.
3. **Interface**: pressure and normal flux are transferred between a bounded
   FEM region and the exterior BEM region.
4. **Plane-wave tube termination**: the bounded FEM weak form receives a
   first-order outgoing-wave Robin condition. It is not an interface.

The UI defaults every surface to Rigid and does not provide an unassigned or
unused state. This prevents an omitted assignment from silently behaving as an
opening.

An exterior-only model accepts prescribed-velocity components. Interior and
coupled models may use prescribed-velocity components, linear electrodynamic
transducers, or both. A component can own several moving surface groups, and
each group can have a relative velocity weight. This supports idealized motion
profiles such as a dome and surround moving at different amplitudes.

Prescribed-velocity components use a canonical 1 m/s normal-velocity basis.
Electrodynamic components use a canonical 2.83 V basis and a single rigid-body
translation degree of freedom with direct Re, Le, Bl, Mmd, Cms, and Rms
parameters. Their front and rear acoustic surfaces may belong to different
regions and do not need matching meshes.

The electrodynamic model includes linear motor force, back EMF, mechanical
mass, compliance, damping, voice-coil inductance, and reciprocal acoustic
loading. It does not model cone breakup, nonlinear or position-dependent motor
parameters, thermal compression, suspension nonlinearities, or flexible
enclosure structures. See [Coupled Solver](Coupled%20Solver.md) for the precise
supported contract and limitations.

## Linear Excitation Bases and Channel Synthesis

The acoustic solver computes an independent complex reference response for
each active excitation port. Application channels then combine those bases and
apply level, polarity, delay, and idealized analog HPF/LPF transfer functions.
Because this synthesis is linear and occurs after the physical solve, ordinary
channel edits can reuse completed basis data without rerunning BEM or FEM.

When **Normalized Channel Correction** is enabled, Boundary Lab evaluates each
channel's isolated response at the configured horizontal reference angle and
applies a real magnitude correction before the channel controls. This makes
the isolated channel magnitude target flat before crossover shaping. It is not
a complex phase inverse, does not remove propagation or transducer phase, and
does not guarantee that a summed multiway response will be flat.

Passive electrical networks, amplifier source-impedance interaction, feedback
control, and level-dependent processing would alter the physical solve rather
than act as ordinary post-solve channel weights. They are not part of the
current application model.

## Exterior Boundary Integral Model

In the unbounded region, acoustic pressure satisfies the frequency-domain
Helmholtz equation:

$$
\nabla^2 p + k^2 p = 0
$$

where:

- `k = omega / c`;
- `omega = 2 pi f`;
- `c` is the sound speed.

The exterior mesh is the boundary of the acoustic domain. Prescribed outward
normal velocity is converted to Neumann data using the solver's
`exp(-i omega t)` convention:

$$
q = \frac{\partial p}{\partial n} = i \rho \omega v_n
$$

where `rho` is fluid density. Rigid surfaces use `v_n = 0`; moving surfaces use
the component velocity projected onto the face normal. Interface surfaces in a
coupled system receive their normal derivative from the FEM-BEM solution.

Boundary pressure is represented with continuous first-order (`P1`) basis
functions and boundary velocity/flux with discontinuous constant (`DP0`) basis
functions. BEAT Engine assembles dense Galerkin operators and uses direct dense
linear algebra. The Bempp backend uses Bempp-cl operators and GMRES.

The classical exterior Neumann equation can be written as:

$$
\left(K - \frac{1}{2} I\right) p = S q
$$

with single-layer operator `S`, double-layer operator `K`, identity `I`,
unknown boundary pressure `p`, and prescribed Neumann data `q`. This classical
form can become unreliable at fictitious interior resonances. Boundary Lab's
Burton-Miller combined-field path adds the hypersingular and adjoint
double-layer equations to suppress those artifacts, at additional assembly
cost. The shared BEAT Engine implementation is described in [BEAT Engine
Core](advanced/beat-engine-core.md).

After solving the boundary pressure, exterior pressure at observation point
`x` is evaluated from the representation formula:

$$
p(x) = D[p](x) - S[q](x)
$$

where `D` and `S` are the double- and single-layer potentials.

## Interior FEM Model

With bounded regions and no unbounded region, Boundary Lab assembles only the
first-order pressure FEM system and factors it as a sparse matrix on the CPU.
Disconnected chambers remain acoustically separate unless one shared
electrodynamic component couples their moving surfaces. Rigid walls are the
natural boundary condition; homogeneous bulk loss and rigid-backed Miki wall
treatments use the same formulations as a coupled solve.

A plane-wave tube termination imposes the first-order outgoing condition
$\partial p/\partial n=i k p$ for the `exp(-i omega t)` convention. It represents
the characteristic impedance $\rho c$ of a locally uniform tube, not a hidden
exterior region. Its accuracy degrades when higher-order or evanescent modes
are significant at the cut plane. See [Interior FEM Solver](Interior%20FEM%20Solver.md).

## Coupled FEM-BEM Model

Each bounded region uses first-order pressure FEM on selected tetrahedral
volume groups. Rigid boundaries contribute the natural zero-normal-velocity
condition; moving boundaries contribute prescribed or component-coupled loads.
An interface enforces pressure continuity and normal-flux conservation between
its FEM boundary facets and the conforming BEM surface facets.

The current coupled model assumes linear pressure acoustics and the same
density and sound speed in every participating acoustic region. Each bounded
region may have its own homogeneous bulk-loss factor. Bounded rigid walls may
also use the supported locally reacting, rigid-backed Miki porous treatment;
this is not a thermoviscous boundary-layer model.

The coupled application path requires BEAT Engine CPU, Nvidia CUDA, or AMD ROCm.
Production solves exactly eliminate eligible FEM interior degrees of freedom
with a Schur complement and reconstruct the eliminated FEM pressure afterward.
CPU uses UMFPACK for the sparse interior and a CPU dense solve. CUDA performs the
condensed factorization and retained solve on the GPU. ROCm factors the sparse
interior on the CPU, uploads the retained system, and uses rocSOLVER for its dense
solve. These are algebraically equivalent execution strategies for the same
linear model.

## Symmetry

X symmetry uses a positive-X half model; XY symmetry uses a
positive-X/positive-Y quarter model. The active mesh must lie in that
fundamental domain. BEAT Engine includes reflected BEM source contributions
and evaluates the full observation field from the reduced solve.

Boundary Lab determines separately whether each component and FEM-BEM
interface is actually cut by an active symmetry plane. For an electrodynamic
component, selected moving-surface perimeter adjacency determines its surface
completion factor and the number of distinct physical components represented
by symmetry images. These values are inferred from topology rather than chosen
from a manual multiplier.

Symmetry assumes the omitted geometry and excitation are exact mirror images.
It is therefore invalid for asymmetric geometry, material properties,
component parameters, or drive conditions.

## SPL, Directivity, and Balloon Data

Complex pressure magnitude is converted to sound-pressure level with:

$$
\mathrm{SPL} = 20\log_{10}\left(\frac{|p|}{20 \times 10^{-6}}\right)
$$

On-axis plots and on-axis text exports retain absolute SPL for each solved
channel basis after the selected channel correction. Directivity plots are
normalized during visualization to their configured reference angles and then
optionally smoothed and clipped to the selected display range. Polar exports
contain normalized magnitude and relative phase.

When Balloon Sampling is enabled, the solver evaluates approximately
equal-area Fibonacci-sphere directions. Those solved points are used directly
as balloon vertices, and a shared spherical triangulation supplies the display
and export topology. The balloon mesh therefore scales with the requested
sampling precision without first interpolating onto a separate latitude and
longitude grid. Slice plots still interpolate the spherical data along their
requested great-circle directions.

## Practical Limitations

Results remain subject to mesh resolution, geometry and physical-group
quality, interface conformity, numerical precision, observation distance, and
the assumptions above. Dense BEM storage and factorization grow rapidly with
boundary size; fine meshes should be tested at a few representative
frequencies before committing to a long sweep.

Boundary Lab currently does not provide:

- structural diaphragm, suspension, or enclosure-panel FEM;
- nonlinear, transient, thermal, or level-dependent behavior;
- automatic conversion from a conventional T/S set containing Mms to the
  direct Mmd parameter used by the electrodynamic model;
- passive-radiator components;
- arbitrary surface impedance functions or spatially varying volume loss;
- multiple unbounded exterior regions or different fluids within one coupled
  system;
- iterative, fast-multipole, or distributed coupled solving.

Use measurement or a suitable structural/electroacoustic tool where these
effects dominate, and treat Boundary Lab's result as the prediction of the
configured linear acoustic model rather than the complete behavior of a real
loudspeaker.
