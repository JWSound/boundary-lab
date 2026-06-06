# Model Assumptions

Boundary lab is a BEM-only solver application. As a result, it is important to define what it can and cannot accomplish in the context of loudspeaker design. The fundamental use-case of Boundary Lab is to answer the question:

"Given a loudspeaker enclosure and/or waveguide design, what is the resultant normalized directivity produced by the radiating elements contained in the device"

Or:

"Given a multiway loudspeaker enclosure, what is the resultant normalized directivity produced by all radiating elements assuming idealized analog crossover transfer functions, plus user-defined gain, polarity, and delay"

## Source Definition & Driving Model
The application allows for two types of boundaries to be defined:

1. `Rigid`: An infinitely rigid reflective surface with zero normal velocity.
2. `Driven`: A surface with prescribed normal velocity according to an idealized radiator model.

Because each driven source is an idealized model of a perfect radiator, it is important to denote that Boundary Lab does not support electroacoustic modeling of loudspeaker transducers. It does not model diaphragm breakup, suspension compliance, motor force factor, voice-coil impedance, or other transducer behavior. External Lumped Element tools or more advanced FEA tools are preferred for electroacoustic transducer modelling and to simulate internal enclosure, port, suspension, motor, and diaphragm resonances, while Boundary Lab can supplement these tools to perform the directivity analysis.

### Normalized Channel Transfer Function
Despite each source being driven according to an idealized source, the resultant non-normalized reference-axis frequency response is not necessarily flat. This can introduce complexities when modeling crossover slopes since the acoustic targets are not adhered to. To combat this, Boundary Lab computes a per-channel reference-axis magnitude correction from the unit-velocity response of that channel, then applies the user-defined channel gain, polarity, delay, and crossover filters. This correction flattens the isolated channel magnitude target **before** crossover shaping; it does not apply a complex phase inverse and does not guarantee that the final summed multiway response is flat.

## Boundary Integral Equation Model
Boundary Lab solves the exterior acoustic Helmholtz problem in the frequency domain. For each solved frequency, the acoustic pressure `p` outside the mesh is assumed to satisfy:

$$
\nabla^2 p + k^2 p = 0
$$

where:

- `k = \omega / c`
- `\omega = 2 \pi f`
- `c` is the speed of sound

The mesh surface is treated as the boundary of the exterior acoustic domain. Boundary Lab uses a Neumann boundary condition, meaning the normal pressure gradient is prescribed from the configured surface velocity:

$$
q = \frac{\partial p}{\partial n} = i \rho \omega v_n
$$

where:

- `q` is the boundary Neumann data
- `rho` is air density
- `v_n` is the prescribed outward normal surface velocity

Rigid surfaces use `v_n = 0`. Driven surfaces use the configured ideal velocity drive, including per-radiator velocity offset and any channel gain, polarity, delay, crossover filtering, and flat-target magnitude correction.

### Discretization
The solver uses Bempp-cl boundary element spaces:

- Boundary pressure `p` is represented with continuous linear `P1` basis functions.
- Boundary velocity/flux `q` is represented with discontinuous constant `DP0` basis functions.

At each frequency, Boundary Lab assembles Helmholtz boundary operators on these spaces and solves for the unknown boundary pressure coefficients. The solve is performed with GMRES using the configured solver tolerance.

### Classical Exterior Neumann Form
When Burton-Miller is disabled, Boundary Lab solves the classical exterior Neumann boundary integral equation:

$$
(K - \frac{1}{2} I) p = S q
$$

where:

- `S` is the single-layer boundary operator
- `K` is the double-layer boundary operator
- `I` is the identity operator
- `p` is the unknown boundary pressure
- `q` is the prescribed Neumann boundary data

This corresponds to the exterior representation used later for field evaluation:

$$
p(x) = D[p](x) - S[q](x)
$$

where `D[p]` is the double-layer potential and `S[q]` is the single-layer potential evaluated at observation point `x`.

### Burton-Miller Formulation
The classical exterior Helmholtz integral equation can become unreliable at fictitious interior resonance frequencies. Boundary Lab can use a Burton-Miller combined-field formulation to reduce these irregular-frequency artifacts.

When Burton-Miller is enabled, Boundary Lab assembles the double-layer, single-layer, hypersingular, and adjoint double-layer operators:

- `K`: double-layer boundary operator
- `S`: single-layer boundary operator
- `W`: hypersingular boundary operator
- `K'`: adjoint double-layer boundary operator

The coupling factor is:

$$
\alpha = \frac{i}{k}
$$

Using Bempp-cl sign conventions, the implemented linear system is:

$$
\left(\frac{1}{2} I - K - \alpha(-W_\mathrm{bempp})\right)p
=
\left(-S - \alpha\left(K' + \frac{1}{2} I\right)\right)q
$$

The code explicitly applies the `-W_bempp` term because Bempp-cl's hypersingular operator has the opposite sign from the convention used in the Burton-Miller equation implemented here.

The practical effect is that Boundary Lab combines the pressure equation with its normal derivative equation. This suppresses the fictitious cavity resonances that can appear in the classical exterior Neumann formulation, especially on closed or nearly closed loudspeaker enclosure meshes. Burton-Miller typically increases solve cost because it requires the additional hypersingular and adjoint double-layer operators.

### Field Evaluation and SPL Output
After solving for boundary pressure, Boundary Lab evaluates the acoustic pressure at polar and spherical observation points using:

$$
p(x) = D[p](x) - S[q](x)
$$

The complex pressure magnitude is converted to SPL with:

$$
\mathrm{SPL} = 20\log_{10}\left(\frac{|p(x)|}{20 \times 10^{-6}}\right)
$$

The raw solver output is normalized to the horizontal on-axis response. Plotting and export paths can then re-normalize the horizontal and vertical planes to the configured reference angles for directivity visualization.
