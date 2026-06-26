# Forward Beam Shape Plot

The Forward Beam Shape plot is a diagnostic view in the balloon plot window. It reduces each front-facing spherical radiation balloon to two frequency-dependent traces:

- the fitted superellipse exponent `p` of the forward `-6 dB` beam contour
- the spherical directivity index in dB

The goal is to describe how the forward beam shape changes with frequency. It is not a replacement for the full balloon, isobar, or polar views. It is a compact shape descriptor that makes broad trends easier to see, especially for horns, waveguides, and other non-axisymmetric radiators.

## What The Plot Shows

The left axis shows the fitted superellipse exponent `p` for the front-facing `-6 dB` contour:

- `p = 1`: diamond or rhombus-like contour
- `p = 2`: ellipse or circle-like contour
- `p = 4`: rounded square-like contour
- `p = 8`: squarer contour with harder corners

The plotted points are colored by fit residual. Green means the extracted contour is well-described by the nearest superellipse. Red means the contour is less superellipse-like, usually because it has lobing, ripples, asymmetry, notches, or local diffraction features.

The right axis shows `Spherical DI (dB)`, computed from the full spherical SPL data at each frequency. This lets the beam shape trend be compared against the overall concentration of radiated energy.

A vertical cursor line follows the current frequency selected in the balloon viewer.

## Source Data

The plot uses the prepared spherical balloon arrays:

- `freq_hz`: solved frequencies
- `theta_grid_rad`: polar angle from the forward `+z` axis
- `phi_grid_rad`: azimuth angle around `+z`
- `balloon_surface_spl`: normalized SPL in dB on the spherical grid

The balloon coordinate convention is:

```text
x = horizontal right
y = vertical-side axis
z = forward reference axis
```

For each spherical sample, the corresponding unit direction is:

$$
x = \sin\theta\cos\phi,
$$

$$
y = \sin\theta\sin\phi,
$$

$$
z = \cos\theta.
$$

Only the forward hemisphere is used for the beam shape fit. Boundary Lab also limits the front angular map to approximately `+/-89 deg` horizontally and vertically so the tangent-plane transform remains finite.

## Front Tangent-Plane Coordinates

The shape fit is performed in a tangent front plane rather than directly in independent horizontal and vertical angle coordinates. This is important for wide beams.

The horizontal and vertical angular coordinates are first computed as:

$$
\alpha = \operatorname{atan2}(x,z),
$$

$$
\beta = \operatorname{atan2}(y,z).
$$

They are then mapped to tangent-plane coordinates:

$$
u = \tan(\alpha),
$$

$$
v = \tan(\beta).
$$

This is equivalent to projecting the forward spherical data onto the plane tangent to the front axis. A circular cone on the sphere remains circular in this coordinate system. That avoids an artifact where very wide axisymmetric beams can look artificially square when fitted in raw horizontal/vertical angle space.

Boundary Lab builds a 2D interpolator from the forward samples:

$$
S_f(u,v),
$$

where `S_f` is normalized SPL in dB for one frequency `f`.

## Extracting The -6 dB Contour

For each frequency, Boundary Lab samples radial rays in the tangent plane:

$$
(u(r),v(r)) = (r\cos\psi, r\sin\psi),
$$

where `psi` is the ray angle around the forward axis.

Along each ray, the app finds the first crossing where the interpolated SPL falls through the target level:

$$
S_f(u(r),v(r)) = -6\ \mathrm{dB}.
$$

The crossing radius is linearly interpolated between adjacent ray samples. This produces a set of contour samples:

$$
(\psi_i, r_i).
$$

Frequencies are considered invalid for the shape trace when there is not enough usable `-6 dB` contour data. This can happen at low frequencies when the forward hemisphere has not yet fallen to `-6 dB`.

## Aspect-Corrected Superellipse Fit

The fitted shape family is a superellipse in polar form. For horizontal radius `a`, vertical radius `b`, and exponent `p`, the model radius is:

$$
r_{model}(\psi) =
\left(
\left|\frac{\cos\psi}{a}\right|^p +
\left|\frac{\sin\psi}{b}\right|^p
\right)^{-1/p}.
$$

Boundary Lab estimates the horizontal and vertical extents from the `-6 dB` crossings along the horizontal and vertical tangent-plane axes. These extents become the aspect correction terms `a` and `b`.

The exponent `p` is then found by minimizing mean squared radial error:

$$
\operatorname*{arg\ min}_{p}
\frac{1}{N}\sum_i
\left(r_i - r_{model}(\psi_i)\right)^2.
$$

The current fit bounds are approximately:

```text
0.75 <= p <= 8.0
```

The displayed beamwidth values remain angular quantities. Tangent-plane extents are converted back to degrees with:

$$
\theta_{deg} = \frac{180}{\pi}\tan^{-1}(r).
$$

For example, the reported horizontal beamwidth is the sum of the positive and negative horizontal crossing angles.

## Fit Residual

The fit residual is a normalized RMS radial error in the tangent plane:

$$
\mathrm{residual} =
100
\frac{
\sqrt{\frac{1}{N}\sum_i \left(r_i - r_{model}(\psi_i)\right)^2}
}{
\frac{a+b}{2}
}.
$$

The residual is shown as the color of the `p` trace. The color scale is fixed from `0` to `15` percent so plots can be compared across projects.

Low residual means the contour is close to the fitted superellipse. High residual means the contour is not well-described by this four-way shape family, even if the fitted `p` value is still plotted.

## Spherical Directivity Index

The secondary axis shows spherical directivity index computed from the spherical normalized SPL data.

If raw equal-area spherical samples are available, Boundary Lab uses them directly. Since SPL has already been normalized to the reference axis, it converts normalized dB values to relative linear energy:

$$
E_i = 10^{S_i/10}.
$$

The mean relative energy over the sphere is:

$$
\bar{E} = \frac{1}{N}\sum_i E_i.
$$

The spherical directivity index is then:

$$
DI = -10\log_{10}(\bar{E}).
$$

If only the prepared theta/phi grid is available, Boundary Lab uses a spherical area weighting proportional to:

$$
\sin\theta.
$$

This compensates for the fact that a regular theta/phi grid has many more samples near the poles than near the equator.

## Interpreting Results

A flat trace near `p = 2` usually indicates an axisymmetric or ellipse-like forward beam. A trace moving upward toward `p = 4` or `p = 8` indicates a squarer contour. A trace moving downward toward `p = 1` indicates a more diamond-like contour.

The exponent should be read together with residual color:

- low residual and stable `p`: the contour is well-described by the shape family
- high residual and unstable `p`: the contour likely has lobing or local features that a superellipse cannot summarize well
- invalid low-frequency points: the forward `-6 dB` contour may not exist inside the front map

The Spherical DI trace helps distinguish a shape change from a simple narrowing or widening of the beam. For example, `p` may remain close to `2` while DI rises, meaning the beam is becoming narrower but staying circular.

## Limitations

The Forward Beam Shape plot intentionally compresses a 3D radiation pattern into a few scalar values per frequency. It does not show:

- rear radiation
- side lobes outside the forward contour
- multiple disconnected `-6 dB` regions
- asymmetry not captured by a single horizontal and vertical aspect ratio
- local ripple around the contour, except indirectly through residual

Use the plot as a trend view. When the residual rises, inspect the full balloon, contour lines, polar slices, and isobar views to understand the actual radiation pattern.
