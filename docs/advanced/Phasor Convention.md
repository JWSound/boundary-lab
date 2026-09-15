# Phasor convention

Boundary Lab requests native `exp(+i omega t)` from BEAT Engine. Complex result
storage, DSP, phase plots, impedance, displacement, field animation, exports,
and Deploy use that same convention. Physical frequency and wavenumber are positive.

| Operation | Positive-time form |
| --- | --- |
| Time derivative | `+i omega` |
| Outgoing Green function | `exp(-i k r)/(4 pi r)` |
| Normal derivative | `q = -i rho omega v_n` |
| Burton-Miller coupling | `-i/k` |
| Passive FEM bulk loss | `K - k^2 M + i eta k^2 M` |
| Surface admittance contribution | `+i rho omega Y B` |
| Plane-wave termination | `q = -i k p` |
| Electrical impedance | `R + i omega L` |
| Mechanical impedance | `R + i(omega M - 1/(omega C))` |
| Displacement | `x = v/(+i omega)` |
| Positive delay | `exp(-i omega tau)` |
| Propagation alignment | `exp(+i k r)` |
| Animation | `Re(P exp(+i theta))` |

Normals, real jump terms, interface orientation and linear-algebra adjoints
retain their definitions. This change does not improve resonance conditioning:
changing conventions conjugates a physical solution and preserves its magnitude.

## Worker compatibility

Compiled-system requests select `solver_options.phasor_convention`; source,
Deploy and retained-field requests select top-level `phasor_convention`.
Workers advertise supported conventions. Results carry the selected convention
in diagnostics; binary field events carry it alongside the array descriptor.
Clients check both capability and response. Update the app, engine and remote
server together. Old workers and unknown conventions are rejected for new solves.

Unlabelled engine requests and low-level Julia calls keep the historical
`exp(-i omega t)` default for compatibility. Julia callers can wrap the complete
assembly, solve and field evaluation in
`with_phasor_convention(POSITIVE_TIME_PHASOR) do ... end`. Each worker processes
requests serially, and restores the previous convention even after an error.
GPU kernels receive a signed internal propagation parameter at host entry;
there is no post-assembly conjugation of dense operators.

CPU and CUDA positive-time paths are qualified. ROCm signs are wired, but
positive-time worker requests are blocked until an AMD hardware qualification
run passes. Geometry-only caches can be reused across conventions. Deploy's
retained solution cache rejects field requests with a different convention.

## Legacy data

Version-1/2 unlabelled system results and schema-1 unlabelled speaker packages
are interpreted as legacy negative-time data. Complex pressure, flux, motion,
current, impedance and all complex ROM coefficient arrays are conjugated once
on ingestion. Canonical data is then labelled positive time. Package archive
bytes and checksums are preserved during loading.

The older source-result protocol is a special case: its impedance pairs were
already serialized as standard audio impedance, so only its pressure arrays
are conjugated. New result exports explicitly label their convention.

## Validation

The frozen engine reference fixtures are unchanged. The regression gates cover
legacy references; native CPU/CUDA conjugation; direct/operator and RHS-only
Burton-Miller parity; singular, near and ground-image corrections; field replay;
passive loss and electromechanical models; and complex voltage excitations on
both sides of a weakly damped cavity pole with monolithic and condensed solves.

Application checks cover source/system transports, remote capability rejection,
legacy result/package/ROM ingestion, delay and group delay, phase exports,
particle velocity, displacement, impedance and animation. Deploy's TypeScript
package reader and complex ground-interference tests cover the frontend boundary.
