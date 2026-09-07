# Standalone BEAT numerical references

Run the required CPU extraction gate from any working directory:

```text
julia --threads=2 --startup-file=no --project=<engine>/julia_local <engine>/julia_local/tests/reference_tests.jl
```

It requires the Julia environment to be instantiated. The gate and its fixtures
live entirely under `julia_local`; neither Boundary Lab Python code, Qt, Bempp,
nor application project files are needed. Preserve the sibling `beat_contract`
directory when copying the complete engine/worker distribution. CI runs this gate
from a copied engine tree outside the checkout, after the ordinary Julia suite.

## Coverage and interpretation

| Behavior | Reference | Acceptance |
|---|---|---|
| Exterior BEM and arbitrary complex-pressure probes | Manufactured outgoing Helmholtz point-source solution enclosed by a generated 128-face surface | Relative complex field error below 8% at three exterior points |
| Independent excitations | Centered and displaced enclosed point sources; complex combination of separately solved traces versus a combined solve | Each source matches its analytical field; complex superposition at relative tolerance 1e-12 |
| Retained fields | Re-evaluate solved pressure/Neumann traces at exterior points; complex scaling and combination | Relative tolerance 1e-12 |
| X and XY symmetry | Reduced field evaluation versus separately constructed full reflected geometry, including normal orientation | Relative tolerance 1e-12, absolute 1e-14 |
| Interior FEM | Generated sealed unit-cube cavity modes, exact affine tetrahedron matrices, and preserved mesh matrix references | Existing mode discretization tolerance 8%; existing matrix/precision tolerances unchanged |
| Coupled FEM-BEM | Full direct solves, pressure continuity, flux conservation, and BEM replay | Existing residual/continuity tolerances, including 1e-8 and 1e-10 checks |
| Coupled FEM-BEM-LEM | Voltage-driven transducer solved using monolithic and condensed formulations | Existing FP64 relative tolerance 1e-9 on acoustic/electromechanical quantities |
| Precision | FP32 versus FP64 coupled and condensed results | Existing relative tolerance 1e-4 on primary fields |

The analytical exterior oracle is `p(x) = exp(i k r) / r`, for the solver's
`exp(-i omega t)` convention. Its facet-normal derivative is
`(i k - 1/r) p(x) dot(n, (x-source)/r)`. The source is enclosed, so the entire
exterior domain satisfies homogeneous Helmholtz. The oracle does not call BEAT's
Green-function implementation. Facet-centroid DP0 data and the coarse P1 mesh
introduce discretization error; the 8% bound is an accuracy guard, not a promise
of production convergence. The assertions compare complex pressure, not SPL.

Condensed-versus-monolithic tests are independent elimination/formulation checks
but share assembly kernels. They complement the analytical checks; they do not
constitute an independently implemented BEM solver. Frozen fixture hashes protect
the input baseline; they are not numerical expected outputs.

The required runner forces `BLAB_RUN_COUPLED_REFERENCE=1` and first-order coupled
quadrature settings, preventing ambient opt-out settings from skipping the dense
comparisons. Those coupled tests compare algebra/formulations at identical
discretization. The analytical exterior tests independently use order 3. Ordinary
`runtests.jl` retains its existing faster default and hardware-dependent checks.

## Accelerator and extended qualification

The existing `BLAB_RUN_COUPLED_CUDA=1` and `BLAB_RUN_COUPLED_ROCM=1` gates in
`runtests.jl` still require functioning hardware and their corresponding Julia
projects. CPU reference success does not qualify CUDA, ROCm, or future Metal.
Hardware release runs must inspect actual test execution; an unavailable-device
skip is not a numerical pass.

The larger noncubic-cavity convergence family remains in Boundary Lab's
`tests/fixtures/noncubic_cavity` with its existing optional runner. It is extended
application validation, not a dependency of this portable gate. Preserve that
family if it is included in a future engine release's extended qualification.

The remaining source-request reference harness is retained. Remove it only when
the physical-system replacement covers the comparisons it provides; this gate
creates a portable baseline and does not silently retire additional comparisons.
