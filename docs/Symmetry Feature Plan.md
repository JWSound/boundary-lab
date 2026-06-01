# Symmetry Feature Plan

Temporary planning document for adding X / XY symmetry acceleration to the Julia solver path.

## Goal

Allow compatible loudspeaker meshes to solve faster by exploiting mirror symmetry, starting with the local Julia CUDA backend only. The user-facing control should live in Mesh Config as a `Symmetry` dropdown near `Stitch Imported Meshes`, with options:

- `Off`
- `X`
- `XY`

The feature must be disabled or unavailable when the selected solver backend does not advertise symmetry support.

Boundary Lab's symmetry convention is that the global 3D origin is the symmetry origin. X symmetry reflects across the global X=0 plane, and XY symmetry reflects across both global X=0 and Y=0. This matches the existing observation-data convention, where directivity calculations are performed relative to the global origin.

## Implementation Progress

- Done: project-level symmetry setting, protocol serialization, solver capability gating, Mesh Config dropdown, and Julia-side parsing/rejection for unimplemented symmetry modes.
- Done: preview-only mirrored geometry generation for X/XY symmetry, with darker image colors and duplicate suppression for triangles on symmetry planes.
- Done: reduced ATH solver mesh cache/routing and first-pass positive-side validation for symmetry solve requests.
- Done: Julia shared symmetry primitives for X/XY image transforms, point/normal reflection, pseudovector curl reflection, reduction factors, and reduced-domain validation.
- Done: first CUDA regular-operator image assembly hook. Reflected trial/source image passes can now accumulate into the reduced matrices while singular corrections remain attached to identity-domain pairs.
- Done: added raw ATH `sample_half.msh` and `sample_quarter.msh` fixture smoke coverage for positive-domain validation and CUDA image assembly pair-count behavior.
- Done: added cleaned/staged reduced-vs-expanded comparison scaffolding and initial operator-action checks.
- Done: added P1 seam/orbit row weights for reduced operators and Burton-Miller identity/mass terms.
- Done: added image-singular correction deltas for reflected trial pairs that become coincident/edge/vertex adjacent through a symmetry plane.
- Done: moved image-singular correction deltas onto the compact CUDA compute/scatter path, added symmetry-expanded field evaluation, added impedance force scaling, and removed the Julia non-`off` solve rejection.
- Next: run end-to-end solve checks through the application workflow and decide whether any solver output diagnostics should expose symmetry reduction details.

## Non-Goals For First Implementation

- No symmetry acceleration for the local Bempp-cl backend.
- No assumption that server backends support symmetry unless the server capability contract explicitly reports it.
- No support for antisymmetric radiator drives in the first pass.
- No support for arbitrary mirror planes. Only global X=0 and Y=0 planes.
- No half/quarter mesh solve by simply deleting mirrored panels. The application will provide a reduced mesh plus symmetry metadata, and the solver must account for image contributions or an equivalent reduced operator.

## Current State

The application already understands ATH symmetry at mesh-clean time:

- `src/blab/ath.py` reads `Sym=...` from `solving.txt`.
- `src/blab/mesh_clean.py` mirrors ATH meshes across requested axes before writing the cleaned solver mesh.

The Julia backend currently receives an already-expanded full mesh:

- `src/blab/solvers/julia_local/solver.jl` loads configured meshes and combines them into one `BoundaryMesh`.
- `src/blab/solvers/julia_local/src/JBEMCore.jl` builds P1 pressure and DP0 flux spaces.
- Dense Galerkin operators `S`, `D`, `D*`, and `H` are assembled each frequency.
- `solve_burton_miller_neumann` forms and directly solves the dense Burton-Miller system.
- Field evaluation sums over all source quadrature points on the full mesh.

Because Burton-Miller is a linear combination of operators that are themselves invariant under mirror transforms, symmetry is compatible in principle. The implementation work is in constructing the reduced degrees of freedom and accumulating mirrored-image operator contributions correctly.

Target workflow change:

- The application passes the Julia solver the reduced fundamental-domain mesh.
- The application also passes symmetry metadata (`off`, `x`, or `xy`).
- The Julia solver does not need to infer the fundamental domain from a full expanded mesh and does not need full-mesh orbit matching.
- The GUI 3D viewport should still display the full mirrored geometry for user visibility.

## Reference Implementation Notes

`reference/JAX-BEM` is useful as conceptual guidance, not code to copy.

Key takeaways:

- Build the reduced operator by looping over all active reflected trial/source images in addition to the identity contribution.
- X symmetry in Boundary Lab means reflecting the `x` coordinate across the global YZ plane. XY symmetry means the image set: reflect X, reflect Y, reflect X+Y.
- Single-layer image terms reflect trial/source points only.
- Double-layer image terms reflect trial/source points and trial/source normals.
- Adjoint double-layer image terms reflect trial/source points; the test normal remains in the reduced-domain frame.
- Hypersingular image terms need extra care: normals transform like vectors, but surface curls transform like pseudovectors. For a reflection matrix `R`, use `curl_image = det(R) * R * curl`.
- Singular treatment should remain attached to the true reduced-domain coincident/edge/vertex pairs. Reflected image pairs can use regular quadrature unless a focused validation case proves a seam-specific singular rule is needed.
- Field evaluation can use the same straightforward image-source summation idea. It is not a runtime bottleneck for current workflows, so clarity is more important than field-only optimization.
- A full-mesh-to-reduced cutter can use centroid side tests and preserve physical tags, but Boundary Lab's primary workflow should be application-owned reduced mesh preparation plus validation, not solver-side orbit inference.

## Track 1: Solver Backend

### 1.1 Add Symmetry Mode Parsing

Add a Julia-side config parser for:

```json
{
  "symmetry": "off" | "x" | "xy"
}
```

Initial behavior:

- Missing field means `off`.
- Unsupported value throws a clear solver error.
- Only apply reduced solve paths when `symmetry != "off"`.

### 1.2 Define Reduced Symmetry Geometry

Introduce a solver-side representation that can describe:

- Active fundamental-domain elements.
- P1 vertex dofs on the reduced mesh.
- DP0 element dofs on the reduced mesh.
- Reflection transforms in the selected group:
  - X mode: identity, reflect X.
  - XY mode: identity, reflect X, reflect Y, reflect X+Y.
- Per-image orientation / normal transform.
- Seam entities lying on symmetry planes.

Resolved direction: the application passes a reduced mesh plus symmetry metadata. The solver should not implement general orbit matching against an expanded full mesh. This keeps the numerical backend focused on image assembly/evaluation and lets the application own project workflow, preview behavior, and user-facing mesh preparation.

### 1.3 Compatibility Validator

Before assembling a reduced system, validate:

- Mesh vertices/elements are in the expected reduced side/quadrant for the selected global-origin symmetry mode.
- Surface physical tags are internally consistent on the reduced mesh.
- Radiator definitions produce mirror-symmetric Neumann data by construction.
- Mesh translations do not break the selected global-origin symmetry.
- Imported meshes selected for symmetry are already prepared as reduced-domain meshes.
- Normals and triangle orientations are consistent after reflection.

Failure mode should be explicit:

- Either reject the solve with a clear message.
- Or fall back to full solve only if the UI asked for `Auto` later. Current requested UI has explicit `Off/X/XY`, so rejection is safer.

### 1.4 Reduced Operator Assembly

Implement reduced operators by summing image contributions into the fundamental-domain test/trial dofs.

For each test element in the reduced mesh and each trial element in the reduced mesh:

- Accumulate contributions from every reflected image of the trial element implied by the selected symmetry group.
- Apply reflected source geometry and normals.
- Preserve the same Galerkin block formulas for `S`, `D`, `D*`, and `H`.
- Apply the same singular correction logic for true coincident/adjacent pairs in the fundamental domain.
- Treat reflected images as regular pairs unless they become coincident/adjacent through a symmetry plane.

Important Burton-Miller detail:

- Hypersingular assembly uses surface curls and `normal_product`.
- Reflection changes handedness, so triangle orientation and normal/curl transforms must be verified with focused tests.
- Surface curls are pseudovectors: `curl_image = det(R) * R * curl`, while normals use `normal_image = R * normal`.

### 1.5 Reduced Burton-Miller Solve

The reduced solve should form the same equation:

```text
(0.5 I - D + (i/k) H) p = (-S - (i/k)(D* + 0.5 I)) q
```

but on reduced P1/DP0 spaces.

Settled during prototyping:

- Seam dofs on X=0/Y=0 need P1 orbit row weights.
- P1 mass/identity blocks should use the same row-weighting treatment as reduced operator rows.
- Field evaluation can materialize mirrored quadrature sources while reusing the existing CPU/GPU evaluation kernels.
- Impedance should scale driven element force by the symmetry reduction factor for the reduced-domain solve.

### 1.6 Field Evaluation

Field evaluation must include contributions from all mirror images.

Field evaluation is not currently a meaningful runtime bottleneck, even with thousands of observation points. Prefer the simplest robust implementation:

- Reconstruct or materialize mirrored source contributions for evaluation.
- Reuse as much of the existing field evaluation path as practical.
- Avoid a complex field-only optimization unless profiling later shows it matters.

### 1.7 Impedance

Radiator impedance currently integrates pressure over driven elements. With symmetry:

- Integrate over all physical images of the driven radiator.
- Ensure force scaling matches the full expanded solve.
- Confirm reported impedance remains comparable to existing full-mesh results.

### 1.8 CUDA Solver Path

Implementation order:

1. Implement correctness-focused shared geometry/symmetry helpers in `JBEMCore.jl` where useful.
2. Implement the active solver path in CUDA-oriented assembly/solve code.
3. Validate against full expanded solves on small meshes.
4. Extend CUDA regular assembly, singular correction handling, and field evaluation/reconstruction as needed.

CPU dense solving in `JBEMCore.jl` is deprecated and not used by current solver workflows. Do not make CPU solving the primary implementation target or required correctness oracle.

### 1.9 Solver Tests And Benchmarks

Add small Julia smoke tests:

- X-symmetric mesh, one driven symmetric tag: reduced vs full pressure and field.
- XY-symmetric mesh: reduced vs full pressure and field.
- Asymmetric source drive rejected.
- Reduced mesh on the wrong side/quadrant rejected.
- Seam-only triangles do not double-count.

Add benchmark runs:

- Full vs X vs XY for representative ATH meshes.
- Track assembly, solve, field, peak memory, and result deltas.

Current fixture notes:

- `sample_half.msh` and `sample_quarter.msh` are raw ATH meshes, not mesh-cleaned/stiched solver meshes.
- They are useful now for side/quadrant validation and CUDA image assembly smoke tests.
- Full-vs-reduced physical correctness comparisons should use a cleaned/staged set so surface connectivity and discretization differences do not dominate the numerical delta.
- `scripts/stage_symmetry_comparison_meshes.py` builds reduced-clean and mirrored-full staged meshes from the raw ATH half/quarter fixtures.
- `scripts/compare_symmetry_operator_actions.jl` compares reduced symmetry operator actions with expanded full operator actions by applying symmetric vectors and restricting the full result back to the fundamental domain.

Initial comparison results:

- Half/X, 96 reduced faces: non-seam relative errors were roughly `2.7e-4` single-layer, `1.3e-3` adjoint double-layer, `1.2e-3` double-layer, and `9.8e-4` hypersingular.
- Quarter/XY, 96 reduced faces: non-seam relative errors were roughly `4.7e-4` single-layer, `1.3e-2` adjoint double-layer, `3.5e-3` double-layer, and `1.2e-3` hypersingular.
- All-row errors are much larger because symmetry-plane P1 seam vertices are currently not orbit-weighted. This confirms seam/orbit weighting needs to be implemented before enabling production solves.

After P1 seam/orbit row weights:

- Half/X all-row errors improved to roughly `1.3e-3` single-layer, `7.0e-3` adjoint double-layer, `9.8e-3` double-layer, and `2.3e-3` hypersingular.
- Quarter/XY all-row errors improved to roughly `1.3e-3` single-layer, `3.5e-2` adjoint double-layer, `4.0e-2` double-layer, and `2.5e-3` hypersingular.
- Remaining seam-heavy error is concentrated in double/adjoint double layer and is consistent with reflected image pairs that are geometrically adjacent across symmetry planes but are still being integrated with regular quadrature.

After image-singular correction deltas:

- Half/X, 96 reduced faces: all-row relative errors were roughly `7.0e-5` single-layer, `1.9e-4` adjoint double-layer, `1.2e-4` double-layer, and `2.9e-5` hypersingular. The comparison detected 51 image singular pairs.
- Quarter/XY, 96 reduced faces: all-row relative errors were roughly `1.2e-4` single-layer, `1.6e-3` adjoint double-layer, `1.6e-3` double-layer, and `6.8e-5` hypersingular. The comparison detected 108 image singular pairs.
- The CUDA path now computes image-singular deltas into compact GPU block buffers and scatters them into dense correction buffers like the identity-domain singular cache. The comparison errors above were unchanged after moving this correction path to GPU.

## Track 2: Solver-Application Handshake

### 2.1 Extend Solver Capabilities

Add a capability field:

```python
supports_symmetry: bool = False
```

in `src/blab/solvers/base.py`.

Set:

- `julia_local`: `True`
- `local` Bempp-cl: `False`
- `server`: initially `False`, unless a server capability endpoint is added later.

Tests:

- Update `tests/test_solver_backends.py` to assert Julia supports symmetry and Bempp does not.

### 2.2 Extend SimulationConfig

Add a solver input field:

```python
symmetry: str = "off"
```

Validation/conventions:

- Allowed values are `off`, `x`, `xy`.
- Store lowercase in protocol/config.
- UI labels can be `Off`, `X`, `XY`.

### 2.3 Extend Protocol Serialization

Update `src/blab/protocol.py` so `symmetry` round-trips through:

- `simulation_config_to_dict`
- `simulation_config_from_dict`
- `solve_request_from_config_and_frequencies`
- server job input parsing

Tests:

- Add protocol round-trip coverage in `tests/test_protocol.py`.
- Add server serialization coverage if server request tests assert payload shape.

### 2.4 Worker Behavior

`SolveWorker` should pass the selected symmetry value inside `SimulationConfig`.

Backend behavior:

- Julia local accepts non-`off`.
- Bempp local should never receive non-`off` from the GUI; if called programmatically, either ignore only `off` or reject non-`off`.
- Server should reject non-`off` unless server capabilities later say otherwise.

### 2.5 Diagnostics

Solver result diagnostics should expose when symmetry was used:

```json
"diagnostics": {
  "message": "Julia direct dense solve (XY symmetry)",
  "symmetry": "xy",
  "symmetry_reduction_factor": 4
}
```

Optional but useful:

- fundamental-domain face count
- reduced P1 dofs
- reduced DP0 dofs
- rejected fallback reason if a future `Auto` mode is added

## Track 3: Application-Side

### 3.1 Mesh Config UI

Add a `Symmetry` dropdown beside `Stitch Imported Meshes` in `MeshConfigDialog`.

Suggested layout:

- `Import .msh`
- `Remove`
- spacer
- `Stitch Imported Meshes`
- `Symmetry` label
- dropdown: `Off`, `X`, `XY`

Disable dropdown when selected backend capability does not support symmetry.

Implementation notes:

- `MeshConfigDialog` currently only receives `meshes` and `stitch_imported_meshes`.
- Pass a `symmetry` value and a `symmetry_enabled` flag from `MainWindow.open_mesh_config`.
- Use `backend_info(self.preferences.solve_backend).capabilities.supports_symmetry` to gate it.

### 3.2 Persist Project State

Add project payload key:

```json
"symmetry": "off"
```

Update:

- `src/blab/ui/project_io.py`
- `MainWindow._project_payload`
- `MainWindow._apply_project_payload`
- `new_project`
- project tests

This is project state, not a global preference, because symmetry is geometry-specific.

### 3.3 Mesh Preparation Path

Current ATH cleanup expands ATH symmetry into a full mesh. For the symmetry solver feature, selected solver symmetry means:

1. The solver receives a reduced fundamental-domain mesh plus symmetry mode.
2. The preview/source-config workflow should remain understandable to users by displaying mirrored geometry in the viewport.
3. Julia does not infer orbits from an expanded full mesh.

Preferred long-term path:

- Keep preview/source UX stable.
- Preserve enough metadata to give Julia the reduced fundamental-domain mesh when symmetry is enabled.
- Avoid double-expanding ATH symmetry when the Julia solver will account for mirror images.

Need design work:

- Imported meshes do not have `solving.txt`; user-selected symmetry must drive validation.
- ATH-generated meshes may already carry `Sym=xy`; the UI should default to that only if it is unambiguous.
- Stitched meshes and symmetry need a clear order of operations. Likely: clean/reduce individual meshes, apply transforms, validate symmetry, then solve. If stitching creates a cached expanded mesh, symmetry cache keys must include the symmetry mode.

External/imported mesh support:

- Apply the same symmetry logic to imported meshes as to generated ATH meshes.
- When symmetry is enabled, imported meshes should be prepared by the user as reduced-domain meshes relative to the global origin.
- Future user documentation should explain how to place/trim imported meshes for X and XY symmetry before import.

Implementation direction:

- Keep the existing expanded cleaned ATH mesh for non-symmetry solves and for legacy behavior.
- Add a reduced cleaned ATH mesh cache created with `mirror_axes=()` for symmetry solves.
- Route Julia symmetry solves to the reduced cache; route non-symmetry solves to the existing expanded cleaned mesh.
- Include `symmetry` in stitched mesh cache keys so changing Off/X/XY cannot reuse the wrong stitched solver mesh.
- Imported meshes are already treated as reduced when symmetry is enabled; the application should validate/warn rather than auto-cut imported geometry.

### 3.4 Viewport Mirrored Display

When symmetry is enabled, the 3D viewport should display the mirrored full geometry for user-facing visibility, even though the solver receives a reduced mesh.

Rendering convention:

- Original/reduced mesh elements use the existing normal mesh colors.
- Mirrored image elements render as a slightly darker shade of gray than original rigid mesh elements.
- Driven/source coloration should remain clear; if mirrored driven elements are shown, use a darker/secondary driven shade that still reads as driven but distinguishes image geometry from original geometry.

Implementation notes:

- Mirrored display geometry should be generated in the application preview layer, not written back into the solver mesh unless a separate full-preview cache is needed.
- Reflections use the global origin and the selected symmetry group.
- The viewport should avoid changing source config identity unexpectedly; mirrored elements are visual images of the same logical surfaces.

### 3.5 Source Config UX

Source Config should still show surface tags in a way users understand.

First implementation can require:

- Driven surfaces on mirrored copies share the same physical tag and channel/drive.
- If the source config would create asymmetric drives, block the solve with a clear message.

Later improvement:

- Show grouped mirrored surfaces as one logical source row when symmetry is enabled.

### 3.6 Backend Gating

When Bempp OpenCL CPU is selected:

- Disable `Symmetry`.
- Set or display `Off`.
- If a project with `symmetry != off` is loaded while Bempp is selected, preserve the project value but do not allow solve until either Julia is selected or symmetry is set to Off.

When Server is selected:

- Treat as unsupported until server capability reporting exists.
- Future server support should come from a server capabilities endpoint, not a hardcoded assumption.

### 3.7 User Feedback

Before solve:

- If symmetry is selected and backend does not support it, show a blocking warning.
- If symmetry is selected and validation fails, show the Julia/application validation error with enough detail to identify the offending mesh/tag/radiator.

During solve:

- Status line should include `symmetry X` or `symmetry XY`.

## Suggested Milestones

### Milestone A: Plumbing And UI Skeleton

- Add `supports_symmetry` capability.
- Add `SimulationConfig.symmetry`.
- Add protocol/project round trips.
- Add Mesh Config dropdown gated by backend.
- Pass `symmetry` through to Julia.
- Julia accepts the field but rejects non-`off` with `not implemented`.

This gives the application shape without numerical risk.

### Milestone B: Symmetry Validation Prototype

- Add Julia or Python validation for X/XY compatibility.
- Implement the reduced-mesh plus symmetry-metadata contract.
- Add tests for accepted/rejected symmetric configurations.

### Milestone C: CUDA Reduced Solve Prototype

- Implement reduced operator assembly and direct solve in the active Julia CUDA path.
- Compare against full expanded solves on tiny meshes.
- Reconstruct or symmetry-evaluate fields.

### Milestone D: Viewport And Mesh Workflow

- Render mirrored preview geometry in the 3D viewport.
- Route reduced solver meshes to Julia while keeping user-facing preview/source workflows clear.
- Support imported reduced-domain meshes with symmetry enabled.
- Benchmark memory and time improvements.

### Milestone E: Product Polish

- Better validation messages.
- Documentation in User Guide and Julia backend notes.
- Optional automatic default from ATH `Sym=...`.
- Optional future `Auto` mode.

## Open Questions

- Should the user-selected `Symmetry` setting be project-level, per-mesh, or per-solve? Current plan: project-level solve setting.
- Should ATH `Sym=xy` automatically set the dropdown to `XY`, or only suggest it?
- Can seam dofs be handled entirely by orbit weights, or do we need explicit boundary-plane classification?
- Should the server backend grow a capabilities endpoint before this feature ships?
- How should the UI warn when an imported mesh appears to be full geometry rather than a reduced-domain mesh?

## Early Risk List

- Incorrect normal/curl transform signs in `D`, `D*`, or `H`.
- Double-counting source strength on symmetry-plane triangles.
- Source Config creating asymmetric drives accidentally.
- Stitch-cache and symmetry-cache invalidation bugs.
- Preview mesh and solver mesh diverging in ways that confuse users.
- Imported meshes prepared around the wrong origin or already expanded before import.
