# Afterburner BEM: Regular Near/Far Pair Classification Plan

**Purpose:** define a near-term implementation path for reducing dense regular-pair assembly cost in the CUDA acoustic BEM backend by classifying regular element pairs into near/mid/far tiers and using reduced quadrature for sufficiently separated regular pairs.

**Scope:** FP32-first. This plan deliberately postpones BF16/FP16/NVFP4 work until the reduced-quadrature and pair-tiering path is numerically characterized.

**Baseline context:** current backend uses direct dense Galerkin BEM with CUDA regular-pair assembly, GPU Duffy corrections for adjacent/coincident pairs, and a dense Burton-Miller solve. The current benchmark shows the regular assembly kernel is the dominant runtime component.

---

## 1. Executive summary

The near-term optimization should be:

```text
Current path:
    all regular pairs -> full quadrature FP32 regular kernels
    singular/adjacent pairs -> FP32 Duffy correction path

Proposed near-term path:
    singular/adjacent pairs -> unchanged FP32 Duffy correction path
    regular-near pairs      -> full quadrature FP32 regular kernels
    regular-mid pairs       -> reduced quadrature FP32 regular kernels
    regular-far pairs       -> more aggressive reduced quadrature FP32 regular kernels
```

The important design decision is that **topology and geometry classification should happen outside the frequency loop**, while the final quadrature tier should remain **frequency-aware**. The mesh geometry does not change across a 100-frequency sweep, but the Helmholtz kernel does:

\[
G_k(x,y)=\frac{e^{ikr}}{4\pi r}
\]

As frequency rises, phase variation across each element pair increases, so a pair that is safe for reduced quadrature at low frequency may need to be promoted to a higher quadrature tier at high frequency.

Do **not** start with lower precision. Start with:

```text
FP32 evaluation
FP32 accumulation
same dense matrix layout
same dense solve
same Duffy correction path
new pair-tiered regular worklists
new reduced-quadrature regular kernels
```

This keeps the first implementation close to the current architecture and gives clear A/B validation against the existing `split_atomic_balanced` mode.

---

## 2. Existing architecture seam to exploit

The current CUDA backend already has a strong separation between:

1. regular element-pair assembly; and
2. adjacent/coincident singular or near-singular corrections.

That seam is exactly where the near/far regular-pair tiering should be inserted.

Current regular assembly behavior:

```text
regular element pairs
    -> CUDA regular-pair kernels
    -> one CUDA block per test/trial element pair
    -> quadrature-pair work distributed across threads
    -> shared-memory reduction
    -> FP32 atomic scatter into dense real/imag operator buffers
```

Current singular behavior:

```text
coincident / edge-sharing / vertex-sharing / adjacent pairs
    -> separate Duffy correction cache
    -> Duffy block kernel
    -> compact correction blocks
    -> atomic scatter into dense correction buffers
    -> add corrections after regular assembly
```

The proposed tiering should **not** disturb the Duffy path. It should only alter how topologically regular pairs are scheduled into regular assembly kernels.

---

## 3. Why reduced quadrature before mixed precision

The benchmarked regular path uses quadrature order 4 with 36 quadrature-pair evaluations per regular element pair. With approximately 49 million regular kernel blocks / element-pair blocks, this is a very large number of scalar Helmholtz kernel evaluations.

The core cost being attacked is therefore:

\[
O(N_e^2 q^2)
\]

where:

- \(N_e\) is the face/element count;
- \(q\) is the number of quadrature points per triangle;
- \(q^2\) is the number of test/trial quadrature-pair evaluations per element pair.

For sufficiently separated element pairs, the kernel is smoother over the product of the two triangles. That makes lower quadrature order a more direct optimization than lower arithmetic precision.

Mixed precision does not remove:

- distance evaluation;
- reciprocal distance;
- phase evaluation;
- normal derivative evaluation;
- hypersingular weak-form terms;
- atomics into dense matrices.

Reduced quadrature **does** reduce the number of times those operations are performed.

---

## 4. Definitions

### 4.1 Pair topology

Keep the current topology classification:

```text
singular / near-singular:
    coincident
    edge-sharing
    vertex-sharing
    adjacent
    image-singular under symmetry

regular:
    all other test/trial triangle pairs
```

Only `regular` pairs enter the new tiering system.

### 4.2 Per-face geometry metrics

Precompute once per mesh:

```julia
struct FaceMetric{T}
    centroid_x::T
    centroid_y::T
    centroid_z::T
    h_max::T          # max edge length, or another conservative element diameter
    r_bound::T        # bounding-sphere radius from centroid to farthest vertex
    area::T
end
```

Recommended initial definitions:

\[
h_i = \max_{\text{edges } e \subset T_i} |e|
\]

\[
r_i = \max_{v \in T_i} \|v-c_i\|
\]

where \(c_i\) is the triangle centroid.

### 4.3 Pair geometry metrics

For a test triangle \(T_i\) and trial triangle \(T_j\):

\[
r_{ij} = \|c_i-c_j\|
\]

\[
d_{ij}^{gap} = \max(0, r_{ij} - r_i - r_j)
\]

\[
h_{ij} = \max(h_i,h_j)
\]

\[
\rho_{ij} = \frac{d_{ij}^{gap}}{h_{ij}+\epsilon}
\]

Here:

- \(r_{ij}\) is centroid distance;
- \(d_{ij}^{gap}\) is a conservative bounding-sphere gap;
- \(h_{ij}\) is the local element-size scale;
- \(\rho_{ij}\) is the normalized separation ratio.

Use \(\rho_{ij}\), not raw distance alone. A 10 cm gap means different things for 2 mm triangles than for 8 cm triangles.

### 4.4 Frequency-aware phase metric

At frequency \(f\):

\[
k = \frac{2\pi f}{c}
\]

\[
\theta_{ij} = k(h_i+h_j)
\]

Use \(\theta_{ij}\) as a crude measure of phase variation across the two elements. Reduced quadrature should become more conservative as \(\theta_{ij}\) grows.

---

## 5. Proposed tier policy

Use a two-stage policy:

1. **Static geometric ranking:** computed once from mesh geometry.
2. **Dynamic frequency promotion:** applied per frequency.

### 5.1 Static ranking

Assign each regular pair a base separation bucket:

```text
REGULAR_NEAR:
    rho < rho_mid

REGULAR_MID:
    rho_mid <= rho < rho_far

REGULAR_FAR:
    rho >= rho_far
```

Initial tuning knobs:

```julia
rho_mid = 3.0f0
rho_far = 8.0f0
```

These should be treated as starting points, not fixed constants. The validation sweep should determine useful values for your target meshes and frequency ranges.

### 5.2 Dynamic frequency promotion

At each frequency, promote pairs to safer tiers when the local phase metric is large.

Conceptual policy:

```julia
function choose_runtime_tier(base_tier, theta, operator_family, params)
    # theta = k * (h_i + h_j)

    # Hypersingular is more sensitive, so it gets stricter limits.
    guard = operator_family == :hypersingular ? params.hyp_guard_factor : 1.0f0

    theta_q2 = params.theta_q2_max * guard
    theta_q3 = params.theta_q3_max * guard

    if base_tier == REGULAR_FAR && theta <= theta_q2
        return Q_REDUCED_AGGRESSIVE
    elseif base_tier in (REGULAR_MID, REGULAR_FAR) && theta <= theta_q3
        return Q_REDUCED_MODERATE
    else
        return Q_FULL
    end
end
```

Initial tuning knobs:

```julia
theta_q2_max = 0.5f0
theta_q3_max = 1.0f0
hyp_guard_factor = 0.5f0
```

Again, these are only starting values. The important behavior is monotonic:

```text
as frequency increases:
    far q2 -> q3 -> q4
```

Never make high frequency less conservative than low frequency.

### 5.3 Conservative first policy

For the first numerical implementation, use only two runtime tiers:

```text
Q_FULL:
    current quadrature order 4

Q_REDUCED:
    one reduced rule, probably q2 or q3 depending on available quadrature rules
```

Then expand to three tiers after validation:

```text
Q_FULL
Q_REDUCED_MODERATE
Q_REDUCED_AGGRESSIVE
```

---

## 6. Worklist-based CUDA design

Do **not** sort or reorder the dense matrices. Sort or bucket the **regular-pair work**.

### 6.1 Current conceptual mapping

Current regular kernels effectively do this:

```julia
pair_linear = blockIdx().x
i = div(pair_linear, n_trial) + 1
j = mod(pair_linear, n_trial) + 1

if is_adjacent_or_singular(i, j)
    return
end

assemble_pair_full_quadrature!(i, j, ...)
```

### 6.2 Proposed mapping

Build compact pair-id lists:

```julia
regular_pairs_q4::CuArray{UInt32}
regular_pairs_q3::CuArray{UInt32}
regular_pairs_q2::CuArray{UInt32}
```

Then launch specialized kernels:

```julia
_cuda_regular_q4_slp_hyp_kernel!(..., regular_pairs_q4)
_cuda_regular_q4_dlp_adjoint_kernel!(..., regular_pairs_q4)

_cuda_regular_q3_slp_hyp_kernel!(..., regular_pairs_q3)
_cuda_regular_q3_dlp_adjoint_kernel!(..., regular_pairs_q3)

_cuda_regular_q2_slp_hyp_kernel!(..., regular_pairs_q2)
_cuda_regular_q2_dlp_adjoint_kernel!(..., regular_pairs_q2)
```

Inside each kernel:

```julia
work_idx = blockIdx().x
pair_linear = pair_ids[work_idx]

i = div(pair_linear, n_trial) + 1
j = mod(pair_linear, n_trial) + 1

assemble_pair_with_compile_time_rule!(i, j, ...)
```

This avoids a giant branchy kernel and lets each quadrature tier have:

- its own quadrature rule;
- its own accumulator loop length;
- its own register/shared-memory footprint;
- its own timing instrumentation.

### 6.3 Preserve balanced split

The current balanced split is good and should be preserved initially:

```text
SLP + hypersingular kernel
DLP + adjoint DLP kernel
```

The new dimension is pair-list tiering, not operator fusion.

Resulting launch layout:

```text
near q4:
    SLP/H
    DLP/D*

mid q3:
    SLP/H
    DLP/D*

far q2:
    SLP/H
    DLP/D*
```

That is more launches, but each launch is still simple and specialized. The launch overhead should be negligible relative to millions of regular pairs.

### 6.4 Optional conservative hypersingular mode

Hypersingular terms are likely the most sensitive. If validation shows the all-operator reduced quadrature mode is too aggressive, split the SLP/H kernel:

```text
SLP kernel
H kernel
DLP/D* kernel
```

Then use:

```text
SLP, DLP, D*:
    reduced quadrature allowed for mid/far regular pairs

H:
    q4 for near/mid
    reduced quadrature only for very far pairs
```

This adds code and one more operator-family split, so it should be a fallback if needed, not the first implementation.

---

## 7. Cache design

### 7.1 Extend the regular assembly cache

The existing regular assembly cache keeps fixed geometry and quadrature arrays resident across frequencies. Extend it with a pair-tier cache:

```julia
struct CudaRegularPairTierCache{T}
    n_test::Int32
    n_trial::Int32

    # Static geometry metrics
    face_h::CuArray{T}
    face_r_bound::CuArray{T}
    face_centroid_x::CuArray{T}
    face_centroid_y::CuArray{T}
    face_centroid_z::CuArray{T}

    # Static pair metadata
    pair_base_bucket::CuArray{UInt8}      # optional dense NxN bucket table
    pair_hsum::CuArray{T}                 # optional, for frequency promotion
    pair_rho::CuArray{T}                  # optional, for diagnostics/adaptive policy

    # Runtime worklists built per frequency, or built once per frequency band
    pair_ids_q4::CuArray{UInt32}
    pair_ids_q3::CuArray{UInt32}
    pair_ids_q2::CuArray{UInt32}

    # Counts
    count_q4::CuArray{Int32}
    count_q3::CuArray{Int32}
    count_q2::CuArray{Int32}
end
```

You do not need all of these arrays in the first implementation. A simple prototype can start with CPU-built worklists and only transfer:

```julia
pair_ids_q4
pair_ids_q3
pair_ids_q2
```

### 7.2 Memory cost

For the 7k-element benchmark:

```text
Npairs ≈ 49,000,000
UInt8 tier table ≈ 49 MB
UInt32 pair-id list for all regular pairs ≈ 196 MB
```

This is reasonable for the benchmark scale but could become expensive at larger sizes.

Recommended progression:

1. CPU-built pair lists for prototype.
2. GPU-built pair lists if classification time or memory transfer is significant.
3. Tile-level classification for larger meshes.
4. Cluster/block compression once moving beyond dense direct assembly.

---

## 8. Pair classification implementation options

### 8.1 Prototype: CPU classifier

Simplest first implementation:

```julia
function build_regular_pair_worklists_cpu(mesh, topology_cache, params)
    q4 = UInt32[]
    q3 = UInt32[]
    q2 = UInt32[]

    metrics = build_face_metrics(mesh)

    for i in 1:n_faces
        for j in 1:n_faces
            if is_singular_or_adjacent(topology_cache, i, j)
                continue
            end

            rho, hsum = pair_metrics(metrics, i, j)
            base = base_bucket(rho, params)

            # For first prototype, choose a single frequency or a conservative max frequency.
            tier = choose_tier_for_frequency(base, hsum, params.f_ref, params)

            pair_linear = UInt32((i-1) * n_faces + (j-1))
            push_to_worklist!(tier, pair_linear, q4, q3, q2)
        end
    end

    return upload_to_gpu(q4, q3, q2)
end
```

Pros:

- fastest to implement;
- easy to debug;
- deterministic;
- adequate to validate the numerical idea.

Cons:

- O(\(N_e^2\)) CPU loop;
- transfers pair lists to GPU;
- may become noticeable for larger meshes.

Given a 100-frequency sweep, even a moderately expensive one-time classification may be acceptable if it materially reduces each frequency assembly.

### 8.2 GPU classifier

Once the prototype works, move classification to GPU:

```text
kernel 1:
    one thread per pair
    skip singular/adjacent pairs
    compute rho and hsum
    compute runtime tier
    atomic count per tier

prefix sum:
    compute offsets

kernel 2:
    one thread per pair
    write pair id into compact q4/q3/q2 list
```

This avoids CPU bottlenecks and host-device transfer of large pair lists.

### 8.3 Frequency-band worklists

For 100-frequency sweeps, avoid rebuilding pair lists from scratch every frequency if possible.

Option A: **build runtime worklists per frequency**

```text
simple
robust
more overhead
```

Option B: **build worklists per frequency band**

```text
0-500 Hz
500-1000 Hz
1000-2000 Hz
...
```

Each band uses conservative promotion based on the highest frequency in that band.

Option C: **precompute max-safe frequencies per pair**

For each pair, precompute:

\[
f_{q2,max}^{ij} = \frac{c}{2\pi} \frac{\theta_{q2,max}}{h_i+h_j}
\]

\[
f_{q3,max}^{ij} = \frac{c}{2\pi} \frac{\theta_{q3,max}}{h_i+h_j}
\]

Then at runtime:

```julia
if rho >= rho_far && f <= f_q2_max
    q2
elseif rho >= rho_mid && f <= f_q3_max
    q3
else
    q4
end
```

This makes the runtime tier decision cheap and monotonic.

---

## 9. Symmetry handling

Implement in stages.

### 9.1 Stage 1: symmetry off

First validate with `symmetry = off`. This isolates pair classification from reflected image geometry.

### 9.2 Stage 2: identity-domain symmetry pairs

For symmetry modes, identity-domain singular handling remains unchanged:

```text
identity coincident / edge / vertex / adjacent
    -> normal Duffy correction cache
```

For regular identity-domain pairs, use the same near/far worklist logic as symmetry-off.

### 9.3 Stage 3: reflected image pairs

For reflected image regular contributions, classify using transformed trial/source geometry:

```text
test triangle: original reduced-domain geometry
trial triangle: reflected image geometry
```

The distance and gap metrics should use reflected centroids and reflected bounding volumes.

Keep image-singular corrections unchanged:

```text
reflected coincident / edge / vertex / adjacent
    -> image-singular correction cache
    -> FP32 correction path
```

Do not let the regular far-pair tiering absorb image-singular cases.

---

## 10. Benchmark instrumentation

Add timing and count fields to the benchmark JSON.

### 10.1 Pair counts

```json
{
  "regular_pair_tiering_enabled": true,
  "regular_pair_tier_policy": "rho_phase_promoted_v1",
  "regular_pairs_q4": 12345678,
  "regular_pairs_q3": 23456789,
  "regular_pairs_q2": 13000000,
  "regular_pairs_total": 48906260
}
```

### 10.2 Effective quadrature work

Track effective quadrature-pair count:

```json
{
  "regular_kernel_qpair_count_baseline": 36,
  "regular_kernel_effective_qpair_evals": 1234567890,
  "regular_kernel_baseline_qpair_evals": 1760625360,
  "regular_kernel_qpair_reduction_ratio": 0.30
}
```

Formula:

\[
Q_{eff} =
N_{q4} q_4^2 +
N_{q3} q_3^2 +
N_{q2} q_2^2
\]

\[
\text{reduction} =
1 - \frac{Q_{eff}}{N_{regular} q_4^2}
\]

### 10.3 Per-tier timings

```json
{
  "regular_operator_kernel_q4_slp_hyp": 0.0,
  "regular_operator_kernel_q4_dlp_adjoint": 0.0,
  "regular_operator_kernel_q3_slp_hyp": 0.0,
  "regular_operator_kernel_q3_dlp_adjoint": 0.0,
  "regular_operator_kernel_q2_slp_hyp": 0.0,
  "regular_operator_kernel_q2_dlp_adjoint": 0.0,
  "regular_pair_classification_cache_build": 0.0,
  "regular_pair_worklist_build": 0.0
}
```

### 10.4 Accuracy metrics

Add optional benchmark output for comparison against a reference run:

```json
{
  "reference_regular_assembly_mode": "split_atomic_balanced",
  "reference_quadrature_order": 4,
  "pressure_relative_l2_error": 0.0,
  "rhs_relative_l2_error": 0.0,
  "spl_p95_abs_error_db": 0.0,
  "spl_max_abs_error_db": 0.0,
  "impedance_mag_relative_error": 0.0,
  "impedance_phase_abs_error_deg": 0.0
}
```

---

## 11. Validation plan

### 11.1 Unit tests

#### Pair classification

Verify:

```text
all singular/adjacent pairs are excluded from reduced regular worklists
all regular pairs appear in exactly one runtime worklist
q4 + q3 + q2 count == regular_pairs
pair_linear maps back to correct (i,j)
symmetry-off worklists are deterministic
```

#### Kernel equivalence

First run tiered kernels with all pairs forced to q4:

```text
regular_assembly_mode = split_atomic_balanced
regular_assembly_mode = split_atomic_balanced_tiered_fp32, force_all_q4=true
```

Expected result:

```text
same numerical output within existing FP32 atomic-order tolerance
similar or slightly worse runtime due to worklist indirection and extra launches
```

This test validates the new worklist path before changing quadrature.

### 11.2 Matrix-level validation

For small meshes where full operator comparisons are cheap, compute:

\[
\frac{\|S_{tiered}-S_{ref}\|_F}{\|S_{ref}\|_F}
\]

\[
\frac{\|D_{tiered}-D_{ref}\|_F}{\|D_{ref}\|_F}
\]

\[
\frac{\|D^*_{tiered}-D^*_{ref}\|_F}{\|D^*_{ref}\|_F}
\]

\[
\frac{\|H_{tiered}-H_{ref}\|_F}{\|H_{ref}\|_F}
\]

Also inspect row-wise errors:

\[
\max_i
\frac{\|A_{tiered}[i,:]-A_{ref}[i,:]\|_2}
{\|A_{ref}[i,:]\|_2+\epsilon}
\]

Matrix norm errors alone are not enough, but they are useful for diagnosing which operator family is causing error.

### 11.3 Solve-level validation

For each frequency:

\[
e_p =
\frac{\|p_{tiered}-p_{ref}\|_2}{\|p_{ref}\|_2}
\]

\[
e_{res} =
\frac{\|A_{tiered}p_{tiered}-b_{tiered}\|_2}{\|b_{tiered}\|_2}
\]

Also compare application outputs:

```text
SPL p50 / p95 / max absolute error
radiator impedance magnitude relative error
radiator impedance phase absolute error
peak frequency shift
null frequency shift
frequency sweep smoothness
```

### 11.4 Suggested initial acceptance gates

These are starting gates, not final product requirements:

```text
pressure relative L2 error:
    <= 1e-3 for non-resonant frequencies

SPL p95 absolute error:
    <= 0.05 dB

SPL max absolute error:
    <= 0.2 dB outside deep nulls

impedance magnitude relative error:
    <= 0.5%

impedance phase absolute error:
    <= 0.5 degrees

frequency sweep:
    no visible nonphysical jaggedness introduced by tier changes
```

Deep nulls require special handling because tiny complex-pressure differences can become large dB differences.

---

## 12. Rollout phases

### Phase 0 — Instrumentation only

Goal: understand how much of the pair set is geometrically reducible.

Tasks:

```text
build face metrics
compute rho histogram
compute hsum histogram
compute estimated tier counts for each frequency
do not change assembly
dump benchmark fields
```

Useful outputs:

```json
{
  "rho_histogram": "...",
  "regular_pair_base_near_count": 0,
  "regular_pair_base_mid_count": 0,
  "regular_pair_base_far_count": 0,
  "regular_pair_runtime_q4_count": 0,
  "regular_pair_runtime_q3_count": 0,
  "regular_pair_runtime_q2_count": 0
}
```

No numerical risk.

### Phase 1 — Worklist path, all q4

Goal: validate the worklist kernel infrastructure without approximation.

Tasks:

```text
build q4 worklist containing all regular pairs
launch q4 worklist versions of current balanced split kernels
compare against current split_atomic_balanced mode
```

Expected result:

```text
accuracy: same within FP32 atomic tolerance
speed: similar or slightly slower
```

This establishes that worklist scheduling is correct.

### Phase 2 — Two-tier FP32 reduced quadrature

Goal: first real speed/accuracy experiment.

Policy:

```text
near regular:
    q4

far regular:
    q3 or q2, based on conservative rho and theta thresholds
```

Keep:

```text
FP32 arithmetic
FP32 accumulation
same dense matrices
same Burton-Miller solve
same Duffy correction path
```

Benchmark:

```text
single frequency
small mesh matrix diff
benchmark mesh solve diff
multiple threshold settings
```

### Phase 3 — Three-tier FP32 policy

Goal: improve the speed/accuracy tradeoff.

Policy:

```text
near:
    q4

mid:
    q3

far:
    q2
```

Add frequency promotion:

```text
as f increases, promote q2 -> q3 -> q4
```

Measure sweep smoothness. Watch for discontinuities when tier decisions change between adjacent frequencies.

### Phase 4 — Operator-specific policy

Goal: control hypersingular sensitivity if needed.

Possible policies:

```text
Policy A:
    all operators share same pair quadrature tier

Policy B:
    H uses stricter thresholds

Policy C:
    H stays q4 except very far pairs

Policy D:
    split SLP and H into separate kernels
```

Use matrix-level diagnostics to decide whether this complexity is needed.

### Phase 5 — Compression and mixed precision exploration

Only after reduced quadrature is validated.

Potential next steps:

```text
far block storage quantization, starting with SLP
low-rank or block-compressed far interactions
iterative solve with dense near field + compressed far field
BF16/FP16 storage experiments
FP4 only for compressed far factors, not raw dense entries
```

Do not mix these with the first reduced-quadrature rollout. Keep the experiment isolated.

---

## 13. Configuration proposal

Add a structured config block rather than ad hoc flags.

Example request JSON:

```json
{
  "regular_assembly_mode": "split_atomic_balanced_tiered_fp32",
  "regular_pair_tiering": {
    "enabled": true,
    "classifier": "rho_phase_v1",
    "rho_mid": 3.0,
    "rho_far": 8.0,
    "theta_q2_max": 0.5,
    "theta_q3_max": 1.0,
    "hypersingular_guard_factor": 0.5,
    "force_all_q4": false,
    "symmetry_tiering": false,
    "debug_dump_histograms": true
  }
}
```

Modes:

```text
split_atomic_balanced:
    current baseline

split_atomic_balanced_tiered_fp32:
    worklist pair-tier path with FP32 arithmetic and accumulation

split_atomic_balanced_tiered_fp32_force_q4:
    worklist path, no quadrature reduction; correctness test mode
```

---

## 14. Code integration sketch

### 14.1 New files or modules

Potential locations:

```text
AfterburnerCore.jl:
    shared pair metric definitions
    classifier parameter struct
    CPU classifier prototype

AfterburnerCuda.jl:
    CudaRegularPairTierCache
    GPU worklist upload/build helpers
    tiered regular kernels
    benchmark timing fields

AfterburnerCudaProfiling.jl:
    optional tier histogram probes
    threshold sweep tools
```

### 14.2 New data types

```julia
@enum RegularPairBaseBucket::UInt8 begin
    REGULAR_BASE_NEAR = 0
    REGULAR_BASE_MID  = 1
    REGULAR_BASE_FAR  = 2
end

@enum RuntimeQuadratureTier::UInt8 begin
    Q_FULL = 0
    Q_REDUCED_MODERATE = 1
    Q_REDUCED_AGGRESSIVE = 2
end

struct RegularPairTieringParams{T}
    enabled::Bool
    rho_mid::T
    rho_far::T
    theta_q2_max::T
    theta_q3_max::T
    hypersingular_guard_factor::T
    force_all_q4::Bool
    symmetry_tiering::Bool
end
```

### 14.3 Worklist construction

```julia
function build_regular_pair_tier_worklists(
    mesh::BoundaryMesh,
    adjacency_cache,
    freq_hz::T,
    sound_speed::T,
    params::RegularPairTieringParams{T},
) where {T}

    metrics = build_face_metrics(mesh)
    k = T(2π) * freq_hz / sound_speed

    q4 = UInt32[]
    q3 = UInt32[]
    q2 = UInt32[]

    for i in 1:mesh.nfaces
        for j in 1:mesh.nfaces
            if is_singular_or_adjacent(adjacency_cache, i, j)
                continue
            end

            rho, hsum = compute_pair_rho_hsum(metrics, i, j)
            theta = k * hsum

            tier = choose_runtime_tier(rho, theta, params)

            pair_linear = UInt32((i-1) * mesh.nfaces + (j-1))

            if params.force_all_q4 || tier == Q_FULL
                push!(q4, pair_linear)
            elseif tier == Q_REDUCED_MODERATE
                push!(q3, pair_linear)
            else
                push!(q2, pair_linear)
            end
        end
    end

    return CudaRegularPairTierWorklists(
        CuArray(q4),
        CuArray(q3),
        CuArray(q2),
    )
end
```

### 14.4 Runtime integration

Inside the per-frequency operator assembly:

```julia
if regular_assembly_mode == :split_atomic_balanced
    assemble_regular_split_atomic_balanced!(...)
elseif regular_assembly_mode == :split_atomic_balanced_tiered_fp32
    worklists = get_or_build_tier_worklists(cache, frequency, params)

    assemble_regular_tier_q4!(..., worklists.q4)
    assemble_regular_tier_q3!(..., worklists.q3)
    assemble_regular_tier_q2!(..., worklists.q2)

    apply_singular_duffy_corrections!(...)
else
    error("unsupported regular assembly mode")
end
```

Important: singular corrections remain after regular assembly, same as now.

---

## 15. Things not to do in the first implementation

Avoid these initially:

```text
Do not reduce or approximate Duffy corrections.

Do not use centroid distance alone as the classifier.

Do not drop far-pair entries entirely.

Do not change FP32 kernel evaluation precision.

Do not change FP32 atomic accumulation.

Do not quantize dense matrices.

Do not introduce compressed matrix storage before direct dense baseline parity is established.

Do not combine pair tiering, mixed precision, and solver changes in one experiment.

Do not make high-frequency runs less conservative than low-frequency runs.
```

---

## 16. Expected speed model

Let:

```text
N4 = number of q4 regular pairs
N3 = number of q3 regular pairs
N2 = number of q2 regular pairs
```

Let:

```text
Q4 = q4 quadrature-pair count
Q3 = q3 quadrature-pair count
Q2 = q2 quadrature-pair count
```

Then the idealized regular-kernel work ratio is:

\[
R =
\frac{N_4 Q_4 + N_3 Q_3 + N_2 Q_2}
{(N_4+N_3+N_2)Q_4}
\]

The idealized regular-kernel speedup is:

\[
S_{regular} \approx \frac{1}{R}
\]

The end-to-end speedup is bounded by the regular assembly share of total runtime. Since the current benchmark is dominated by regular assembly, even moderate regular-kernel reductions may be meaningful.

Example:

```text
60% of regular pairs move from q4 to q2
q4 pair count = 36
q2 pair count = 9
```

Then:

\[
R = 0.4 + 0.6 \frac{9}{36} = 0.55
\]

\[
S_{regular} \approx 1.82
\]

Actual speedup will be lower due to memory traffic, atomics, launch overhead, and non-quadrature overhead, but this gives a useful target model.

---

## 17. Open design questions

Resolve these experimentally:

1. **What separation thresholds are safe for your waveguide/enclosure workloads?**

   Start with conservative \(\rho\) thresholds and tune using full frequency sweeps.

2. **Should hypersingular use reduced quadrature at the same thresholds as SLP/DLP/D\*?**

   If not, split SLP/H or use stricter hypersingular promotion.

3. **How should tier transitions be smoothed across frequency?**

   If frequency-to-frequency output becomes jagged, use frequency bands or hysteresis.

4. **Is CPU classification acceptable for production meshes?**

   If not, move classification and partitioning to GPU.

5. **Does the tiered path interact with symmetry image contributions cleanly?**

   Implement symmetry after the non-symmetry path is validated.

6. **At what mesh size should tile-level classification replace pair-level lists?**

   Pair-level lists are straightforward at 7k faces but will scale quadratically in memory.

---

## 18. Recommended first milestone

Implement this minimal milestone:

```text
Mode name:
    split_atomic_balanced_tiered_fp32_force_q4

Behavior:
    build one regular-pair worklist containing all topologically regular pairs
    launch q4 worklist versions of the existing two balanced kernels
    keep Duffy corrections unchanged
    compare against split_atomic_balanced baseline
```

Success criteria:

```text
q4 worklist result matches baseline within FP32 atomic-order tolerance
regular pair count equals benchmark regular_pairs
singular pair count unchanged
runtime overhead is measured and acceptable
benchmark JSON reports worklist counts and timings
```

Then implement:

```text
Mode name:
    split_atomic_balanced_tiered_fp32

Behavior:
    q4 near worklist
    q3/q2 far worklist
    frequency-aware promotion
    FP32 everywhere
```

Success criteria:

```text
regular kernel time decreases
application-level outputs remain within acceptance gates
frequency sweep remains smooth
Duffy path remains untouched
```

---

## 19. Long-term path after this plan

Once the FP32 reduced-quadrature path is validated, the natural next steps are:

```text
1. operator-specific far policies
2. tile-level near/far classification
3. block-level far compression
4. iterative solver path with dense near field + compressed far field
5. BF16/FP16 far storage experiments
6. FP4 only for compressed far factors, not raw dense BEM entries
```

The key architectural direction is:

```text
FP32 singular/near field
+ reduced-quadrature regular far field
+ eventually compressed far field
+ only then mixed precision for storage/factors
```

That sequence minimizes numerical risk while still moving toward the hardware-friendly structure needed for future mixed-precision acceleration.
