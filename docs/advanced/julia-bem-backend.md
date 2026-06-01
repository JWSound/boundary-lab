# Julia BEM Backend

Boundary Lab's Julia backend is a local direct dense BEM solver used through `src/blab/solvers/julia_local_backend.py`. The Python side stages mesh assets and request JSON, while `src/blab/solvers/julia_local/solver.jl` owns the numerical solve. The CUDA implementation in `src/blab/solvers/julia_local/src/JBEMCuda.jl` is the primary high-performance path; `JBEMCore.jl` provides shared mesh, quadrature, formulation, and fallback utilities.

The backend solves exterior acoustic radiation from prescribed normal velocity on tagged radiator surfaces. It currently uses single precision (`Float32`) for local solves.

## Solve Pipeline

For each solve request, Julia:

1. Loads one or more Gmsh 2.2 ASCII triangle meshes.
2. Applies per-mesh scale and translation, then combines them into one `BoundaryMesh`.
3. Builds P1 pressure and DP0 velocity spaces.
4. Builds frequency-independent identity matrices.
5. For each frequency, assembles Helmholtz boundary operators.
6. Solves the Burton-Miller Neumann system for boundary pressure.
7. Evaluates SPL at polar and optional spherical observation points.
8. Computes per-radiator acoustic impedance from pressure over driven elements.

Radiators are resolved by both physical tag and mesh id, so duplicate physical tags are allowed across meshes when the radiator specifies its mesh.

## Boundary Integral Formulation

The backend uses the outgoing Helmholtz Green function

$$
G_k(x,y) = \frac{e^{i k \lVert y-x \rVert}}{4\pi \lVert y-x \rVert}.
$$

The assembled operator tuple contains:

- `single_layer`: maps DP0 Neumann data to P1 test functions.
- `double_layer`: maps P1 pressure data to P1 test functions.
- `adjoint_double_layer`: maps DP0 Neumann data to P1 test functions.
- `hypersingular`: maps P1 pressure data to P1 test functions.

The dense Burton-Miller system is formed as:

$$
\left(\frac{1}{2} I_{P1,P1} - D + \eta H\right)p
=
\left(-S - \eta\left(D^{*} + \frac{1}{2} I_{P1,DP0}\right)\right)q,
$$

where:

$$
\eta = \frac{i}{k}.
$$

Here \(p\) is the solved boundary pressure and \(q\) is the Neumann/radiator drive vector. Radiator normal velocity is converted to Neumann data with:

$$
q = i \rho \omega v_n.
$$

## CUDA Operator Assembly

The CUDA path assembles dense Galerkin operators by splitting triangle pairs into regular pairs and adjacent/coincident singular pairs. Regular pairs are assembled by CUDA kernels over test/trial element pairs. Adjacent, edge-sharing, vertex-sharing, and coincident pairs are handled by a separate GPU Duffy correction path.

For a test triangle \(T_i\), trial triangle \(T_j\), test basis function \(\phi_a\), and trial basis function \(\psi_b\), a typical single-layer block entry is:

$$
S_{ab}^{ij}
=
\int_{T_i}\int_{T_j}
\phi_a(x) G_k(x,y) \psi_b(y)\,dS_y\,dS_x.
$$

The double-layer and adjoint double-layer use normal derivatives of \(G_k\):

$$
D_{ab}^{ij}
=
\int_{T_i}\int_{T_j}
\phi_a(x) \frac{\partial G_k(x,y)}{\partial n_y} \psi_b(y)\,dS_y\,dS_x,
$$

$$
(D^{*})_{ab}^{ij}
=
\int_{T_i}\int_{T_j}
\phi_a(x) \frac{\partial G_k(x,y)}{\partial n_x} \psi_b(y)\,dS_y\,dS_x.
$$

The hypersingular block is assembled with the surface-curl weak form:

$$
H_{ab}^{ij}
=
\int_{T_i}\int_{T_j}
G_k(x,y)
\left[
\operatorname{curl}_\Gamma \phi_a(x)\cdot\operatorname{curl}_\Gamma \psi_b(y)
- k^2 \phi_a(x)\psi_b(y)n_x\cdot n_y
\right]\,dS_y\,dS_x.
$$

The Julia implementation stores all four dense matrices explicitly. This makes GPU dense solves straightforward, but memory usage scales quadratically with the P1/DP0 unknown counts.

The application backend requires CUDA and moves geometry and quadrature arrays to the GPU:

- face vertices
- normals
- areas
- global face indices
- surface curls
- quadrature points and weights
- test/trial element index arrays

`build_cuda_regular_assembly_cache` keeps these arrays resident across frequencies, avoiding repeated host-to-device transfers for fixed mesh geometry.

For each frequency, CUDA allocates real and imaginary dense matrices for:

- SLP real/imag
- DLP real/imag
- adjoint DLP real/imag
- hypersingular real/imag

The regular-pair kernel skips adjacent pairs. Singular and near-singular corrections use Duffy quadrature and are added after regular assembly. In the application path, `return_gpu=true`, so the regular dense matrices stay on GPU and CUDA evaluates the Duffy correction blocks on GPU before adding them to the resident operators.

The singular path builds a frequency-independent correction cache for the mesh and singular quadrature order. The cache stores adjacent/coincident element pairs, orientation-remapped Duffy rules, surface curls, and pair geometry scalars once, then reuses them across the frequency loop. Per-frequency work still evaluates the Helmholtz kernels because they depend on \(k\).

When CUDA operators are resident on GPU, a Duffy block kernel computes compact per-pair correction blocks directly on device. A CUDA scatter kernel atomically accumulates those compact blocks into dense GPU correction buffers before adding them to the resident operators. The GPU Duffy kernel reuses the regular assembly geometry cache, so the per-frequency singular correction path does not transfer dense CPU correction matrices.

The CUDA singular correction cache stores:

- test and trial element indices
- rule indices for orientation-remapped Duffy rules
- P1 row/column dofs and DP0 columns
- pair Jacobian scales and normal products
- flattened remapped Duffy points and weights

## CUDA Kernel Modes

The CUDA backend has multiple regular assembly kernels, one singular correction block kernel, and two field-evaluation kernels.

`_cuda_regular_kernel!` maps GPU threads over element pairs. Each thread computes all quadrature pairs for one or more test/trial element pairs.

`_cuda_regular_quadrature_kernel!` is the fused parallel-quadrature regular assembly kernel. It maps one CUDA block to one test/trial element pair and distributes the quadrature-pair work across threads in the block. Per-thread partial sums are reduced in dynamic shared memory:

```julia
scratch = CUDA.@cuDynamicSharedMem(typeof(k), blockDim().x * accumulator_count)
```

The fused kernel computes all regular operators in one pass:

- single layer;
- adjoint double layer;
- double layer;
- hypersingular.

It then atomically scatters all real and imaginary element-block entries into dense operator buffers. This path remains available as `regular_assembly_mode=:fused` and is useful as a reference/fallback implementation.

The default CUDA application path uses faster `regular_assembly_mode=:split_atomic`. This keeps the same one-block-per-element-pair parallel quadrature strategy and the same dense atomic accumulation model, but splits regular assembly into two kernels:

- `_cuda_regular_quadrature_slp_adjoint_kernel!` computes single layer and adjoint double layer contributions;
- `_cuda_regular_quadrature_dlp_hyp_kernel!` computes double layer and hypersingular contributions.

The split atomic method intentionally recomputes the Green-function geometry in each split kernel, but it reduces the number of live accumulators and the dynamic shared-memory reduction width per launch. On moderate real-world meshes this has benchmarked faster than the fused all-operator kernel, because the fused kernel's 48 accumulator slots create enough register/shared-memory pressure to outweigh the extra launch and repeated math.

The application default can be overridden in request config with:

```json
{
  "julia_cuda_regular_assembly_mode": "fused"
}
```

The benchmarking script exposes the same choice with `--regular-assembly-mode fused|split_atomic`.

`_cuda_duffy_blocks_kernel!` maps GPU threads over cached adjacent/coincident element pairs. Each thread computes the compact singular correction block for one or more pairs using the cached remapped Duffy rule. `_cuda_singular_scatter_kernel!` then atomically scatters those compact blocks into dense GPU correction buffers.

`_cuda_weighted_field_sources_kernel!` maps GPU threads over cached source quadrature points and builds weighted pressure and Neumann source strengths. `_cuda_field_eval_kernel!` maps one CUDA block to one observation point and reduces source contributions in dynamic shared memory.

## CUDA Atomics

The GPU kernels accumulate element-block contributions into global dense matrices. Multiple CUDA blocks can target the same global row/column entries, especially because P1 basis functions are shared across adjacent faces. The regular assembly and singular scatter paths use device atomics for global accumulation:

```julia
@inline function _cuda_atomic_add!(array, index, value)
    CUDA.@atomic array[index] += value
    return nothing
end
```

Dense operator accumulation buffers are stored as separate real and imaginary arrays during kernel execution, so atomic additions operate on scalar `Float32` values. After the kernel finishes, real and imaginary matrices are materialized into complex matrices on GPU:

$$
A = A_{\mathrm{re}} + i A_{\mathrm{im}},
$$

This avoids a slow serial scatter stage and lets regular-pair assembly remain massively parallel. The default split atomic regular assembly mode preserves this property: it still atomically accumulates into the same dense buffers, but does so through smaller operator-family kernels instead of one all-operator kernel. The tradeoff is that floating-point atomic accumulation is order-dependent, so tiny run-to-run differences can occur at the last few bits.

## GPU Dense Solve

If operators are returned on GPU, `solve_burton_miller_neumann` forms the Burton-Miller matrix and RHS directly as `CuArray`s:

$$
A = \frac{1}{2}I - D + \eta H,
$$

$$
b = \left(-S - \eta(D^{*} + \frac{1}{2}I_{P1,DP0})\right)q.
$$

It then calls Julia's dense linear solve:

```julia
d_pressure = d_lhs \ d_rhs
```

With CUDA arrays, this dispatches through CUDA.jl to GPU dense linear algebra. The solved pressure vector is currently copied back to CPU after the solve; CUDA field evaluation then transfers pressure and Neumann vectors back to the GPU for the observation pass. This keeps the public result path simple while leaving room to keep pressure resident across downstream stages later.

Temporary GPU allocations are explicitly released with `CUDA.unsafe_free!` after assembly or solve stages to reduce memory pressure during frequency sweeps.

## Field Evaluation

After boundary pressure is solved, the backend evaluates the potential at observation points:

$$
u(x)
=
\int_\Gamma
\frac{\partial G_k(x,y)}{\partial n_y}p(y)
- G_k(x,y)q(y)
\,dS_y.
$$

The implementation precomputes a field-evaluation cache containing quadrature source points, normals, weights, source faces, source elements, and P1 basis values. `build_cuda_field_evaluation_cache` mirrors those arrays to GPU as a `CudaFieldEvaluationCache`, using structure-of-arrays storage for coalesced source-point and normal reads.

For each frequency, CUDA field evaluation uses two stages:

1. `_cuda_weighted_field_sources_kernel!` interpolates solved P1 pressure to every source quadrature point, multiplies by quadrature weights, and builds the weighted Neumann source term.
2. `_cuda_field_eval_kernel!` assigns one CUDA block to each observation point. Threads in the block stride over all source quadrature points, evaluate the single-layer and double-layer kernels, accumulate local real/imaginary potentials, and reduce those partial sums in dynamic shared memory.

The kernel evaluates:

$$
u(x)
=
\sum_m
\left[
\frac{\partial G_k(x,y_m)}{\partial n_m}p_m
- G_k(x,y_m)q_m
\right]w_m,
$$

where \(y_m\), \(n_m\), and \(w_m\) come from the cached source quadrature data. The GPU result is materialized as a compact potential vector and copied back for SPL conversion and result serialization.

SPL is reported as:

$$
\mathrm{SPL}(x)
=
20\log_{10}\left(\frac{|u(x)|}{20\ \mu\mathrm{Pa}}\right).
$$

Horizontal, vertical, and spherical observation points are concatenated into one field evaluation per frequency, then sliced back into result arrays. This avoids rebuilding source strengths for each observation set.

## Performance Notes

The expensive stages are:

- dense operator assembly, roughly \(O(N_e^2 q^2)\), where \(N_e\) is face count and \(q\) is quadrature point count;
- dense direct solve, roughly \(O(N_p^3)\), where \(N_p\) is P1 dof count;
- field evaluation, roughly \(O(N_{\mathrm{obs}} N_e q)\).

CUDA accelerates regular-pair assembly, singular Duffy corrections, the dense solve, and field evaluation. The remaining dominant cost in CUDA solves is usually regular-pair assembly for the dense operators, followed by the dense solve for larger P1 systems. The default split atomic regular assembly mode reduces regular-kernel pressure by assembling SLP/adjoint and DLP/hypersingular in separate launches while keeping dense operators resident on the GPU.

## Important Files

- `src/blab/solvers/julia_local/solver.jl`: request handling, mesh/radiator setup, frequency loop, drive calculation.
- `src/blab/solvers/julia_local/src/JBEMCore.jl`: mesh representation, shared quadrature/formulation code, Burton-Miller solve, field evaluation interfaces.
- `src/blab/solvers/julia_local/src/JBEMCuda.jl`: CUDA geometry cache, regular-pair kernels, GPU Duffy corrections, GPU atomics, GPU matrix materialization, GPU field evaluation.
- `src/blab/solvers/julia_local/src/JBEMCudaProfiling.jl`: optional CUDA regular-kernel probe and profiling launches used by benchmark scripts.
- `src/blab/solvers/julia_local_backend.py`: Python adapter that stages assets and streams JSON events.
