# BEAT Engine extraction milestones

BEAT will become an independently versioned numerical engine consumed by Boundary
Lab. The repository move follows retirement of the application's legacy runtime.

## Application runtime retirement

- GUI exterior, interior, and coupled solves use `SystemSolveRequest` and
  `PhysicalSystemProductionBackend`. There is no exterior compatibility fallback.
- Bempp, its multiprocessing worker, the legacy HTTP server/client, and the old
  server Docker image are removed. Bempp and PyOpenCL are no longer dependencies
  or startup requirements.
- The old `blab solve` and `blab server` entry points only explain retirement and
  direct users to the project workflow. They do not execute numerical jobs.
- Saved Bempp/Server preferences migrate to BEAT CPU; existing BEAT selections
  are preserved. The headless CLI retains its CUDA-when-functional/CPU-fallback
  default.
- Legacy project migration and historical result readers remain. Live plot
  projections still use `FrequencyResult`; canonical results come only from the
  physical-system worker, preserving the per-excitation basis.

This does not yet remove every legacy-shaped internal contract. The BEAT
source-request adapter and Julia entry points remain for numerical comparisons,
benchmarks, and the Deploy prototype. These are not GUI or project-CLI execution
routes. The shared worker infrastructure is now separated as described below.

## Reusable worker boundary

`src/blab/solvers/beat_worker.py` is a standard-library-only subprocess client.
`WorkerProcess` receives executable/script paths and a child-process environment,
streams opaque JSON events, and supports the existing solve and field operations.
It has no dependency on Boundary Lab models, preferences, NumPy, Qt, or bundled
Julia locations. `WorkerPool` owns pooled workers and shuts them down explicitly;
directly constructed workers remain the caller's responsibility.

`src/blab/solvers/beat_engine_runtime.py` supplies the application-specific Julia
paths, ROCm discovery, process environment, and one shared pool. Physical-system,
retained-field, headless, and Deploy callers use this module directly through
`get_beat_engine_worker`, `BeatEngineWorkerProcess`, and `julia_process_env`.
The old source adapter re-exports compatibility names for existing reference
harnesses, using the same implementation and pool.

Worker environments are copied at construction. Pool identity includes executable,
script, project, sysimage, resolved thread count, and environment, so SDK or runtime
environment changes create a separately configured worker. Shutdown clears the
pool and terminates its processes. Request/result schemas and numerical code are
unchanged by this separation.

## Prepare the extraction boundary

1. **Complete:** separate reusable worker/client infrastructure from the older
   `BeatEngineBackend` source-request adapter, free of application models,
   project preferences, and Qt.
2. Define an engine-owned compiled-system wire specification from the existing
   versioned system contract. Boundary Lab keeps project authoring, migration,
   mesh preparation, and compilation into that specification.
3. Add worker version/capability negotiation and protocol compatibility checks
   before a solve. Specify complex array layout, field-evaluation cache lifetime,
   cancellation, and error semantics.
4. Record engine revision, Julia environment, precision, actual backend/device,
   mesh hashes, and numerical options in run provenance.
5. Preserve independent numerical comparisons and analytical/reference fixtures
   before removing the remaining source-request harness. Cover exterior BEM,
   interior FEM, coupled FEM-BEM-LEM, multiple excitations, symmetry, retained
   fields, and complex-pressure probes.

## Extract and release

Move the Julia engine, backend tests, numerical fixtures, and a small Python
worker client into one BEAT repository. Publish a versioned release before
changing Boundary Lab's dependency. Keep CPU, CUDA, ROCm, and future Metal
implementations together initially. Provide a contributor checkout override and
an explicit Julia/runtime installation strategy; a Python package alone does not
provision the numerical runtime.

Boundary Lab should test a pinned engine release through the headless project
workflow and GUI integration tests. BEAT should own numerical regression tests
and qualify accelerator releases on the corresponding hardware. A future remote
service must implement the physical-system contract rather than restore the
retired source-model HTTP protocol.
