# Container development

The root `Dockerfile` builds the Boundary Lab service with its pinned BEAT Engine
wheel. See [Server developer reference](boundary-lab-server.md) for the contract
and [Docker setup](../Docker.md) for operator instructions.

## Build structure

The `cpu` and `cuda` targets share Debian Bookworm with Julia 1.12.6 pinned by
image digest. Python runs in `/opt/venv`; GUI extras are omitted. Published BEAT
manifests determine Julia dependencies. CUDA preparation selects runtime 12.8
and imports CUDA/CUDSS during the build, including GPU compiler preparation
without a build GPU. See [CUDA.jl container preparation](https://cuda.juliagpu.org/stable/installation/overview/#Precompiling-CUDA.jl-without-CUDA).

`JULIA_CPU_TARGET=generic` makes cached code portable across x86 hosts.
`/opt/julia-depot` holds prepared dependencies and allows cache writes by the
unprivileged service user. Startup never runs package installation; first-use
kernel compilation can still take time.

Installed versions are recorded in `/opt/python-packages.txt` and
`/opt/beat-cpu.json`. Pass `--build-arg VCS_REF=<commit>` for the application OCI
revision label. Python transitive dependencies and Debian packages resolve at
build time: this is not yet a fully locked release build. Retain the tested
image digest when qualifying it.

Tini forwards termination to the entrypoint, which routes SIGTERM through normal
server cleanup. Health checks authenticate without printing keys. Readiness
means some solving is available; GPU qualification must also assert CUDA
availability and the actual completed job backend.

## Local qualification

From a checkout with Python application dependencies installed:

```bash
python scripts/check_docker_integration.py --image boundarylabserver:local-cpu --backend cpu --output runs/docker-cpu
python scripts/check_docker_integration.py --image boundarylabserver:local-cuda --backend cuda --output runs/docker-cuda
```

Use a new output directory each time. CUDA requires an NVIDIA GPU exposed through
Docker. The script validates Simple Sealed before solving its coupled physical
system at 500 Hz. It compares a direct solve inside the image with a solve
submitted from the host client. CPU requires exact equality; CUDA uses the
established `1e-5` normalized maximum error tolerance. Complex results, excitation
identities, actual backend, and engine provenance are checked.

The downloaded arrays are also compared exactly with the server's saved result
event. This separates transport integrity from differences between independent
GPU solves. A numerical comparison failure does not skip lifecycle checks; the
report retains both outcomes and the script still exits unsuccessfully.

It also checks missing/wrong keys, hosted startup rejection without a key,
private mode without a key, cancellation, SIGTERM during active work, and
journal persistence across container replacement. The server uses an internal
Docker network without outbound access. A disposable TCP relay exposes HTTP
to localhost. Temporary containers, networks, and volumes are removed; logs
and results remain under the requested `runs/` directory.

Provider HTTPS, large uploads/results, other GPU/driver combinations, publishing,
and a Runpod template remain subsequent work. ROCm and ARM64 are not qualified
by this milestone.

## Qualification findings

Local testing on Docker Desktop Linux/AMD64 with an NVIDIA RTX 2080 Ti found:

- CPU: all five coupled-system result arrays match exactly between direct and
  remote solves; authentication and lifecycle checks pass.
- CUDA: offline runtime discovery and matrix operations pass. The direct and
  remote coupled solves complete using CUDA/cuDSS, but retained BEM boundary
  pressure differs by `2.142e-4` normalized maximum error, above the `1e-5` gate.
- Repeating the direct CUDA solve also differs from the first direct solve
  (`9.347e-5` for the same boundary-pressure quantity). Other result quantities
  in that comparison remain within the gate. This finding is not specific to
  remote submission.
- CUDA authentication, cancellation, graceful shutdown, retained jobs after
  replacement, and CPU availability with GPU access disabled all pass. All five
  downloaded arrays exactly match the server's saved result event. A second
  independent direct/remote comparison still exceeds the numerical gate
  (`4.441e-5` boundary-pressure error).

The comparison tolerance has not been relaxed. Investigate coupled CUDA
repeatability in the engine/runtime before promoting the CUDA image as a
qualified release. Evidence is retained locally under `runs/docker-cuda-final/`,
including `repeatability.json`; generated run artifacts are not committed.
The full lifecycle and exact transport evidence is under
`runs/docker-cuda-qualified-checks/`. Its `qualification.json` deliberately
retains `status: failed` because the numerical gate remains unsatisfied.
