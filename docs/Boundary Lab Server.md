# Boundary Lab physical-system server preview

`blab server` now provides an experimental **localhost-only CPU service** using
Boundary Lab's compiled physical-system contract and the installed BEAT Engine.
The legacy source-model HTTP API and Bempp runtime remain retired.

## Run a CPU job

Use the Boundary Lab Python environment with Julia installed and the engine CPU
environment prepared:

```bash
python -m beat_engine instantiate --backend cpu
python -m blab.cli server --root runs/server-jobs --port 8765 --julia-threads 2
```

In another terminal, validate a physical-system project, then submit it:

```bash
python -m blab.cli project validate examples/Simple_Sealed/simple_sealed.blab.json --backend beat_cpu --json
python -m blab.cli project solve examples/Simple_Sealed/simple_sealed.blab.json --backend beat_cpu --server-url http://127.0.0.1:8765 --output runs/server-result
```

Use a request overlay with `--request request.json` to choose a small frequency
set for initial checks. The default uses the project's full frequency range.
The client prepares and validates the same physical system used locally, uploads
its mesh snapshots, and writes the usual headless result directory. Complex
per-excitation values, result domains, engine provenance, and worker provenance
are preserved. `manifest.json` also records `remote_job_id`; `request.json` records
that remote observation planes were excluded.

Remote preparation omits observation planes while preserving polar/balloon
sampling preferences. Explicit point probes and retained BEM/FEM quantities remain
available subject to existing solve-kind restrictions. No remote retained-field
resampling or observation-plane viewer integration is provided.

## First milestone contract

The upload is a ZIP archive containing `job.json` and `assets/<sha256>.msh` files.
The envelope identifies `protocol: blab-remote`, integer `version: 1`,
`backend_id: beat_cpu`, `observation_planes: false`, the existing engine-defined
solve request, and an asset manifest with sizes and SHA-256 hashes. This service
version is independent of the engine request/result versions.

Mesh file references in the submitted compiled system are manifest asset names.
The server checks versions, hashes, archive membership, upload/expanded size
limits (256 MiB), and entry count (1024), then writes assets into a new job
UUID directory. It does not extract arbitrary archive paths. Every mesh must
reference a bundled asset; unused assets and client runtime-path overrides are
rejected. The typed request is validated again before worker execution. Numerical
options are explicitly allowlisted for this preview.

| Endpoint | Behavior |
| --- | --- |
| `GET /v1/capabilities` | Service and engine wire versions, CPU configuration, upload limit, and excluded planes |
| `POST /v1/jobs` | Submit ZIP bytes; returns HTTP 202 and `job_id` after staging/validation |
| `GET /v1/jobs/<id>?cursor=0` | Read up to 32 retained events and the next cursor |
| `DELETE /v1/jobs/<id>` | Request cooperative cancellation; completed results remain readable |

Events include `accepted`, `status`, engine-contract `result` payloads, and a
terminal `completed`, `failed`, or `cancelled` event. Completed/failed jobs retain
worker provenance when available. Capability discovery describes service support;
the BEAT worker performs runtime availability and compatibility checks when a job
executes. Discovery alone does not guarantee a working Julia installation.

One job executes at a time; another submission receives HTTP 409. Jobs execute
independently of their submitting HTTP connection. Clients can replay retained
events by job ID and cursor; the CLI does not yet resume an interrupted result
download automatically. On service restart, interrupted jobs are marked failed,
complete records remain readable, and an incomplete final event is discarded.

## Preview limits and qualification

The service and client restrict connections to localhost. Authentication, LAN
exposure, GPU discovery/selection, GUI integration, automatic retention cleanup,
and queuing are later milestones. Use one service process per dedicated job root.
Job assets and event journals remain on disk until the operator removes them
while the service is stopped. This is a development preview, not a public HTTP
service deployment.

Run the real CPU parity check from the repository root:

```bash
python scripts/check_remote_integration.py --output runs/remote-integration
```

It validates the Simple Sealed project, runs one 500 Hz coupled FEM-BEM-LEM solve
locally and through a separate HTTP server process, and checks exact array equality,
quantity metadata, excitation ordering, and engine/worker provenance. It writes
`comparison.json` and both ordinary result directories; choose a new output path
for each run. Broader exterior/interior and accelerator qualification remains
part of subsequent milestones.
