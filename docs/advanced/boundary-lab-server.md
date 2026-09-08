# Boundary Lab server developer reference

For server setup, access keys, and connecting from the application, see
[Boundary Lab Server](../Boundary%20Lab%20Server.md). This reference covers the
implementation, remote contract, job lifecycle, and qualification workflow.

For image construction and offline container qualification, see
[Container development](container-development.md).

## Application and engine boundaries

Boundary Lab owns model preparation, HTTP transport, asset transfer, and job
management. Numerical execution stays in the installed BEAT Engine dependency.
The retired source-model HTTP API and Bempp runtime are not compatibility paths.

GUI and CLI clients prepare the same compiled physical-system request used
locally. GUI connection checks and solve negotiation run off the UI thread;
remote sessions feed complex per-excitation results into the existing live plots.
Observation planes are omitted during preparation and disabled for remote GUI
results. Polar/balloon sampling remains available. Explicit probes and retained
BEM/FEM quantities retain the existing solve-kind restrictions; remote retained-field
resampling is not exposed.

## Authentication and transport boundaries

Private-network mode permits requests without a key unless one is configured.
Hosted mode requires a key before startup. A configured key protects every
endpoint, including discovery and result retrieval. Requests carry an
`Authorization: Bearer <key>` header, checked with `hmac.compare_digest`. Keys must
contain at least 32 ASCII characters without whitespace; the GUI generates them
with `secrets.token_urlsafe(32)`.

The application serves HTTP. A provider HTTPS proxy, VPN, or SSH tunnel supplies
transport protection for hosted deployments. Hosted mode does not create
encryption or verify that upstream arrangement. The client uses normal HTTPS
trust and hostname verification and rejects redirects to prevent credential
forwarding. No custom certificate configuration or verification bypass is exposed.

The GUI holds its key in memory for the current session. It is excluded from
persisted settings, diagnostics, and preference/prepared-request representations.
Connection options remain application metadata and do not enter the numerical
wire payload. The server reads `BLAB_SERVER_TOKEN` or the variable selected by
`--token-env`; the CLI uses that default or `--server-token-env`. The GUI reads
its access-key field directly. There are no individual accounts or job ownership
checks: authorized clients share access to all jobs.

## Client result artifacts

The headless client writes ordinary result directories, preserving complex values,
excitation ordering, result domains, engine provenance, and worker provenance.
`manifest.json` records `remote_job_id` and the actual backend; `request.json`
records the exclusion of remote observation planes. See the
[headless project workflow](cli-workflow.md#headless-project-workflow) for artifact use.

## Backend discovery and selection

The server probes its installed CPU, CUDA, and ROCm worker environments in the
background at startup. `/v1/capabilities` reports each backend's `state` (`checking`
or `ready`), `available` flag, and failure reason, plus engine/runtime information
when startup succeeds. These checks perform no matrix assembly. Discovery has a
120-second timeout per runtime; restart the server after installing or repairing
an environment to refresh the snapshot.

Prepare accelerator environments on the server with
`python -m beat_engine instantiate --backend cuda` or `--backend rocm`, alongside
the required drivers/SDK. Missing runtimes remain unavailable with a reason; they
do not prevent use of a working CPU environment.

The client submits models without choosing a backend. The server's default
`--backend auto` policy prefers available CUDA, then ROCm, then CPU. Operators may
pin execution using `blab server --backend cpu`, `--backend cuda`, or
`--backend rocm`; a pinned unavailable runtime fails rather than silently falling
back. Pure interior FEM always uses the server CPU runtime.

Preferences shows only connection readiness, not hardware capabilities. The
client waits for server startup, and the server chooses a runtime for each model
before staging assets. Actual workers still negotiate compatibility for every
job. Backend availability does not guarantee that a model fits device memory.
Result artifacts retain the actual execution backend and worker provenance.
The CLI's `--backend` option is for local solves; omit it with `--server-url`.

## Remote job contract

The upload is a ZIP archive containing `job.json` and `assets/<sha256>.msh` files.
The envelope identifies `protocol: blab-remote`, integer `version: 2`,
no client backend selector,
`observation_planes: false`, the existing engine-defined
solve request, and an asset manifest with sizes and SHA-256 hashes. This service
version is independent of the engine request/result versions. Older clients
are rejected during negotiation; update client and server together.

Mesh file references in the submitted compiled system are manifest asset names.
The server checks versions, hashes, archive membership, upload/expanded size
limits (256 MiB), and entry count (1024), then writes assets into a new job
UUID directory. It does not extract arbitrary archive paths. Every mesh must
reference a bundled asset; unused assets and client runtime-path overrides are
rejected. The typed request is validated again before worker execution. Numerical
options are explicitly allowlisted for this preview.

| Endpoint | Behavior |
| --- | --- |
| `GET /v1/capabilities` | Wire versions, per-backend runtime status, upload limit, and excluded planes |
| `POST /v1/jobs` | Submit ZIP bytes; returns HTTP 202 and `job_id` after staging/validation |
| `GET /v1/jobs/<id>?cursor=0` | Read up to 32 retained events and the next cursor |
| `DELETE /v1/jobs/<id>` | Request cooperative cancellation; completed results remain readable |

Events include `accepted`, `status`, engine-contract `result` payloads, and a
terminal `completed`, `failed`, or `cancelled` event. Completed/failed jobs retain
worker provenance when available. Capability discovery checks runtime startup;
the solve worker checks availability and compatibility again when a job executes.
Discovery alone does not qualify numerical accuracy or available device memory.

One job executes at a time; another submission receives HTTP 409. Jobs execute
independently of their submitting HTTP connection. Clients can replay retained
events by job ID and cursor; the CLI does not yet resume an interrupted result
download automatically. On service restart, interrupted jobs are marked failed,
complete records remain readable, and an incomplete final event is discarded.

## Retention and qualification

Automatic retention cleanup and queuing are not implemented. Use one service process per dedicated job root.
Job assets and event journals remain on disk until the operator removes them
while the service is stopped. This is a development preview, not a public HTTP
service deployment.

Run the real CPU parity check from the repository root:

```bash
python scripts/check_remote_integration.py --output runs/remote-integration
```

It validates the Simple Sealed project, runs one 500 Hz coupled FEM-BEM-LEM solve
locally and through a separate HTTP server process, and checks CPU array equality,
quantity metadata, excitation ordering, and engine/worker provenance. It writes
`comparison.json` and both ordinary result directories; choose a new output path
for each run. The script configures hosted mode with an ephemeral authentication
key and uses local HTTP. Other solve-kind checks:

```bash
python scripts/check_remote_integration.py --project tests/fixtures/remote-exterior.blab.json --frequency 500 --output runs/remote-exterior
python scripts/check_remote_integration.py --project examples/compression_driver/compression_driver.blab.json --frequency 1000 --output runs/remote-interior
```

The exterior fixture derives from BEAT's example request and uses the existing
test mesh. Interior qualification retains nodal pressure without exterior probes.
These checks establish transport parity, not a broad numerical accuracy benchmark.
The qualification script accepts `--backend beat_cuda` or `--backend beat_rocm`
to configure both its local reference and the server policy; the remote client
still sends no backend choice.
GPU and actual cross-machine deployment qualification require the corresponding
hardware and network; mocked routing tests alone do not qualify numerical execution.

GPU comparison allows a maximum absolute array error divided by the local array's
maximum absolute value of `1e-5`, to accommodate FP32 execution differences. CPU
comparison remains exact. The report includes per-array measured errors and the
number of arrays that were exactly equal. Use `--compare-only --output RUN_DIR`
to recheck existing artifacts without rerunning the solver.

## Implementation entry points

| File | Responsibility |
| --- | --- |
| [server.py](../../src/blab/server.py) | HTTP endpoints, deployment modes, server policy, and job lifecycle |
| [server_capabilities.py](../../src/blab/server_capabilities.py) | Runtime discovery |
| [remote_contract.py](../../src/blab/remote_contract.py) | Envelope validation and portable asset staging |
| [remote.py](../../src/blab/remote.py) | Qt-free client and solve sessions |
| [server_preferences.py](../../src/blab/ui/server_preferences.py) | Access-key controls and connection checks |
| [system_solve.py](../../src/blab/ui/system_solve.py) | GUI worker and live-result integration |

Focused regression checks:

```bash
python -m pytest tests/test_remote.py tests/test_server_capabilities.py tests/test_remote_tls.py tests/test_gui_server.py
python -m ruff check src tests
```

TLS tests generate an ephemeral certificate with OpenSSL for a test-only HTTPS
front end. Certificate-dependent checks skip when OpenSSL is absent. Do not commit
generated certificates or solve artifacts.
