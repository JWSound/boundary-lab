# Boundary Lab physical-system server preview

`blab server` provides a **CPU/CUDA/ROCm service with optional authenticated HTTPS for LAN use**, using
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
python -m blab.cli project solve examples/Simple_Sealed/simple_sealed.blab.json --server-url http://127.0.0.1:8765 --output runs/server-result
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

## Application preferences

In **Preferences → Application**, set **Solver** to **Boundary Lab Server**.
The Server fields contain the address,
token environment-variable name, and optional trusted PEM certificate path.
The actual token is never stored in preferences. Set its environment variable
before launching Boundary Lab; restart the app after changing that environment.

**Check connection** queries capabilities in the background and displays runtime
readiness or the connection error. If the server is still starting, check again
after startup completes. Changing connection fields invalidates the displayed
status. Local solver choices disable the server fields while preserving their values.

Solves use the ordinary GUI start/stop controls and live plots, preserving complex
per-excitation results. Server negotiation runs on the solve worker thread.
Remote observation planes remain unavailable; polar and balloon sampling remain
enabled according to the Observation Config preferences.

## Authenticated LAN setup

The default binds to `127.0.0.1`. A LAN binding requires a shared token and TLS.
Every endpoint requires authentication when a token is configured. This is a
single-operator service: token holders can read and cancel all jobs.

Set `BLAB_SERVER_TOKEN` in both server and client environments to the same random
secret of at least 32 ASCII characters without whitespace. Generate a value with
`python -c "import secrets; print(secrets.token_urlsafe(32))"` and distribute it
through your normal secret-management channel. Tokens are read from environment
variables, not CLI argument values, and are not saved in result artifacts.
`--token-env` (server) and `--server-token-env` (client) select an alternate variable.

Use a certificate whose Subject Alternative Name matches the server hostname/IP.
Keep its private key on the server:

```bash
python -m blab.cli server --root runs/server-jobs --host 0.0.0.0 --port 8765 --tls-cert server-cert.pem --tls-key server-key.pem
```

After setting the token in the client environment:

```bash
python -m blab.cli project solve speaker.blab.json --server-url https://solver.example:8765 --server-ca lab-ca.pem --output runs/server-result
```

Replace `solver.example` with the actual certificate hostname. Omit `--server-ca`
when Python's default trust store already trusts the certificate. Trust and
hostname verification remain enabled; redirects are rejected to prevent credential
forwarding. Plain HTTP is supported only for localhost connections. No firewall
changes are made automatically. Limit port access to intended LAN clients;
Internet hosting and multi-user isolation remain outside this service's scope.

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

## Preview limits and qualification

Automatic retention cleanup,
and queuing are later milestones. Use one service process per dedicated job root.
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
for each run. The script uses an ephemeral authentication token. Add `--tls-cert`
and `--tls-key` for HTTPS with a certificate valid for `127.0.0.1`; the script
trusts that certificate only for its test client. Other solve-kind checks:

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
