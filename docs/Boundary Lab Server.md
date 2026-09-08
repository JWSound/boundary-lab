# Boundary Lab physical-system server preview

`blab server` provides a **CPU/CUDA/ROCm service for private networks and hosted deployments**, using
Boundary Lab's compiled physical-system contract and the installed BEAT Engine.
The legacy source-model HTTP API and Bempp runtime remain retired.

## Run a CPU job

Use the Boundary Lab Python environment with Julia installed and the engine CPU
environment prepared:

```bash
python -m beat_engine instantiate --backend cpu
python -m blab.cli server --root runs/server-jobs --port 8765 --backend cpu --julia-threads 2
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
Enter the server **Address** and optional **Access key**. Use **Generate** to create
a random key, **Copy** to copy it, and **Show** to reveal the masked value. Save the
key locally in a safe place so you can paste it again next time, as the tooltip
recommends. Boundary Lab keeps it only for the current application session; it
is not written to preferences, projects, result artifacts, diagnostics, or the OS
credential store. The server address is saved normally.

**Check connection** queries capabilities in the background and displays runtime
readiness or the connection error. If the server is still starting, check again
after startup completes. Changing connection fields invalidates the displayed
status. Local solver choices disable the server fields while preserving their values.

Solves use the ordinary GUI start/stop controls and live plots, preserving complex
per-excitation results. Server negotiation runs on the solve worker thread.
Remote observation planes remain unavailable; polar and balloon sampling remain
enabled according to the Observation Config preferences.

## Private LAN / VPN

The server defaults to `private-network` mode and listens on `127.0.0.1`. To accept
connections from your private LAN, explicitly choose the listening address:

```bash
python -m blab.cli server --root runs/server-jobs --mode private-network --host 0.0.0.0 --port 8765
```

Clients enter an address such as `http://192.168.1.20:8765`. An access key is
optional. Setting `BLAB_SERVER_TOKEN` on the server enables key checks on every
endpoint; enter that same key in each client. Without a key, anyone who can reach
the service can submit, read, and cancel jobs. HTTP does not encrypt traffic, so
use this mode only on your trusted private network or through a VPN/SSH tunnel.
No firewall changes are made automatically.

## Hosted server / container

1. In Boundary Lab Preferences, click **Generate**, then **Copy**, and save the key locally.
2. Paste it into the container's `BLAB_SERVER_TOKEN` secret.
3. Start the server in hosted mode behind the provider's HTTPS proxy, a VPN, or an SSH tunnel:

```bash
python -m blab.cli server --root /data/jobs --mode hosted --host 0.0.0.0 --port 8765
```

4. Paste the provider's HTTPS service address into Boundary Lab and click **Check connection**.

Hosted mode refuses startup without a key. Use the provider's HTTP service/proxy
option so it manages public HTTPS while forwarding HTTP to the container. Do not
also expose the container's plain HTTP port directly to the Internet. For SSH,
forward the port and use its localhost HTTP address; a VPN can use a private HTTP
address. Hosted mode itself does not create encryption or verify the upstream
network arrangement; the operator supplies that protected connection.

Boundary Lab serves HTTP and has no certificate files to configure. Its client
supports ordinary HTTPS with normal trust and hostname verification, and rejects
redirects to prevent credential forwarding. There is no custom certificate option
or certificate-verification bypass.

All authorized clients share access to jobs; this is not an account-based service.
Keys must be at least 32 ASCII characters without whitespace. **Generate** produces
a suitable random key. To rotate a key, change the container secret, restart the
server, and paste the new key into clients.

The CLI can read a key from `BLAB_SERVER_TOKEN` (or another variable selected by
`--server-token-env`). Server `--token-env` also selects an alternate variable.
The GUI uses the access-key field directly and does not read these variables.

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
