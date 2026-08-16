# Boundary Lab Server

Boundary Lab can run a local or LAN-accessible solve server. The GUI submits a complete solve request over HTTP, uploads the required mesh assets with the job, and streams per-frequency results back as newline-delimited JSON events.

## Starting The Server

Run the server with an explicit solver selector:

```bash
blab server --host 127.0.0.1 --port 8765 --solver bempp_cpu
blab server --host 127.0.0.1 --port 8765 --solver beat_cpu --julia-threads auto
blab server --host 127.0.0.1 --port 8765 --solver beat_cuda --julia-threads auto
```

For LAN use, bind to the machine's LAN address or to all interfaces:

```bash
blab server --host 0.0.0.0 --port 8765 --solver beat_cuda
```

Then use `http://<server-ip>:8765` as the GUI's Solve Server URL.

## Optional Authentication

Authentication is opt-in. If `BLAB_AUTH_TOKEN` is unset or empty, the server
continues to accept unauthenticated requests for localhost and trusted private
network use.

To protect a server, set a high-entropy token in the environment before
starting it:

```bash
export BLAB_AUTH_TOKEN="paste-generated-token-here"
blab server --host 0.0.0.0 --port 8765 --solver beat_cuda
```

When a token is configured, every API endpoint, including `/health`, requires
an `Authorization: Bearer ...` header. The server never logs the token.

In Boundary Lab, select `BEM Solver: Server` in Preferences, click `Generate`
beside `Server access token`, and use `Copy` to place the generated token in
your deployment's secret store. Keep a safe copy of the token: Boundary Lab
retains it only for the current application session and does not store it in
the operating system credential vault or application settings.

Bearer tokens must be used with HTTPS except for loopback addresses. Boundary
Lab refuses to send a configured token over remote plain HTTP. An
unauthenticated server can still use plain HTTP on a trusted private network.

## Solver Selectors

Supported `--solver` values:

- `bempp_cpu`: Bempp OpenCL CPU backend.
- `beat_cpu`: BEAT Engine CPU backend through Julia.
- `beat_cuda`: BEAT Engine CUDA backend through Julia.
- `beat_rocm`: BEAT Engine ROCm non-symmetric exterior-BEM backend with GPU-resident operator assembly and solve.

BEAT Engine server options are intentionally narrow:

- `--julia-executable`: Julia executable path. Defaults to `julia`.
- `--julia-threads`: Julia thread count. Defaults to `auto`.

The server defaults to `--solver bempp_cpu` when no solver is specified.

## GUI Workflow

In the Boundary Lab GUI:

1. Open `Edit > Preferences`.
2. Set `BEM Solver` to `Server`.
3. Set `Solve Server URL` to the server address, such as `http://127.0.0.1:8765`.
4. Click `Check Server`.
5. Confirm the server info dialog, then accept Preferences.

`Check Server` calls `GET /health` with the configured access token and updates the application's view of server-advertised capabilities. This matters for features such as BEAT Engine server-side X/XY symmetry. If Boundary Lab starts with `BEM Solver` already set to `Server`, it also runs a silent startup `GET /health` probe with a 5 second timeout. Failed startup probes do not interrupt application launch; the GUI simply falls back to the conservative unavailable state until a later successful check.

## Symmetry Support

Server-side symmetry depends on the configured server solver:

- `beat_cpu` and `beat_cuda` advertise symmetry support and can solve `off`, `x`, and `xy` symmetry requests.
- `bempp_cpu` does not support symmetry acceleration.
- `beat_rocm` advertises the BEAT Engine shape but is not numerically implemented yet.

The GUI uses the checked server health payload to decide whether the symmetry
control in the **Meshes** window is enabled while the selected BEM Solver is
`Server`.

For symmetry solves, the GUI still prepares and uploads the reduced-domain mesh files. The server does not need access to the client's original local paths.

## Job API

The server exposes a small HTTP API:

- `GET /health`: returns status, configured solver, backing backend ID, and capability flags.
- `POST /jobs`: submits a solve request with `SimulationConfig`, `frequencies_hz`, and optional uploaded assets.
- `GET /jobs/{job_id}`: returns job status and artifact links.
- `GET /jobs/{job_id}/events?since=0`: streams job events as newline-delimited JSON.
- `POST /jobs/{job_id}/cancel`: requests cancellation.
- `GET /jobs/{job_id}/artifacts/result.npz`: downloads the completed result bundle.

Typical event flow:

```text
queued
started
initialized
result
result
...
completed
```

Failures are emitted as `failed` events with an error message. Cancellation is cooperative and may wait for the current in-flight frequency solve to finish.

Each event-stream response is intentionally limited to 25 seconds by default.
The client reconnects with the last received event index, so long solves remain
compatible with HTTPS proxies that cap connection duration without duplicating
results.

## Artifacts

Completed jobs write a compressed `result.npz` under the configured artifact directory. By default this is:

```text
runs/server_jobs
```

Override it with:

```bash
blab server --artifact-dir runs/my_server_jobs --solver beat_cpu
```

Terminal job state and artifacts are retained for 24 hours by default. Expired
jobs are purged during subsequent health checks or job submissions. Set
`--job-retention-hours 0` to retain them indefinitely.

## Resource Limits

Servers apply these defaults:

| Limit | Default | Option |
|---|---:|---|
| JSON request body | 20 MiB | `--max-request-mb` |
| Decoded uploaded assets per job | 14 MiB | `--max-asset-mb` |
| Frequencies per job | 1000 | `--max-frequencies` |
| Waiting jobs | 4 | `--max-queued-jobs` |
| Terminal job retention | 24 hours | `--job-retention-hours` |
| Event response window | 25 seconds | `--event-stream-window-seconds` |

The 14 MiB decoded-asset limit leaves space for base64 expansion and request
metadata inside the 20 MiB request ceiling. These limits are configurable for
unusually large meshes.

## Operational Notes

- Keep `--max-running-jobs 1` unless you have intentionally tested concurrent jobs for the selected backend and hardware.
- A server listening on a non-loopback address without `BLAB_AUTH_TOKEN` logs a warning but continues to start.
- BEAT Engine CUDA jobs should usually be run one at a time per GPU.
- Use `--log-level INFO` for normal pod logs, or `--log-level DEBUG` when diagnosing request and job flow.
- The GUI uploads mesh assets with every server job, so the server can run on another machine without shared filesystem paths.
- If a BEAT Engine server fails during startup, check that the matching Julia environment has been instantiated.

Install examples:

```bash
julia --project=src/blab/solvers/julia_local -e "using Pkg; Pkg.instantiate()"
julia --project=src/blab/solvers/julia_cuda -e "using Pkg; Pkg.instantiate()"
```
