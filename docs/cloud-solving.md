# Cloud Solving Prototype

Boundary Lab's cloud solving path is built around a stable solve-event contract:
the desktop client submits a solve job, the server runs the normal BEM solver,
and the server streams initialization, status, per-frequency results, and final
completion events back to the client.

The first prototype is intentionally local and single-process. It proves the
API shape before the backing runner moves to Amazon Fargate, SQS, and S3.

## Local Prototype

Install the cloud API extras:

```bash
python -m pip install -e ".[cloud]"
```

Run the API:

```bash
blab cloud-api --host 127.0.0.1 --port 8080
```

With the API running, launch the GUI and use `Cloud Solve` instead of `Solve`.
The GUI uses the same mesh/source/frequency setup as a local solve, creates a
temporary solve bundle, posts it to `/v1/solve-jobs/bundle`, then connects to
the returned WebSocket stream. The plots update through the same live result
path used by local solving.

The cloud API URL is configurable in `Edit > Preferences`. For the local
prototype, leave it at:

```text
http://127.0.0.1:8080
```

The prototype exposes:

```text
GET  /healthz
POST /v1/solve-jobs
POST /v1/solve-jobs/bundle
GET  /v1/solve-jobs/{job_id}
POST /v1/solve-jobs/{job_id}/cancel
WS   /v1/solve-jobs/{job_id}/stream
```

`POST /v1/solve-jobs` currently expects the server to have filesystem access to
the mesh paths in the submitted `SimulationConfig`. That is useful for local
testing.

`POST /v1/solve-jobs/bundle` is the preferred prototype path. It accepts a
portable zip bundle containing:

```text
manifest.json
inputs/*.msh
```

The manifest carries the simulation config and frequency order. Mesh paths are
rewritten to bundle-relative `inputs/...` paths, then rewritten again to
server-local workspace paths after upload. This is the shape that maps cleanly
to S3 and Fargate later.

Python code can build a bundle with:

```python
import numpy as np

from blab.cloud.bundle import write_solve_bundle

write_solve_bundle(
    "solve.blabsolve.zip",
    config=config,
    frequencies=np.array([200.0, 1000.0], dtype=np.float32),
)
```

Then submit it as raw zip bytes:

```bash
curl -X POST \
  -H "Content-Type: application/zip" \
  --data-binary @solve.blabsolve.zip \
  http://127.0.0.1:8080/v1/solve-jobs/bundle
```

## Event Types

Events are JSON objects with a `type` and `job_id`.

```text
status
initialized
frequency_result
completed
failed
```

Large numeric arrays are represented with:

```json
{
  "dtype": "float32",
  "shape": [3],
  "data": [-6.0, 0.0, -6.0]
}
```

This is deliberately simple for the prototype. For production, large arrays
should move to S3 objects or chunked formats such as Zarr, with WebSocket events
carrying only summaries and signed result URLs.

## Fargate Target Shape

The production architecture should split job admission from job execution:

```text
GUI
  -> HTTPS API: create job and request upload URLs
  -> S3: upload mesh/config payloads
  -> HTTPS API: start job
  -> WebSocket API: receive progress/results

API
  -> Postgres/DynamoDB: job metadata
  -> SQS: queued solve requests
  -> ECS/Fargate: solver task per job or per frequency batch
  -> S3: raw inputs, per-frequency result payloads, final archive
```

Recommended AWS pieces:

```text
ECS Fargate task for solver workers
Application Load Balancer or API Gateway for HTTPS API
API Gateway WebSocket, AppSync, or a small FastAPI realtime service
SQS for job dispatch
S3 for uploaded meshes and result artifacts
Postgres/RDS or DynamoDB for job metadata
CloudWatch logs and metrics
```

Start with one Fargate task per solve job. It is simpler and matches the local
solver's current lifecycle. Later, split a job into frequency batches if
parallel cloud execution is worth the added aggregation complexity.

## Package Boundary

The prototype keeps shared wire-format helpers in `blab.cloud.protocol`. The
GUI, local API, and future cloud workers should all use that module so local and
cloud solves emit compatible result events.

The current cloud API lives in `blab.cloud.api`. It should stay thin: API,
queueing, and streaming glue belong there; solver behavior should remain in the
existing solver/live modules or in a future extracted `boundary-lab-solver`
package.
