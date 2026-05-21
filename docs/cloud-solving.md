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

You can also run the solver worker directly from a bundle. This is the command
shape that a future Fargate task should execute:

```bash
blab cloud-worker \
  --job-id job_local_test \
  --bundle solve.blabsolve.zip \
  --events-jsonl runs/cloud/job_local_test/events.jsonl
```

Without `--events-jsonl`, the worker emits newline-delimited JSON events to
stdout. The local FastAPI prototype uses the same runner internally and forwards
those events to WebSocket subscribers through an in-memory sink.

## Container Prototype

The cloud image is Linux/Fargate-oriented and can run either the API or a single
worker command.

Build the image:

```bash
docker build -t boundary-lab-cloud:local .
```

Run the API:

```bash
docker run --rm -p 8080:8080 boundary-lab-cloud:local
```

Or with Compose:

```bash
docker compose -f docker-compose.cloud.yml up --build
```

Run a worker directly from a mounted bundle:

```bash
docker run --rm \
  -v "$PWD:/work" \
  boundary-lab-cloud:local \
  cloud-worker \
  --job-id job_local_test \
  --bundle /work/solve.blabsolve.zip \
  --events-jsonl /work/runs/cloud/job_local_test/events.jsonl
```

Run a worker from an S3-hosted bundle:

```bash
docker run --rm \
  -e AWS_REGION=us-east-1 \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  boundary-lab-cloud:local \
  cloud-worker \
  --job-id job_s3_test \
  --s3-bucket boundary-lab-jobs \
  --s3-key jobs/job_s3_test/input/solve.blabsolve.zip
```

For Fargate, prefer an IAM task role over static access keys. The worker only
needs read access to the input bundle initially; later it will also need write
access for result artifacts and event logs.

The Dockerfile installs a CPU OpenCL path with `pocl-opencl-icd`. That is useful
for Fargate-style CPU tasks, but final performance should be benchmarked against
the target task size.

The prototype exposes:

```text
GET  /healthz
POST /v1/solve-jobs
POST /v1/solve-jobs/upload-target
POST /v1/solve-jobs/bundle
PUT  /v1/solve-jobs/{job_id}/bundle
POST /v1/solve-jobs/{job_id}/start
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

The production-shaped upload flow is:

```text
1. POST /v1/solve-jobs/upload-target
2. PUT the bundle to the returned upload target
3. POST the returned start_path
4. Connect to the returned stream_path
```

For the local development store, the upload target is an API-relative `PUT`
endpoint:

```bash
CREATE_RESPONSE=$(curl -s -X POST http://127.0.0.1:8080/v1/solve-jobs/upload-target)
```

The response shape is:

```json
{
  "job_id": "job_...",
  "status": "created",
  "bundle_key": "jobs/job_.../input/solve.blabsolve.zip",
  "upload": {
    "url": "/v1/solve-jobs/job_.../bundle",
    "method": "PUT",
    "headers": {
      "Content-Type": "application/zip"
    },
    "key": "jobs/job_.../input/solve.blabsolve.zip"
  },
  "start_path": "/v1/solve-jobs/job_.../start",
  "stream_path": "/v1/solve-jobs/job_.../stream"
}
```

For S3, set:

```text
BLAB_BUNDLE_STORE=s3
BLAB_S3_BUCKET=boundary-lab-jobs
BLAB_S3_PREFIX=dev
```

Then `/upload-target` returns a presigned S3 `PUT` URL instead of the local API
upload endpoint. The `start_path` remains the same.

To submit a worker to ECS/Fargate from `start_path`, also set:

```text
BLAB_JOB_LAUNCHER=ecs
BLAB_ECS_CLUSTER=boundary-lab
BLAB_ECS_TASK_DEFINITION=boundary-lab-worker:1
BLAB_ECS_CONTAINER_NAME=cloud
BLAB_ECS_SUBNETS=subnet-aaa,subnet-bbb
BLAB_ECS_SECURITY_GROUPS=sg-aaa
BLAB_ECS_ASSIGN_PUBLIC_IP=DISABLED
```

The API submits a Fargate task with a container command equivalent to:

```bash
blab cloud-worker \
  --job-id job_... \
  --s3-bucket boundary-lab-jobs \
  --s3-key dev/jobs/job_.../input/solve.blabsolve.zip
```

At this stage, ECS launch is only the execution bridge. A submitted Fargate
worker writes events to stdout unless configured otherwise. To stream events
back to the API WebSocket from a separate task, configure a durable event store.

For DynamoDB-backed events, set:

```text
BLAB_EVENT_STORE=dynamodb
BLAB_DYNAMODB_EVENTS_TABLE=boundary-lab-events
```

The API polls the event store while the WebSocket is open. The ECS launcher also
passes these event-store variables into the worker task override, so the worker
and API use the same table.

The DynamoDB events table should use:

```text
partition key: job_id  (string)
sort key:      seq     (number)
```

Each item stores:

```json
{
  "job_id": "job_...",
  "seq": 1770000000000000000,
  "event": {
    "type": "frequency_result",
    "job_id": "job_...",
    "frequency_hz": 1000.0
  }
}
```

For local durable-event testing, use:

```text
BLAB_EVENT_STORE=local
BLAB_LOCAL_EVENT_ROOT=runs/cloud/events
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

A first-pass Terraform prototype lives in:

```text
infra/aws
```

It creates the ECS/Fargate, S3, DynamoDB, IAM, CloudWatch, and load-balancer
resources needed by the current code path. See `infra/aws/README.md` for the
deploy workflow.

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
DynamoDB for event rows
CloudWatch logs and metrics
```

Start with one Fargate task per solve job. It is simpler and matches the local
solver's current lifecycle. Later, split a job into frequency batches if
parallel cloud execution is worth the added aggregation complexity.

The first Fargate task can be thin:

```text
1. download jobs/{job_id}/input/solve.blabsolve.zip from S3
2. run blab cloud-worker --job-id {job_id} --s3-bucket {bucket} --s3-key {key}
3. write emitted events to DynamoDB, S3 JSONL, or a small event relay
```

The storage adapter in `blab.cloud.storage` provides:

```text
LocalBundleStore
S3BundleStore
```

`S3BundleStore` can create presigned `PUT` upload targets for client bundle
uploads and can download bundles for worker execution.

## Package Boundary

The prototype keeps shared wire-format helpers in `blab.cloud.protocol`. The
GUI, local API, and future cloud workers should all use that module so local and
cloud solves emit compatible result events.

The current cloud API lives in `blab.cloud.api`. It should stay thin: API,
queueing, and streaming glue belong there; solver behavior should remain in the
existing solver/live modules or in a future extracted `boundary-lab-solver`
package.

The reusable worker runner lives in `blab.cloud.worker`. Keep cloud execution
behavior there so local threads, Docker runs, and Fargate tasks all exercise the
same solve loop.
