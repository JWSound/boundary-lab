# CUDA Server Docker Image

Boundary Lab's BEAT Engine CUDA solve server can be packaged as a GPU container. Future support for AMD ROCm Docker images is planned.

## Build

From the repository root:

```bash
docker build -f docker/server-cuda.Dockerfile -t boundary-lab-server:cuda .
```

The build installs Python dependencies, Julia, the `src/blab/solvers/julia_cuda` project, and a CUDA-focused Julia sysimage at `/app/blab-beat-cuda.so`. To skip the sysimage while keeping Julia precompilation, add `--build-arg BLAB_BUILD_SYSIMAGE=0`.

## Run Locally On A GPU Host

```bash
docker run --rm --gpus all \
  -p 8765:8765 \
  -v blab-server-data:/data \
  boundary-lab-server:cuda
```

Check the server:

```bash
curl http://127.0.0.1:8765/health
```

The response should report `"solver":"beat_cuda"`.

The container listens on port `8765` by default. Authentication is optional;
without `BLAB_AUTH_TOKEN`, only expose it on localhost or a trusted private
network.

## Runpod HTTPS Deployment

For an internet-reachable Runpod Pod:

1. In Boundary Lab Preferences, set `BEM Solver` to `Server`.
2. Click `Generate` beside `Server access token`, then click `Copy`.
3. Create a Runpod Secret such as `boundary_lab_access_token` using the copied
   token as its value.
4. In the Pod template, set:

   ```text
   BLAB_AUTH_TOKEN={{ RUNPOD_SECRET_boundary_lab_access_token }}
   ```

5. Expose `8765` as an HTTP port, not a raw TCP port.
6. Set Boundary Lab's server URL to:

   ```text
   https://<pod-id>-8765.proxy.runpod.net
   ```

7. Click `Check Server`, then accept Preferences.

Runpod terminates HTTPS at its HTTP proxy. The bearer token authenticates
requests at the Boundary Lab server. Do not put the token directly in the
Docker image, Pod template text, server URL, or command line.

## Configuration

The entrypoint reads these environment variables:

| Variable | Default |
|---|---|
| `BLAB_SERVER_HOST` | `0.0.0.0` |
| `BLAB_SERVER_PORT` | `8765` |
| `BLAB_SERVER_SOLVER` | `beat_cuda` |
| `BLAB_JULIA_EXECUTABLE` | `/opt/juliaup/bin/julia` |
| `BLAB_JULIA_THREADS` | `auto` |
| `BLAB_JULIA_SYSIMAGE` | `/app/blab-beat-cuda.so` |
| `BLAB_JULIA_CPU_TARGET` | `generic,+aes` |
| `BLAB_WARM_SOLVER` | `off` |
| `BLAB_MAX_RUNNING_JOBS` | `1` |
| `BLAB_MAX_QUEUED_JOBS` | `4` |
| `BLAB_MAX_REQUEST_MB` | `20` |
| `BLAB_MAX_ASSET_MB` | `14` |
| `BLAB_MAX_FREQUENCIES` | `1000` |
| `BLAB_JOB_RETENTION_HOURS` | `24` |
| `BLAB_EVENT_STREAM_WINDOW_SECONDS` | `25` |
| `BLAB_AUTH_TOKEN` | empty; authentication disabled |
| `BLAB_LOG_LEVEL` | `INFO` |
| `BLAB_ARTIFACT_DIR` | `/data/server_jobs` |

Example override:

```bash
docker run --rm --gpus all \
  -e BLAB_JULIA_THREADS=8 \
  -e BLAB_WARM_SOLVER=tiny \
  -e BLAB_ARTIFACT_DIR=/data/jobs \
  -p 8765:8765 \
  -v blab-server-data:/data \
  boundary-lab-server:cuda
```

`BLAB_WARM_SOLVER=worker` starts the persistent Julia worker during server startup. `BLAB_WARM_SOLVER=tiny` also runs a one-frequency tetrahedron solve, which is slower to start but warms more CUDA/JIT paths before the first client job. The sysimage is built with `BLAB_JULIA_CPU_TARGET=generic,+aes`, which avoids host-specific targets while keeping AES-NI available for dependencies that emit AES intrinsics. Set `BLAB_JULIA_SYSIMAGE=` to disable the bundled sysimage for diagnostics.

The 20 MiB request limit accommodates typical Boundary Lab meshes while
bounding memory use before JSON parsing. Base64 expands uploaded files, so the
decoded asset limit defaults to 14 MiB. Event responses rotate every 25 seconds
and the GUI resumes from the last event, avoiding long-lived proxy connection
limits. Terminal jobs and artifacts expire after 24 hours unless retention is
set to zero.

To run a shell instead of the server:

```bash
docker run --rm -it --gpus all boundary-lab-server:cuda bash
```
