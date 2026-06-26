# CUDA Server Docker Image

Boundary Lab's BEAT Engine CUDA solve server can be packaged as a GPU container. Future support for CPU and AMD ROCm Docker images is planned.

## Build

From the repository root:

```bash
docker build -f docker/server-cuda.Dockerfile -t boundary-lab-server:cuda .
```

The build installs Python dependencies, Julia, and the `src/blab/solvers/julia_cuda` project.

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

The container listens on port `8765` by default. Because the server API is intentionally small and currently unauthenticated, prefer an SSH tunnel or private network instead of exposing `8765` publicly:

## Configuration

The entrypoint reads these environment variables:

| Variable | Default |
|---|---|
| `BLAB_SERVER_HOST` | `0.0.0.0` |
| `BLAB_SERVER_PORT` | `8765` |
| `BLAB_SERVER_SOLVER` | `beat_cuda` |
| `BLAB_JULIA_EXECUTABLE` | `/opt/juliaup/bin/julia` |
| `BLAB_JULIA_THREADS` | `auto` |
| `BLAB_MAX_RUNNING_JOBS` | `1` |
| `BLAB_LOG_LEVEL` | `INFO` |
| `BLAB_ARTIFACT_DIR` | `/data/server_jobs` |

Example override:

```bash
docker run --rm --gpus all \
  -e BLAB_JULIA_THREADS=8 \
  -e BLAB_ARTIFACT_DIR=/data/jobs \
  -p 8765:8765 \
  -v blab-server-data:/data \
  boundary-lab-server:cuda
```

To run a shell instead of the server:

```bash
docker run --rm -it --gpus all boundary-lab-server:cuda bash
```
