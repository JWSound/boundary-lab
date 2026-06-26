# CUDA Server Docker Image

Boundary Lab's BEAT Engine CUDA solve server can be packaged as a GPU container. Future support for CPU and AMD ROCm Docker images is planned.

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
| `BLAB_JULIA_SYSIMAGE` | `/app/blab-beat-cuda.so` |
| `BLAB_JULIA_CPU_TARGET` | `generic,+aes` |
| `BLAB_WARM_SOLVER` | `off` |
| `BLAB_MAX_RUNNING_JOBS` | `1` |
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

To run a shell instead of the server:

```bash
docker run --rm -it --gpus all boundary-lab-server:cuda bash
```
