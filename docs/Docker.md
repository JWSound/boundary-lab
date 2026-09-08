# Boundary Lab Server with Docker

These instructions build and run the current physical-model server locally.
These are local builds; a new Docker Hub release and Runpod template have not
been published yet. Older server images use the retired protocol.

CPU local qualification passes. The CUDA image remains experimental: coupled
solves complete, but repeated solves have exceeded the strict numerical
comparison tolerance for retained BEM boundary pressure. See the
[qualification findings](advanced/container-development.md#qualification-findings)
before using it for production work.

CPU local qualification passes. The CUDA image remains experimental: coupled
solves complete, but repeated solves have exceeded the strict numerical
comparison tolerance for retained BEM boundary pressure. See the
[qualification findings](advanced/container-development.md#qualification-findings)
before using it for production work.

## Build

Use Docker with Linux containers. From a Boundary Lab checkout:

```bash
docker build --target cpu -t boundarylabserver:local-cpu .
docker build --target cuda -t boundarylabserver:local-cuda .
```

The first build downloads and prepares Python and Julia dependencies and can
take several minutes. The CUDA image includes CPU support. GPU access is needed
when running CUDA solves, but not when building the image.

## Private LAN

```bash
docker volume create blab-jobs
docker run -d --name blab-server --restart unless-stopped -p 8765:8765 -v blab-jobs:/data boundarylabserver:local-cpu --mode private-network
```

For NVIDIA solving, use `boundarylabserver:local-cuda` and add `--gpus all`
before the image name. The host needs a compatible NVIDIA driver and Docker GPU
support (Docker Desktop WSL2 on Windows, NVIDIA Container Toolkit on Linux).

In Preferences, choose **Boundary Lab Server** and enter
`http://<server-LAN-address>:8765`. This example needs no key; anyone who can
reach it can use the server. For same-machine testing, publish
`127.0.0.1:8765:8765` instead. The server automatically selects its backend.

## Access key and hosted mode

Generate a key in Boundary Lab Preferences, copy it, and save it locally.
For example, in PowerShell:

```powershell
$env:BLAB_SERVER_TOKEN = 'replace-with-your-generated-key'
docker run -d --name blab-server --gpus all --restart unless-stopped -p 127.0.0.1:8765:8765 -v blab-jobs:/data -e BLAB_SERVER_TOKEN boundarylabserver:local-cuda
```

Hosted mode is the default and refuses startup without a key. This example
binds to localhost for testing or SSH forwarding. On a hosted provider, place
container HTTP port `8765` behind its HTTPS proxy or a VPN. The image does not
manage certificates or provide HTTPS. See
[Boundary Lab Server](Boundary%20Lab%20Server.md) for connection instructions.

## Storage and operation

The named volume stores jobs at `/data/jobs` and survives container replacement.
Keep the volume when updating. Each server needs its own job directory. Jobs
require manual cleanup; there is no automatic retention policy or job queue.

```bash
docker logs blab-server
docker inspect --format '{{.State.Health.Status}}' blab-server
docker stop --timeout 30 blab-server
```

Wait for startup before solving. Stopping the container cancels active work;
it does not pause or resume a solve.

| Setting | Default | Purpose |
| --- | --- | --- |
| `BLAB_SERVER_TOKEN` | Empty | Shared key; required in hosted mode |
| `BLAB_SERVER_MODE` | `hosted` | `private-network` allows optional authentication |
| `BLAB_SERVER_ROOT` | `/data/jobs` | Persistent job directory |
| `BLAB_SERVER_PORT` | `8765` | Internal HTTP port; adjust published port too |

Normal server arguments can follow the image name, such as `--backend cpu` or
`--root /workspace/blab/jobs`. Bind mounts must be writable by UID `10001`;
the named volume above is initialized automatically.

Use `--julia-threads 2` after the image name to set solver worker threads.
Without that argument, the application defaults to 8 CPU threads and 4 CUDA
threads. These explicit worker settings take precedence over `JULIA_NUM_THREADS`.

See [Container development](advanced/container-development.md) for build and
qualification details.
