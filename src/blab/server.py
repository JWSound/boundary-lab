"""Physical-system HTTP service with private/hosted access modes and BEAT discovery."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
import zipfile
from collections.abc import Sequence
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from blab.phasor import SOLVER_PHASOR_CONVENTION
from blab.remote_contract import MAX_BUNDLE_BYTES, REMOTE_BACKENDS, REMOTE_VERSION, stage_remote_job
from blab.server_capabilities import BackendDiscovery
from blab.solvers.beat_engine_runtime import shutdown_beat_engine_workers
from blab.solvers.coupled_backend import PhysicalSystemProductionBackend
from blab.solvers.engine_contract import SYSTEM_RESULT_VERSION, SYSTEM_SOLVE_REQUEST_VERSION
from blab.system_contract import system_frequency_result_to_dict

LOGGER = logging.getLogger(__name__)


class SolveService:
    """One active job, disk-backed events, and independent client connections."""

    def __init__(self, root: Path, *, backend=None, backends=None, discover=False, backend_policy="auto"):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend or PhysicalSystemProductionBackend(bem_backend="cpu")
        self.backends = backends or {"beat_cpu": self.backend}
        self.backend_policy = backend_policy
        self.discovery = BackendDiscovery(self.backends) if discover else None
        self.busy = threading.Lock()
        self.lock = threading.RLock()
        self.sessions = {}
        self.threads = []
        for directory in self.root.iterdir():
            if directory.is_dir() and re.fullmatch(r"[0-9a-f]{32}", directory.name):
                journal = directory / "events.ndjson"
                if journal.exists():
                    data = journal.read_bytes()
                    if data and not data.endswith(b"\n"):
                        journal.write_bytes(data[: data.rfind(b"\n") + 1])
                events = self.events(directory.name, 0, limit=None)
                if not events or events[-1]["type"] not in {"completed", "failed", "cancelled"}:
                    self.append(directory.name, {"type": "failed", "error": "Service restarted before completion."})
                    LOGGER.warning("Interrupted solve recovered as failed | job=%s", directory.name)

    def append(self, job_id, event):
        with self.lock, (self.root / job_id / "events.ndjson").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    def capabilities(self):
        return (
            self.discovery.snapshot()
            if self.discovery
            else {key: {"available": True, "state": "ready"} for key in self.backends}
        )

    def readiness(self):
        records = self.capabilities()
        if any(item.get("state") == "checking" for item in records.values()):
            return "starting"
        # CPU can still serve interior FEM when the pinned BEM runtime is unavailable.
        candidates = list(records) if self.backend_policy == "auto" else ["beat_" + self.backend_policy, "beat_cpu"]
        return "ready" if any(records.get(key, {}).get("available") for key in candidates) else "unavailable"

    def select_backend(self, request):
        from blab.physical_model import AcousticRegionKind

        interior = not any(
            region.kind == AcousticRegionKind.UNBOUNDED_AIR for region in request.compiled_system.regions
        )
        candidates = (
            ["beat_cpu"]
            if interior
            else (
                ["beat_cuda", "beat_rocm", "beat_cpu"]
                if self.backend_policy == "auto"
                else ["beat_" + self.backend_policy]
            )
        )
        records = self.capabilities()
        for key in candidates:
            if records.get(key, {}).get("state") == "checking":
                raise ValueError("Server is still starting. Try again shortly.")
            if records.get(key, {}).get("available"):
                return key
        raise ValueError("The server cannot run this model with its configured solver. Contact the server operator.")

    def events(self, job_id, cursor, *, limit=32):
        directory = self.root / job_id
        if not re.fullmatch(r"[0-9a-f]{32}", job_id) or not directory.is_dir():
            raise FileNotFoundError("Unknown job.")
        with self.lock:
            path = directory / "events.ndjson"
            # An abrupt process exit may leave an incomplete final record.
            data = path.read_bytes() if path.exists() else b""
            lines = data[: data.rfind(b"\n") + 1].splitlines()
            if cursor < 0 or cursor > len(lines):
                raise ValueError("Invalid event cursor.")
            return [json.loads(line) for line in lines[cursor : None if limit is None else cursor + limit]]

    def submit(self, bundle):
        if not self.busy.acquire(blocking=False):
            LOGGER.info("Solve job rejected | server busy")
            raise BlockingIOError("A solve is already active; retry after it finishes.")
        try:
            job_id = uuid.uuid4().hex
            request, backend_id = stage_remote_job(bundle, self.root / job_id, select_backend=self.select_backend)
            request = replace(
                request, status_callback=lambda message: self.append(job_id, {"type": "status", "message": message})
            )
            session = self.backends[backend_id].create_system_session(request)
            with self.lock:
                self.sessions[job_id] = session
            self.append(
                job_id,
                {
                    "type": "accepted",
                    "backend_id": backend_id,
                    "execution_backend": session.request.solver_options.get(
                        "bem_backend", backend_id.removeprefix("beat_")
                    ),
                },
            )
            LOGGER.info(
                "Solve job accepted | job=%s | backend=%s | frequencies=%d | meshes=%d",
                job_id,
                backend_id,
                len(request.frequencies_hz),
                len(request.compiled_system.meshes),
            )
            thread = threading.Thread(target=self.run, args=(job_id, session), daemon=True)
            self.threads.append(thread)
            thread.start()
            return job_id
        except Exception:
            self.busy.release()
            raise

    def run(self, job_id, session):
        started = time.monotonic()
        count = 0
        try:
            for result in session.solve_stream():
                self.append(job_id, {"type": "result", "result": system_frequency_result_to_dict(result)})
                count += 1
            with self.lock:
                cancelled = self.sessions.get(job_id) is None
                if not cancelled and count != len(session.request.frequencies_hz):
                    raise RuntimeError("Worker ended without all requested frequencies.")
                self.append(
                    job_id,
                    {
                        "type": "cancelled" if cancelled else "completed",
                        "worker": getattr(session, "worker_provenance", None),
                    },
                )
                LOGGER.info(
                    "Solve job %s | job=%s | frequencies=%d | elapsed=%.2fs",
                    "cancelled" if cancelled else "completed",
                    job_id,
                    count,
                    time.monotonic() - started,
                )
        except Exception as exc:
            LOGGER.error(
                "Solve job failed | job=%s | error=%s | frequencies=%d | elapsed=%.2fs",
                job_id,
                type(exc).__name__,
                count,
                time.monotonic() - started,
            )
            self.append(
                job_id, {"type": "failed", "error": str(exc), "worker": getattr(session, "worker_provenance", None)}
            )
        finally:
            with self.lock:
                self.sessions.pop(job_id, None)
            self.busy.release()

    def cancel(self, job_id):
        self.events(job_id, 0)
        with self.lock:
            session = self.sessions.get(job_id)
            if session is not None:
                self.sessions[job_id] = None
                LOGGER.info("Solve cancellation requested | job=%s", job_id)
                session.stop()

    def close(self):
        if self.discovery:
            self.discovery.close()
        with self.lock:
            active = list(self.sessions)
        for job_id in active:
            self.cancel(job_id)
        for thread in self.threads:
            thread.join(timeout=5)


def validate_server_access(mode, token):
    if mode not in {"private-network", "hosted"}:
        raise ValueError("Unknown server deployment mode.")
    if mode == "hosted" and not token:
        raise ValueError("Hosted mode requires an access key in BLAB_SERVER_TOKEN (or --token-env).")
    if token is not None and (len(token) < 32 or not token.isascii() or any(c.isspace() for c in token)):
        raise ValueError("Access key must contain at least 32 ASCII characters without whitespace.")


def create_http_server(
    service: SolveService,
    port: int = 8765,
    *,
    host: str = "127.0.0.1",
    token: str | None = None,
    mode: str = "private-network",
) -> ThreadingHTTPServer:
    validate_server_access(mode, token)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, code, value):
            data = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def handle_request(self):
            self.connection.settimeout(30)
            if token and not hmac.compare_digest(
                self.headers.get("Authorization", "").encode("utf-8"), f"Bearer {token}".encode("ascii")
            ):
                LOGGER.info("Client authentication rejected | peer=%s", self.client_address[0])
                self.reply(401, {"error": "Authentication required."})
                return
            path = urlsplit(self.path)
            try:
                if self.command == "GET" and path.path == "/v1/capabilities":
                    if parse_qs(path.query).get("connection_test") == ["1"]:
                        LOGGER.info(
                            "Client connection test received | peer=%s | state=%s",
                            self.client_address[0],
                            service.readiness(),
                        )
                    self.reply(
                        200,
                        {
                            "protocol": "blab-remote",
                            "version": REMOTE_VERSION,
                            "state": service.readiness(),
                            "request_schema_version": SYSTEM_SOLVE_REQUEST_VERSION,
                            "result_schema_version": SYSTEM_RESULT_VERSION,
                            "backend_ids": [
                                key for key, record in service.capabilities().items() if record["available"]
                            ],
                            "backends": service.capabilities(),
                            "phasor_conventions": [SOLVER_PHASOR_CONVENTION],
                            "observation_planes": False,
                            "max_bundle_bytes": MAX_BUNDLE_BYTES,
                            "runtime_check": "Startup snapshot; the solve worker rechecks runtime compatibility for every job.",
                        },
                    )
                elif self.command == "POST" and path.path == "/v1/jobs":
                    if self.headers.get_content_type() != "application/zip":
                        LOGGER.info("Solve payload rejected | unsupported content type")
                        self.reply(415, {"error": "Expected application/zip."})
                        return
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_BUNDLE_BYTES or self.headers.get("Transfer-Encoding"):
                        raise ValueError("A bounded Content-Length is required.")
                    LOGGER.info("Receiving mesh payload | peer=%s | bytes=%d", self.client_address[0], length)
                    bundle = self.rfile.read(length)
                    if len(bundle) != length:
                        raise ValueError("Incomplete upload.")
                    LOGGER.info("Mesh payload received; validating and staging | bytes=%d", length)
                    self.reply(202, {"job_id": service.submit(bundle)})
                elif match := re.fullmatch(r"/v1/jobs/([0-9a-f]{32})", path.path):
                    job_id = match[1]
                    if self.command == "GET":
                        cursor = int(parse_qs(path.query).get("cursor", ["0"])[0])
                        events = service.events(job_id, cursor)
                        self.reply(200, {"events": events, "next_cursor": cursor + len(events)})
                    elif self.command == "DELETE":
                        service.cancel(job_id)
                        self.reply(202, {"job_id": job_id})
                    else:
                        self.reply(405, {"error": "Unsupported method."})
                else:
                    self.reply(404, {"error": "Unknown endpoint."})
            except BlockingIOError as exc:
                self.reply(409, {"error": str(exc)})
            except FileNotFoundError as exc:
                self.reply(404, {"error": str(exc)})
            except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
                LOGGER.warning("Client request rejected | method=%s | error=%s", self.command, type(exc).__name__)
                self.reply(400, {"error": str(exc)})
            except (ConnectionError, TimeoutError):
                LOGGER.warning(
                    "Client connection interrupted | method=%s | peer=%s", self.command, self.client_address[0]
                )

        do_GET = handle_request
        do_POST = handle_request
        do_DELETE = handle_request

    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if isinstance(sys.exception(), (ConnectionError, TimeoutError)):
                return
            super().handle_error(request, client_address)

        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(30)
            return connection, address

    server = Server((host, port), Handler)
    return server


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog, description="Physical-system server for private networks or hosted deployments."
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Dedicated directory for job assets and retained events"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda", "rocm"),
        default="auto",
        help="Server execution policy; auto prefers CUDA, ROCm, then CPU",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--token-env", default="BLAB_SERVER_TOKEN", help="Environment variable holding the shared server token"
    )
    parser.add_argument("--mode", choices=("private-network", "hosted"), default="private-network")
    parser.add_argument("--julia-executable", default="julia")
    parser.add_argument("--julia-threads", default=None)
    parser.add_argument("--shutdown-on-stdin-eof", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = os.environ.get(args.token_env) or None
    try:
        validate_server_access(args.mode, token)
    except ValueError as exc:
        parser.error(str(exc))
    backends = {
        key: PhysicalSystemProductionBackend(
            bem_backend=key.removeprefix("beat_"),
            julia_executable=args.julia_executable,
            julia_threads=args.julia_threads,
        )
        for key in REMOTE_BACKENDS
    }
    service = SolveService(args.root, backends=backends, discover=True, backend_policy=args.backend)
    server = create_http_server(service, args.port, host=args.host, token=token, mode=args.mode)
    print(f"Physical-system server listening on http://{args.host}:{server.server_port}", flush=True)
    if args.shutdown_on_stdin_eof:

        def wait_for_parent():
            sys.stdin.read()
            server.shutdown()

        threading.Thread(target=wait_for_parent, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        LOGGER.info("Server shutting down")
        server.server_close()
        service.close()
        shutdown_beat_engine_workers()
        LOGGER.info("Server stopped")


if __name__ == "__main__":
    main()
