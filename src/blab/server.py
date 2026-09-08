"""Physical-system HTTP service with authenticated TLS and BEAT runtime discovery."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import ssl
import sys
import threading
import uuid
import zipfile
from collections.abc import Sequence
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from blab.remote_contract import MAX_BUNDLE_BYTES, REMOTE_BACKENDS, REMOTE_VERSION, stage_remote_job
from blab.server_capabilities import BackendDiscovery
from blab.solvers.beat_engine_runtime import shutdown_beat_engine_workers
from blab.solvers.coupled_backend import PhysicalSystemProductionBackend
from blab.solvers.engine_contract import SYSTEM_RESULT_VERSION, SYSTEM_SOLVE_REQUEST_VERSION
from blab.system_contract import system_frequency_result_to_dict


class SolveService:
    """One active job, disk-backed events, and independent client connections."""

    def __init__(self, root: Path, *, backend=None, backends=None, discover=False):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend or PhysicalSystemProductionBackend(bem_backend="cpu")
        self.backends = backends or {"beat_cpu": self.backend}
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

    def append(self, job_id, event):
        with self.lock, (self.root / job_id / "events.ndjson").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    def capabilities(self):
        return (
            self.discovery.snapshot()
            if self.discovery
            else {key: {"available": True, "state": "ready"} for key in self.backends}
        )

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
            raise BlockingIOError("A solve is already active; retry after it finishes.")
        try:
            job_id = uuid.uuid4().hex
            available = {key for key, record in self.capabilities().items() if record["available"]}
            request, backend_id = stage_remote_job(bundle, self.root / job_id, available_backends=available)
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
            thread = threading.Thread(target=self.run, args=(job_id, session), daemon=True)
            self.threads.append(thread)
            thread.start()
            return job_id
        except Exception:
            self.busy.release()
            raise

    def run(self, job_id, session):
        try:
            count = 0
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
        except Exception as exc:
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


def create_http_server(
    service: SolveService,
    port: int = 8765,
    *,
    host: str = "127.0.0.1",
    token: str | None = None,
    tls_context: ssl.SSLContext | None = None,
) -> ThreadingHTTPServer:
    if token is not None and (len(token) < 32 or not token.isascii() or any(c.isspace() for c in token)):
        raise ValueError("Server token must contain at least 32 ASCII characters without whitespace.")
    if host not in {"127.0.0.1", "localhost"} and (not token or tls_context is None):
        raise ValueError("LAN binding requires both a token and a TLS certificate/key.")

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
                self.reply(401, {"error": "Authentication required."})
                return
            path = urlsplit(self.path)
            try:
                if self.command == "GET" and path.path == "/v1/capabilities":
                    self.reply(
                        200,
                        {
                            "protocol": "blab-remote",
                            "version": REMOTE_VERSION,
                            "request_schema_version": SYSTEM_SOLVE_REQUEST_VERSION,
                            "result_schema_version": SYSTEM_RESULT_VERSION,
                            "backend_ids": [
                                key for key, record in service.capabilities().items() if record["available"]
                            ],
                            "backends": service.capabilities(),
                            "observation_planes": False,
                            "max_bundle_bytes": MAX_BUNDLE_BYTES,
                            "runtime_check": "Startup snapshot; the solve worker rechecks runtime compatibility for every job.",
                        },
                    )
                elif self.command == "POST" and path.path == "/v1/jobs":
                    if self.headers.get_content_type() != "application/zip":
                        self.reply(415, {"error": "Expected application/zip."})
                        return
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_BUNDLE_BYTES or self.headers.get("Transfer-Encoding"):
                        raise ValueError("A bounded Content-Length is required.")
                    bundle = self.rfile.read(length)
                    if len(bundle) != length:
                        raise ValueError("Incomplete upload.")
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
                self.reply(400, {"error": str(exc)})

        do_GET = handle_request
        do_POST = handle_request
        do_DELETE = handle_request

    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if isinstance(sys.exception(), (ssl.SSLError, ConnectionError, TimeoutError)):
                return
            super().handle_error(request, client_address)

        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(30)
            if tls_context is not None:
                connection = tls_context.wrap_socket(connection, server_side=True, do_handshake_on_connect=False)
            return connection, address

    server = Server((host, port), Handler)
    return server


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog, description="Physical-system CPU/CUDA/ROCm server (localhost by default; authenticated TLS for LAN)."
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Dedicated directory for job assets and retained events"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--token-env", default="BLAB_SERVER_TOKEN", help="Environment variable holding the shared server token"
    )
    parser.add_argument("--tls-cert", type=Path, help="PEM certificate chain for HTTPS")
    parser.add_argument("--tls-key", type=Path, help="PEM private key for HTTPS")
    parser.add_argument("--julia-executable", default="julia")
    parser.add_argument("--julia-threads", default=None)
    parser.add_argument("--shutdown-on-stdin-eof", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("--tls-cert and --tls-key must be supplied together")
    context = None
    if args.tls_cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.tls_cert, args.tls_key)
    backends = {
        key: PhysicalSystemProductionBackend(
            bem_backend=key.removeprefix("beat_"),
            julia_executable=args.julia_executable,
            julia_threads=args.julia_threads,
        )
        for key in REMOTE_BACKENDS
    }
    service = SolveService(args.root, backends=backends, discover=True)
    server = create_http_server(
        service, args.port, host=args.host, token=os.environ.get(args.token_env), tls_context=context
    )
    scheme = "https" if context else "http"
    print(f"Physical-system server listening on {scheme}://{args.host}:{server.server_port}", flush=True)
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
        server.server_close()
        service.close()
        shutdown_beat_engine_workers()


if __name__ == "__main__":
    main()
