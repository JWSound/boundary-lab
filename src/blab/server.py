"""Experimental loopback physical-system HTTP service."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import uuid
import zipfile
from collections.abc import Sequence
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from blab.remote_contract import MAX_BUNDLE_BYTES, REMOTE_VERSION, stage_job_bundle
from blab.solvers.beat_engine_runtime import shutdown_beat_engine_workers
from blab.solvers.coupled_backend import PhysicalSystemProductionBackend
from blab.solvers.engine_contract import SYSTEM_RESULT_VERSION, SYSTEM_SOLVE_REQUEST_VERSION
from blab.system_contract import system_frequency_result_to_dict


class SolveService:
    """One active job, disk-backed events, and independent client connections."""

    def __init__(self, root: Path, *, backend=None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend or PhysicalSystemProductionBackend(bem_backend="cpu")
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
            request = stage_job_bundle(bundle, self.root / job_id)
            request = replace(
                request, status_callback=lambda message: self.append(job_id, {"type": "status", "message": message})
            )
            session = self.backend.create_system_session(request)
            with self.lock:
                self.sessions[job_id] = session
            self.append(job_id, {"type": "accepted", "backend_id": "beat_cpu"})
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
        with self.lock:
            active = list(self.sessions)
        for job_id in active:
            self.cancel(job_id)
        for thread in self.threads:
            thread.join(timeout=5)


def create_http_server(service: SolveService, port: int = 8765) -> ThreadingHTTPServer:
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
                            "backend_ids": ["beat_cpu"],
                            "observation_planes": False,
                            "max_bundle_bytes": MAX_BUNDLE_BYTES,
                            "runtime_check": "Worker availability is checked at job execution.",
                        },
                    )
                elif self.command == "POST" and path.path == "/v1/jobs":
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

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description="Experimental CPU physical-system server (localhost only).")
    parser.add_argument(
        "--root", type=Path, required=True, help="Dedicated directory for job assets and retained events"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--julia-executable", default="julia")
    parser.add_argument("--julia-threads", default=None)
    parser.add_argument("--shutdown-on-stdin-eof", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    backend = PhysicalSystemProductionBackend(
        bem_backend="cpu", julia_executable=args.julia_executable, julia_threads=args.julia_threads
    )
    service = SolveService(args.root, backend=backend)
    server = create_http_server(service, args.port)
    print(f"Physical-system server listening on http://127.0.0.1:{server.server_port}", flush=True)
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
