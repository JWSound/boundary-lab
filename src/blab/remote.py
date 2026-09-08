"""Qt-free client for the experimental physical-system service."""

from __future__ import annotations

import json
import ssl
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from blab import __version__
from blab.remote_contract import REMOTE_VERSION, build_job_bundle
from blab.solvers.engine_contract import SYSTEM_RESULT_VERSION, SYSTEM_SOLVE_REQUEST_VERSION
from blab.system_contract import SystemSolveMetadata, system_frequency_result_from_dict


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Remote server redirects are not allowed.")


class RemoteBackend:
    backend_id = "beat_remote"
    label = "Boundary Lab Server"

    def __init__(self, url, *, token=None):
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Expected an HTTP(S) server base URL without credentials, path, or query.")
        if token is not None and (not token.isascii() or any(c.isspace() for c in token)):
            raise ValueError("Server token must be ASCII without whitespace.")
        self.url = url.rstrip("/")
        self.token = token
        context = ssl.create_default_context()
        self.opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context), _NoRedirect())

    def call(self, path, *, data=None, method="GET"):
        # Hosted proxies may reject urllib's generic Python user agent.
        headers = {"Content-Type": "application/zip", "User-Agent": f"BoundaryLab/{__version__}"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with self.opener.open(request, timeout=30) as response:
                status = response.status
                body = response.read()
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read())
            except (ValueError, UnicodeError):
                payload = None
            detail = payload.get("error") if isinstance(payload, dict) else None
            if not isinstance(detail, str) or not detail:
                detail = (
                    "The server or proxy returned an unexpected response. "
                    "Check the server address and container logs; if it is starting, try again shortly."
                )
            raise RuntimeError(f"Remote server HTTP {exc.code}: {detail}") from exc
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError(
                f"Remote server HTTP {status}: expected a JSON response. "
                "Check the server address and container logs; if it is starting, try again shortly."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Remote server HTTP {status}: expected a JSON object from Boundary Lab Server.")
        return payload

    def check_capabilities(self, *, connection_test=True):
        path = "/v1/capabilities?connection_test=1" if connection_test else "/v1/capabilities"
        info = self.call(path)
        expected = {
            "protocol": "blab-remote",
            "version": REMOTE_VERSION,
            "request_schema_version": SYSTEM_SOLVE_REQUEST_VERSION,
            "result_schema_version": SYSTEM_RESULT_VERSION,
        }
        if any(type(info.get(key)) is not type(value) or info.get(key) != value for key, value in expected.items()):
            raise ValueError("Incompatible remote server contract versions.")
        return info

    def wait_ready(self, *, timeout=150, stop_requested=None):
        deadline = time.monotonic() + timeout
        while True:
            if stop_requested is not None and stop_requested():
                raise InterruptedError("Connection cancelled.")
            info = self.check_capabilities(connection_test=False)
            state = info.get("state", "ready")
            if state == "ready":
                return
            if state != "starting":
                raise ValueError("The server is not ready to solve. Contact the server operator.")
            if time.monotonic() >= deadline:
                raise TimeoutError("The server is still starting. Try again shortly.")
            time.sleep(0.25)

    def create_system_session(self, request):
        self.check_capabilities(connection_test=False)
        return RemoteSession(self, request)


class RemoteSession:
    def __init__(self, backend, request):
        self.backend = backend
        self.request = request
        self.job_id = None
        self.worker_provenance = None
        self.selected_backend = None
        self._stop = False
        self._terminal = False
        self.metadata = SystemSolveMetadata(
            system_id=request.compiled_system.id,
            assumptions=request.compiled_system.assumptions,
            excitation_port_ids=request.excitation_port_ids,
            available_quantity_ids=tuple(output.id for output in request.outputs),
        )

    def solve_stream(self, *, stop_requested=None):
        if self._stop:
            return
        self.backend.wait_ready(stop_requested=lambda: self._stop or (stop_requested is not None and stop_requested()))
        callback = self.request.status_callback or (lambda _message: None)
        callback("Uploading mesh payload…")
        self.job_id = self.backend.call("/v1/jobs", data=build_job_bundle(self.request), method="POST")["job_id"]
        callback("Remote solve started.")
        cursor = 0
        while True:
            if self._stop or (stop_requested is not None and stop_requested()):
                self.stop()
                return
            reply = self.backend.call(f"/v1/jobs/{self.job_id}?cursor={cursor}")
            cursor = reply["next_cursor"]
            for event in reply["events"]:
                kind = event["type"]
                if kind == "accepted":
                    self.selected_backend = event["backend_id"]
                elif kind == "result":
                    yield system_frequency_result_from_dict(event["result"])
                # Detailed worker statuses remain in the server journal. Result
                # consumers produce the normal per-frequency timing summary.
                elif kind in {"completed", "cancelled", "failed"}:
                    self._terminal = True
                    self.worker_provenance = event.get("worker")
                    if kind != "completed":
                        raise RuntimeError(event.get("error", "Remote job cancelled."))
                    return
            if not reply["events"]:
                time.sleep(0.2)

    def stop(self):
        if not self._stop and self.job_id and not self._terminal:
            try:
                self.backend.call(f"/v1/jobs/{self.job_id}", method="DELETE")
            except (OSError, RuntimeError):
                # Preserve the original transport error. The retained job can be
                # inspected/cancelled by ID when the service becomes reachable.
                pass
        self._stop = True
