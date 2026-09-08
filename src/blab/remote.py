"""Qt-free client for the experimental physical-system service."""

from __future__ import annotations

import json
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from blab.remote_contract import REMOTE_VERSION, build_job_bundle
from blab.solvers.engine_contract import SYSTEM_RESULT_VERSION, SYSTEM_SOLVE_REQUEST_VERSION
from blab.system_contract import SystemSolveMetadata, system_frequency_result_from_dict


class RemoteBackend:
    backend_id = "beat_remote"
    label = "Boundary Lab Server (CPU preview)"

    def __init__(self, url):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("The remote preview requires an http://127.0.0.1:PORT base URL.")
        self.url = url.rstrip("/")
        self.opener = build_opener(ProxyHandler({}))

    def call(self, path, *, data=None, method="GET"):
        request = Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/zip"})
        try:
            with self.opener.open(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = json.loads(exc.read()).get("error", str(exc))
            raise RuntimeError(f"Remote server HTTP {exc.code}: {detail}") from exc

    def check_capabilities(self):
        info = self.call("/v1/capabilities")
        expected = {
            "protocol": "blab-remote",
            "version": REMOTE_VERSION,
            "request_schema_version": SYSTEM_SOLVE_REQUEST_VERSION,
            "result_schema_version": SYSTEM_RESULT_VERSION,
        }
        if any(type(info.get(key)) is not type(value) or info.get(key) != value for key, value in expected.items()):
            raise ValueError("Incompatible remote server contract versions.")
        if "beat_cpu" not in info.get("backend_ids", []):
            raise ValueError("Remote server does not support CPU jobs.")
        return info

    def create_system_session(self, request):
        self.check_capabilities()
        return RemoteSession(self, request)


class RemoteSession:
    def __init__(self, backend, request):
        self.backend = backend
        self.request = request
        self.job_id = None
        self.worker_provenance = None
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
        self.job_id = self.backend.call("/v1/jobs", data=build_job_bundle(self.request), method="POST")["job_id"]
        callback = self.request.status_callback or (lambda _message: None)
        callback(f"Remote job: {self.job_id}")
        cursor = 0
        while True:
            if self._stop or (stop_requested is not None and stop_requested()):
                self.stop()
                return
            reply = self.backend.call(f"/v1/jobs/{self.job_id}?cursor={cursor}")
            cursor = reply["next_cursor"]
            for event in reply["events"]:
                kind = event["type"]
                if kind == "result":
                    yield system_frequency_result_from_dict(event["result"])
                elif kind == "status":
                    callback(event["message"])
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
