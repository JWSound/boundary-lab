"""Boundary Lab HTTP solve-server backend client."""

from __future__ import annotations

import json
from typing import Callable, Iterator
from urllib import error, request

import numpy as np

from blab.protocol import (
    frequency_result_from_dict,
    ndarray_from_wire,
    solve_request_from_config_and_frequencies,
)
from blab.solvers.base import (
    FrequencyResult,
    SolveMetadata,
    SolverCapabilities,
    SolveRequest,
)


class HttpServerSession:
    def __init__(self, request_payload: SolveRequest, server_url: str):
        self.request_payload = request_payload
        self.server_url = server_url.rstrip("/")
        self.job_id: str | None = None
        self._stop = False
        self._response = None
        self._events: Iterator[dict] | None = None
        self._metadata: SolveMetadata | None = None
        self._submit_and_initialize()

    @property
    def metadata(self) -> SolveMetadata:
        if self._metadata is None:
            raise RuntimeError("Server session has not initialized.")
        return self._metadata

    def solve_stream(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[FrequencyResult]:
        if self._events is None:
            return

        try:
            for event in self._events:
                if self._stop or (stop_requested is not None and stop_requested()):
                    self.stop()

                event_type = str(event.get("type", ""))
                if event_type == "result":
                    yield frequency_result_from_dict(event["result"])
                elif event_type == "cancelling":
                    self._emit_status("Server cancellation requested...")
                elif event_type == "cancelled":
                    self._emit_status("Server job cancelled")
                    return
                elif event_type == "completed":
                    self._emit_status("Server job complete")
                    return
                elif event_type == "failed":
                    raise RuntimeError(str(event.get("error", "Server job failed.")))
        finally:
            self._close_response()

    def stop(self) -> None:
        self._stop = True
        if self.job_id is None:
            return
        try:
            self._post_json(f"/jobs/{self.job_id}/cancel", {})
            self._emit_status("Stop requested on server; waiting for current frequency...")
        except Exception as exc:
            self._emit_status(f"Stop request failed: {exc}")

    def _submit_and_initialize(self) -> None:
        self._emit_status(f"Submitting job to {self.server_url}...")
        job = self._post_json(
            "/jobs",
            solve_request_from_config_and_frequencies(
                self.request_payload.config,
                self.request_payload.frequencies_hz,
                include_assets=True,
            ),
        )
        self.job_id = str(job["job_id"])
        self._emit_status(f"Server job {self.job_id[:8]} queued")

        self._response = request.urlopen(f"{self.server_url}/jobs/{self.job_id}/events?since=0", timeout=None)
        self._events = self._iter_events(self._response)

        for event in self._events:
            event_type = str(event.get("type", ""))
            if event_type == "queued":
                self._emit_status("Server job queued")
            elif event_type == "started":
                self._emit_status("Server job started")
            elif event_type == "initialized":
                sphere_metadata = event.get("sphere_metadata") or {}
                self._metadata = SolveMetadata(
                    polar_angle_deg=ndarray_from_wire(event["polar_angle_deg"]),
                    radiator_names=np.asarray(event.get("radiator_names", ["Radiator"])),
                    sphere_metadata={key: ndarray_from_wire(value) for key, value in sphere_metadata.items()},
                )
                self._emit_status("Solving on server...")
                return
            elif event_type == "cancelled":
                raise RuntimeError("Server job cancelled before initialization.")
            elif event_type == "completed":
                raise RuntimeError("Server job completed before initialization.")
            elif event_type == "failed":
                raise RuntimeError(str(event.get("error", "Server job failed.")))

        raise RuntimeError("Server event stream ended before initialization.")

    def _iter_events(self, response) -> Iterator[dict]:
        for raw_line in response:
            if raw_line:
                yield json.loads(raw_line.decode("utf-8"))

    def _post_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.server_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Server returned HTTP {exc.code}: {detail}") from exc

    def _emit_status(self, message: str) -> None:
        if self.request_payload.status_callback is not None:
            self.request_payload.status_callback(message)

    def _close_response(self) -> None:
        if self._response is not None:
            self._response.close()
            self._response = None


def query_server_health(server_url: str, *, timeout_s: float = 5.0) -> dict:
    normalized_url = str(server_url or "").strip().rstrip("/") or "http://127.0.0.1:8765"
    with request.urlopen(f"{normalized_url}/health", timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Solve server health response was not a JSON object.")
    return payload


def server_health_supports_symmetry(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    capabilities = payload.get("capabilities", {})
    if not isinstance(capabilities, dict):
        return False
    return bool(capabilities.get("supports_symmetry"))


class HttpServerBackend:
    backend_id = "server"
    label = "Server"
    capabilities = SolverCapabilities(
        supports_remote_assets=True,
        supports_parallel_workers=True,
        is_remote=True,
    )

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def create_session(self, request_payload: SolveRequest) -> HttpServerSession:
        if request_payload.config.symmetry != "off" and not self._server_supports_symmetry():
            raise RuntimeError("The configured solve server does not advertise symmetry support.")
        return HttpServerSession(request_payload, self.server_url)

    def _server_supports_symmetry(self) -> bool:
        try:
            payload = query_server_health(self.server_url)
        except Exception as exc:
            raise RuntimeError(f"Could not query solve server capabilities: {exc}") from exc
        return server_health_supports_symmetry(payload)
