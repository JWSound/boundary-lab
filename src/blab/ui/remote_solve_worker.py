"""Qt worker wrapper for server-backed BEM solves."""

from __future__ import annotations

import json
from urllib import error, request

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from blab.config import SimulationConfig
from blab.protocol import (
    frequency_result_from_dict,
    ndarray_from_wire,
    solve_request_from_config_and_frequencies,
)


class RemoteSolveWorker(QObject):
    initialized = Signal(object, object, object)
    result_ready = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, config: SimulationConfig, frequencies: np.ndarray, server_url: str):
        super().__init__()
        self.config = config
        self.frequencies = frequencies
        self.server_url = server_url.rstrip("/")
        self.job_id: str | None = None
        self._stop = False

    @Slot()
    def run(self) -> None:
        try:
            self.status.emit(f"Submitting job to {self.server_url}...")
            job = self._post_json(
                "/jobs",
                solve_request_from_config_and_frequencies(self.config, self.frequencies),
            )
            self.job_id = str(job["job_id"])
            self.status.emit(f"Server job {self.job_id[:8]} queued")
            self._stream_events()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._stop = True
        if self.job_id is None:
            return
        try:
            self._post_json(f"/jobs/{self.job_id}/cancel", {})
            self.status.emit("Stop requested on server; waiting for current frequency...")
        except Exception as exc:
            self.status.emit(f"Stop request failed: {exc}")

    def _stream_events(self) -> None:
        if self.job_id is None:
            raise RuntimeError("Cannot stream events before a job is created.")

        url = f"{self.server_url}/jobs/{self.job_id}/events?since=0"
        with request.urlopen(url, timeout=None) as response:
            for raw_line in response:
                if not raw_line:
                    continue
                event = json.loads(raw_line.decode("utf-8"))
                self._handle_event(event)
                if self._stop and event.get("type") in {"cancelled", "completed", "failed"}:
                    return

    def _handle_event(self, event: dict) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "queued":
            self.status.emit("Server job queued")
        elif event_type == "started":
            self.status.emit("Server job started")
        elif event_type == "initialized":
            sphere_metadata = event.get("sphere_metadata") or {}
            self.initialized.emit(
                ndarray_from_wire(event["polar_angle_deg"]),
                np.asarray(event.get("radiator_names", ["Radiator"])),
                {
                    key: ndarray_from_wire(value)
                    for key, value in sphere_metadata.items()
                },
            )
            self.status.emit("Solving on server...")
        elif event_type == "result":
            self.result_ready.emit(frequency_result_from_dict(event["result"]))
        elif event_type == "cancelling":
            self.status.emit("Server cancellation requested...")
        elif event_type == "cancelled":
            self.status.emit("Server job cancelled")
        elif event_type == "completed":
            self.status.emit("Server job complete")
        elif event_type == "failed":
            raise RuntimeError(str(event.get("error", "Server job failed.")))

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
