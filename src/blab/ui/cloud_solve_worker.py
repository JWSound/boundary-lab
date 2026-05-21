"""Qt worker for submitting and streaming cloud solve jobs."""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from blab.cloud.bundle import write_solve_bundle
from blab.cloud.protocol import array_from_payload, frequency_result_from_payload
from blab.config import SimulationConfig


class CloudSolveWorker(QObject):
    initialized = Signal(object, object, object)
    result_ready = Signal(object)
    status = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, config: SimulationConfig, frequencies: np.ndarray, api_base_url: str):
        super().__init__()
        self.config = config
        self.frequencies = frequencies
        self.api_base_url = api_base_url.rstrip("/")
        self._stop = False
        self._job_id: str | None = None

    @Slot()
    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._stop = True
        if self._job_id is not None:
            try:
                self._post_json(f"/v1/solve-jobs/{self._job_id}/cancel", {})
            except Exception:
                pass

    async def _run_async(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError('Install cloud client dependencies with: python -m pip install -e ".[gui]"') from exc

        self.status.emit("Packaging cloud solve bundle...")
        with tempfile.TemporaryDirectory(prefix="blab_cloud_client_") as tmp_dir:
            bundle_path = write_solve_bundle(
                Path(tmp_dir) / "solve.blabsolve.zip",
                config=self.config,
                frequencies=self.frequencies,
            )
            self.status.emit("Uploading cloud solve bundle...")
            job = self._post_bytes("/v1/solve-jobs/bundle", bundle_path.read_bytes())

        self._job_id = str(job["job_id"])
        stream_url = self._websocket_url(str(job["stream_path"]))
        self.status.emit(f"Cloud job queued: {self._job_id}; streaming from {stream_url}")

        connect_kwargs = {"max_size": None}
        if "proxy" in inspect.signature(websockets.connect).parameters:
            connect_kwargs["proxy"] = None

        async with websockets.connect(stream_url, **connect_kwargs) as websocket:
            async for raw_message in websocket:
                if self._stop:
                    break
                event = json.loads(raw_message)
                event_type = event.get("type")
                if event_type == "status":
                    self.status.emit(str(event.get("message", "")))
                elif event_type == "initialized":
                    angles = array_from_payload(event["polar_angle_deg"])
                    radiator_names = array_from_payload(event["radiator_names"])
                    sphere_metadata = {
                        key: array_from_payload(value)
                        for key, value in event.get("sphere_metadata", {}).items()
                    }
                    self.initialized.emit(angles, radiator_names, sphere_metadata)
                elif event_type == "frequency_result":
                    self.result_ready.emit(frequency_result_from_payload(event["result"]))
                elif event_type == "failed":
                    self.failed.emit(str(event.get("message", "Cloud solve failed.")))
                    break
                elif event_type == "completed":
                    break

    def _post_bytes(self, path: str, data: bytes) -> dict:
        url = f"{self.api_base_url}{path}"
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/zip"},
        )
        return self._read_json_response(request)

    def _post_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.api_base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._read_json_response(request)

    def _read_json_response(self, request: urllib.request.Request) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Cloud API request failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach cloud API at {self.api_base_url}: {exc.reason}") from exc

    def _websocket_url(self, stream_path: str) -> str:
        parsed = urllib.parse.urlparse(self.api_base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        netloc = parsed.netloc
        base_path = parsed.path.rstrip("/")
        return urllib.parse.urlunparse((scheme, netloc, f"{base_path}{stream_path}", "", "", ""))
