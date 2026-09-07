"""BEAT Engine solver backend adapter."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from blab.config import ChannelConfig, RadiatorConfig, SimulationConfig
from blab.protocol import (
    build_mesh_assets,
    frequency_result_from_dict,
    ndarray_from_wire,
    solve_request_from_config_and_frequencies,
)
from blab.solvers.base import FrequencyResult, SolveMetadata, SolverCapabilities, SolveRequest
from blab.solvers.beat_engine_runtime import (
    BEAT_ENGINE_BACKENDS,
    BEAT_ENGINE_CPU_BACKEND,
    BEAT_ENGINE_CUDA_BACKEND,
    BEAT_ENGINE_ROCM_BACKEND,
    DEFAULT_BEAT_ENGINE_CPU_PROJECT,
    DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    DEFAULT_BEAT_ENGINE_PROJECT,
    DEFAULT_BEAT_ENGINE_ROCM_PROJECT,
    DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
    BeatEngineWorkerProcess,
    shutdown_beat_engine_workers,
)
from blab.solvers.beat_engine_runtime import (
    default_beat_engine_project as _default_beat_engine_project,
)
from blab.solvers.beat_engine_runtime import (
    friendly_julia_error as _friendly_julia_error,
)
from blab.solvers.beat_engine_runtime import (
    get_beat_engine_worker as _get_julia_worker,
)
from blab.solvers.beat_engine_runtime import (
    julia_command as _julia_command,
)
from blab.solvers.beat_engine_runtime import (
    julia_process_env as _julia_process_env,
)
from blab.solvers.beat_engine_runtime import (
    julia_worker_command as _julia_worker_command,
)
from blab.solvers.beat_engine_runtime import (
    normalize_beat_engine_backend as _normalize_beat_engine_backend,
)
from blab.solvers.beat_engine_runtime import (
    resolve_julia_threads as _resolve_julia_threads,
)

# Compatibility exports for existing numerical reference harnesses.
__all__ = [
    "BEAT_ENGINE_BACKENDS",
    "BEAT_ENGINE_CPU_BACKEND",
    "BEAT_ENGINE_CUDA_BACKEND",
    "BEAT_ENGINE_ROCM_BACKEND",
    "DEFAULT_BEAT_ENGINE_CPU_PROJECT",
    "DEFAULT_BEAT_ENGINE_CUDA_PROJECT",
    "DEFAULT_BEAT_ENGINE_ROCM_PROJECT",
    "DEFAULT_BEAT_ENGINE_PROJECT",
    "DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT",
    "BeatEngineWorkerProcess",
    "shutdown_beat_engine_workers",
    "_normalize_beat_engine_backend",
    "_default_beat_engine_project",
    "_friendly_julia_error",
    "_julia_process_env",
    "_resolve_julia_threads",
    "_julia_command",
    "_julia_worker_command",
    "_get_julia_worker",
    "BeatEngineBackend",
    "BeatEngineSession",
    "BeatEngineCpuBackend",
    "BeatEngineCudaBackend",
    "BeatEngineRocmBackend",
    "DEFAULT_JULIA_PROJECT",
    "DEFAULT_JULIA_SOLVER_SCRIPT",
    "JuliaLocalBackend",
    "JuliaLocalSession",
    "JuliaWorkerProcess",
    "shutdown_julia_workers",
]


def _safe_asset_filename(filename: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._")
    return cleaned or f"asset_{index}.msh"


_DEFAULT_BEAT_ENGINE_PROJECT_SENTINEL = "__default__"


class BeatEngineSession:
    def __init__(
        self,
        request_payload: SolveRequest,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
        julia_threads: str | int = "auto",
        julia_project: str | Path | None = _DEFAULT_BEAT_ENGINE_PROJECT_SENTINEL,
        julia_sysimage: str | Path | None = None,
        persistent_worker: bool = True,
        beat_engine_backend: str = BEAT_ENGINE_CUDA_BACKEND,
    ):
        self.request_payload = request_payload
        self.julia_executable = julia_executable.strip() or "julia"
        self.solver_script = Path(solver_script)
        self.julia_threads = julia_threads
        self.persistent_worker = persistent_worker
        self.beat_engine_backend = _normalize_beat_engine_backend(beat_engine_backend)
        if julia_project == _DEFAULT_BEAT_ENGINE_PROJECT_SENTINEL:
            self.julia_project = _default_beat_engine_project(self.beat_engine_backend)
        else:
            self.julia_project = None if julia_project is None else Path(julia_project)
        self.julia_sysimage = None if julia_sysimage in (None, "") else Path(julia_sysimage)
        self._stop = False
        self._temp_dir = tempfile.TemporaryDirectory(prefix="blab-beat-engine-")
        self._process: subprocess.Popen[str] | None = None
        self._worker: BeatEngineWorkerProcess | None = None
        self._events: Iterator[dict] | None = None
        self._metadata: SolveMetadata | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._start_and_initialize()

    @property
    def metadata(self) -> SolveMetadata:
        if self._metadata is None:
            raise RuntimeError("BEAT Engine solver session has not initialized.")
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
                    if not self._stop:
                        yield frequency_result_from_dict(event["result"])
                elif event_type == "status":
                    self._emit_status(str(event.get("message", "")))
                    continue
                elif event_type == "cancelled":
                    return
                elif event_type == "completed":
                    return
                elif event_type == "failed":
                    if self._worker is not None:
                        # A failed accelerator job may leave the process-local
                        # runtime or allocator unhealthy. Never hand that worker
                        # to the next solve request.
                        self._worker.terminate()
                    raise RuntimeError(
                        _friendly_julia_error(
                            str(event.get("error", "BEAT Engine solver failed.")),
                            julia_project=self.julia_project,
                            beat_engine_backend=self.beat_engine_backend,
                        )
                    )
        finally:
            self._close()

    def stop(self) -> None:
        self._stop = True
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._worker is not None:
            self._request_worker_cancel()

    def _request_worker_cancel(self) -> None:
        cancel_path = getattr(self, "_cancel_path", None)
        if cancel_path is not None:
            Path(cancel_path).write_text("cancel", encoding="utf-8")

    def _start_and_initialize(self) -> None:
        if not self.solver_script.exists():
            raise RuntimeError(f"BEAT Engine solver script does not exist: {self.solver_script}")

        job_dir = Path(self._temp_dir.name)
        request_path = job_dir / "request.json"
        self._cancel_path = job_dir / "cancel"
        config = _stage_config_assets(self.request_payload.config, job_dir / "assets")
        payload = solve_request_from_config_and_frequencies(
            config,
            self.request_payload.frequencies_hz,
            include_assets=False,
        )
        payload["cancel_path"] = str(self._cancel_path)
        payload["beat_engine_backend"] = self.beat_engine_backend
        request_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

        if self.persistent_worker:
            self._emit_status("Initializing BEAT Engine")
            self._worker = _get_julia_worker(
                julia_executable=self.julia_executable,
                solver_script=self.solver_script,
                julia_threads=self.julia_threads,
                julia_project=self.julia_project,
                julia_sysimage=self.julia_sysimage,
            )
            self._events = self._worker.submit(request_path, status_callback=self._emit_status)
        else:
            self._emit_status("Initializing BEAT Engine")
            try:
                self._process = subprocess.Popen(
                    _julia_command(
                        self.julia_executable,
                        self.solver_script,
                        request_path,
                        julia_project=self.julia_project,
                        julia_sysimage=self.julia_sysimage,
                    ),
                    cwd=str(job_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=_julia_process_env(self.julia_threads, self.julia_project),
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Julia executable was not found. Set the Julia executable path in Preferences."
                ) from exc

            self._stderr_thread = threading.Thread(target=self._collect_stderr, daemon=True)
            self._stderr_thread.start()
            self._events = self._iter_events()

        for event in self._events:
            event_type = str(event.get("type", ""))
            if event_type == "status":
                self._emit_status(str(event.get("message", "")))
                continue
            elif event_type == "initialized":
                sphere_metadata = event.get("sphere_metadata") or {}
                self._metadata = SolveMetadata(
                    polar_angle_deg=ndarray_from_wire(event["polar_angle_deg"]),
                    radiator_names=np.asarray(event.get("radiator_names", ["Radiator"])),
                    sphere_metadata={key: ndarray_from_wire(value) for key, value in sphere_metadata.items()},
                )
                return
            elif event_type == "failed":
                if self._worker is not None:
                    self._worker.terminate()
                raise RuntimeError(
                    _friendly_julia_error(
                        str(event.get("error", "BEAT Engine solver failed.")),
                        julia_project=self.julia_project,
                        beat_engine_backend=self.beat_engine_backend,
                    )
                )
            elif event_type in {"completed", "cancelled"}:
                raise RuntimeError(f"BEAT Engine solver ended before initialization: {event_type}")

        raise RuntimeError(self._process_error("BEAT Engine solver ended before initialization."))

    def _iter_events(self) -> Iterator[dict]:
        process = self._process
        if process is None or process.stdout is None:
            return

        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                self._emit_status(text)
                continue
            if not isinstance(event, dict):
                continue
            yield event

        exit_code = process.wait()
        if exit_code != 0 and not self._stop:
            raise RuntimeError(self._process_error(f"BEAT Engine solver exited with code {exit_code}."))

    def _collect_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            text = line.strip()
            if text:
                self._stderr_lines.append(text)
                self._emit_status(text)

    def _process_error(self, fallback: str) -> str:
        detail = "\n".join(self._stderr_lines[-10:])
        message = f"{fallback}\n{detail}" if detail else fallback
        return _friendly_julia_error(
            message,
            julia_project=self.julia_project,
            beat_engine_backend=self.beat_engine_backend,
            detection_text="\n".join(self._stderr_lines),
        )

    def _close(self) -> None:
        events = self._events
        self._events = None
        close = getattr(events, "close", None)
        if close is not None:
            close()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self._temp_dir.cleanup()

    def _emit_status(self, message: str) -> None:
        if self.request_payload.status_callback is not None:
            self.request_payload.status_callback(message)


class BeatEngineBackend:
    backend_id = "beat_cuda"
    label = "BEAT Engine (Nvidia CUDA)"
    beat_engine_backend = BEAT_ENGINE_CUDA_BACKEND
    capabilities = SolverCapabilities(
        supports_remote_assets=False,
        supports_parallel_workers=False,
        supports_symmetry=True,
        supports_channel_resynthesis=True,
        is_remote=False,
    )

    def __init__(
        self,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT,
        julia_threads: str | int = "auto",
        julia_project: str | Path | None = _DEFAULT_BEAT_ENGINE_PROJECT_SENTINEL,
        julia_sysimage: str | Path | None = None,
        persistent_worker: bool = True,
        backend_id: str | None = None,
        label: str | None = None,
        beat_engine_backend: str | None = None,
    ):
        self.julia_executable = julia_executable
        self.solver_script = Path(solver_script)
        self.julia_threads = julia_threads
        self.persistent_worker = persistent_worker
        if backend_id is not None:
            self.backend_id = backend_id
        if label is not None:
            self.label = label
        if beat_engine_backend is not None:
            self.beat_engine_backend = _normalize_beat_engine_backend(beat_engine_backend)
        if julia_project == _DEFAULT_BEAT_ENGINE_PROJECT_SENTINEL:
            self.julia_project = _default_beat_engine_project(self.beat_engine_backend)
        else:
            self.julia_project = julia_project
        self.julia_sysimage = None if julia_sysimage in (None, "") else Path(julia_sysimage)
        if self.beat_engine_backend == BEAT_ENGINE_CPU_BACKEND:
            self.capabilities = BeatEngineCpuBackend.capabilities
        elif self.beat_engine_backend == BEAT_ENGINE_ROCM_BACKEND:
            self.capabilities = BeatEngineRocmBackend.capabilities

    def create_session(self, request: SolveRequest) -> BeatEngineSession:
        return BeatEngineSession(
            request,
            julia_executable=self.julia_executable,
            solver_script=self.solver_script,
            julia_threads=self.julia_threads,
            julia_project=self.julia_project,
            julia_sysimage=self.julia_sysimage,
            persistent_worker=self.persistent_worker,
            beat_engine_backend=self.beat_engine_backend,
        )

    def warm_up(
        self,
        mode: str = "worker",
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        mode = _normalize_warmup_mode(mode)
        if mode == "off":
            return

        if self.persistent_worker:
            worker = _get_julia_worker(
                julia_executable=self.julia_executable,
                solver_script=self.solver_script,
                julia_threads=self.julia_threads,
                julia_project=self.julia_project,
                julia_sysimage=self.julia_sysimage,
            )
            worker.ensure_started(status_callback=status_callback)

        if mode == "tiny":
            with tempfile.TemporaryDirectory(prefix="blab-beat-engine-warmup-") as temp_dir:
                mesh_path = Path(temp_dir) / "warmup_tetrahedron.msh"
                _write_warmup_tetrahedron_mesh(mesh_path)
                config = _warmup_simulation_config(mesh_path)
                request = SolveRequest(
                    config,
                    np.asarray([100.0], dtype=np.float32),
                    status_callback=status_callback,
                )
                session = self.create_session(request)
                list(session.solve_stream())


class BeatEngineCudaBackend(BeatEngineBackend):
    backend_id = "beat_cuda"
    label = "BEAT Engine (Nvidia CUDA)"
    beat_engine_backend = BEAT_ENGINE_CUDA_BACKEND


class BeatEngineCpuBackend(BeatEngineBackend):
    backend_id = "beat_cpu"
    label = "BEAT Engine (CPU)"
    beat_engine_backend = BEAT_ENGINE_CPU_BACKEND
    capabilities = SolverCapabilities(
        supports_remote_assets=False,
        supports_parallel_workers=False,
        supports_symmetry=True,
        supports_channel_resynthesis=True,
        is_remote=False,
    )


class BeatEngineRocmBackend(BeatEngineBackend):
    backend_id = "beat_rocm"
    label = "BEAT Engine (AMD ROCm)"
    beat_engine_backend = BEAT_ENGINE_ROCM_BACKEND
    capabilities = SolverCapabilities(
        supports_remote_assets=False,
        supports_parallel_workers=False,
        supports_symmetry=True,
        supports_channel_resynthesis=True,
        is_remote=False,
    )


def _normalize_warmup_mode(value: object) -> str:
    text = str(value or "off").strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "none": "off",
        "1": "worker",
        "true": "worker",
        "yes": "worker",
        "on": "worker",
        "start": "worker",
        "startup": "worker",
        "solve": "tiny",
        "job": "tiny",
    }
    mode = aliases.get(text, text)
    if mode not in {"off", "worker", "tiny"}:
        raise ValueError("Warm-up mode must be off, worker, or tiny.")
    return mode


def _write_warmup_tetrahedron_mesh(path: Path) -> None:
    path.write_text(
        """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
2 2 "warmup"
$EndPhysicalNames
$Nodes
4
1 0.0 0.0 0.0
2 0.08 0.0 0.0
3 0.0 0.08 0.0
4 0.0 0.0 0.08
$EndNodes
$Elements
4
1 2 2 2 2 1 3 2
2 2 2 2 2 1 2 4
3 2 2 2 2 2 3 4
4 2 2 2 2 3 1 4
$EndElements
""",
        encoding="utf-8",
    )


def _warmup_simulation_config(mesh_path: Path) -> SimulationConfig:
    return SimulationConfig(
        mesh_file=str(mesh_path),
        scale_factor=1.0,
        distance=1.0,
        step_size=90.0,
        min_angle=0.0,
        max_angle=0.0,
        freq_min=100.0,
        freq_max=100.0,
        freq_count=1,
        tag_throat=2,
        radiators=(RadiatorConfig(name="warmup", tag=2, channel="main"),),
        channels=(ChannelConfig(name="main"),),
        flat_target_normalization_enabled=False,
        spherical_sampling_enabled=False,
    )


def _stage_config_assets(config: SimulationConfig, asset_dir: Path) -> SimulationConfig:
    assets = build_mesh_assets(config)
    if not assets:
        return config

    asset_dir.mkdir(parents=True, exist_ok=True)
    staged_by_original_path: dict[str, str] = {}
    used_names: set[str] = set()
    for index, asset in enumerate(assets):
        original_path = str(asset["original_path"])
        filename = _safe_asset_filename(str(asset.get("filename") or Path(original_path).name), index)
        while filename in used_names:
            filename = f"{index}_{filename}"
        used_names.add(filename)
        staged_path = asset_dir / filename
        staged_path.write_bytes(Path(original_path).read_bytes())
        staged_by_original_path[original_path] = str(staged_path)

    meshes = tuple(replace(mesh, file=staged_by_original_path.get(mesh.file, mesh.file)) for mesh in config.meshes)
    return replace(
        config,
        mesh_file=staged_by_original_path.get(config.mesh_file, config.mesh_file),
        meshes=meshes,
    )


DEFAULT_JULIA_SOLVER_SCRIPT = DEFAULT_BEAT_ENGINE_SOLVER_SCRIPT
DEFAULT_JULIA_PROJECT = DEFAULT_BEAT_ENGINE_PROJECT
JuliaLocalSession = BeatEngineSession
JuliaWorkerProcess = BeatEngineWorkerProcess
JuliaLocalBackend = BeatEngineBackend
shutdown_julia_workers = shutdown_beat_engine_workers
