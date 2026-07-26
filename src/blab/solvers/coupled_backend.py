"""Julia backend adapters for directly coupled FEM-BEM systems."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator

from blab.physical_model import (
    BoundaryKind,
    ComponentKind,
    ExcitationPortKind,
)
from blab.solvers.beat_engine_backend import (
    DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
    BeatEngineWorkerProcess,
    _get_julia_worker,
)
from blab.system_contract import (
    SystemFrequencyResult,
    SystemSolveMetadata,
    SystemSolveRequest,
    system_frequency_result_from_dict,
    system_solve_request_to_dict,
)

DEFAULT_COUPLED_SOLVER_SCRIPT = Path(__file__).with_name("julia_local") / "coupled_solver.jl"
DEFAULT_COUPLED_CPU_PROJECT = DEFAULT_COUPLED_SOLVER_SCRIPT.parent
COUPLED_BEM_BACKENDS = {"cpu", "cuda"}
COUPLED_BOUNDARY_KINDS = {
    BoundaryKind.RIGID,
    BoundaryKind.MOVING,
    BoundaryKind.INTERFACE,
}


class CoupledSession:
    def __init__(
        self,
        request: SystemSolveRequest,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_project: str | Path | None = DEFAULT_COUPLED_CPU_PROJECT,
        julia_threads: str | int = "4",
        persistent_worker: bool = True,
    ):
        self.request = request
        self.julia_executable = julia_executable.strip() or "julia"
        self.solver_script = Path(solver_script).resolve()
        self.julia_project = None if julia_project is None else Path(julia_project).resolve()
        self.julia_threads = julia_threads
        self.persistent_worker = persistent_worker
        self._process: subprocess.Popen[str] | None = None
        self._worker: BeatEngineWorkerProcess | None = None
        self._stop = False
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._metadata = SystemSolveMetadata(
            system_id=request.compiled_system.id,
            assumptions=request.compiled_system.assumptions,
            excitation_port_ids=request.excitation_port_ids,
            available_quantity_ids=tuple(output.id for output in request.outputs),
        )

    @property
    def metadata(self) -> SystemSolveMetadata:
        return self._metadata

    def solve_stream(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[SystemFrequencyResult]:
        if self._stop:
            return
        if self.persistent_worker:
            yield from self._solve_stream_persistent(stop_requested=stop_requested)
            return

        command = [self.julia_executable]
        if self.julia_project is not None:
            command.append(f"--project={self.julia_project}")
        command.append(str(self.solver_script))
        environment = os.environ.copy()
        environment.setdefault("JULIA_NUM_THREADS", str(self.julia_threads))
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
        )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._capture_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()
        try:
            self._process.stdin.write(json.dumps(system_solve_request_to_dict(self.request)))
            self._process.stdin.close()
            for line in self._process.stdout:
                if self._stop or (stop_requested is not None and stop_requested()):
                    self.stop()
                    return
                text = line.strip()
                if not text:
                    continue
                yield system_frequency_result_from_dict(json.loads(text))
            return_code = self._process.wait()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2.0)
            if return_code != 0 and not self._stop:
                detail = "\n".join(self._stderr_lines).strip()
                raise RuntimeError(detail or f"Coupled Julia solver exited with status {return_code}.")
        finally:
            self._close_process()

    def _solve_stream_persistent(
        self,
        *,
        stop_requested: Callable[[], bool] | None,
    ) -> Iterator[SystemFrequencyResult]:
        self._worker = _get_julia_worker(
            julia_executable=self.julia_executable,
            solver_script=self.solver_script,
            julia_threads=self.julia_threads,
            julia_project=self.julia_project,
        )
        callback = self.request.status_callback
        with tempfile.TemporaryDirectory(prefix="blab-coupled-") as temp_dir:
            request_path = Path(temp_dir) / "request.json"
            request_path.write_text(
                json.dumps(system_solve_request_to_dict(self.request), separators=(",", ":")),
                encoding="utf-8",
            )
            try:
                events = self._worker.submit(request_path, status_callback=callback)
                for event in events:
                    if self._stop or (stop_requested is not None and stop_requested()):
                        self.stop()
                        return
                    event_type = str(event.get("type", ""))
                    if event_type == "result":
                        yield system_frequency_result_from_dict(event["result"])
                    elif event_type == "status" and callback is not None:
                        callback(str(event.get("message", "")))
                    elif event_type == "failed":
                        raise RuntimeError(str(event.get("error", "Coupled Julia solver failed.")))
                    elif event_type in {"completed", "cancelled"}:
                        return
            except RuntimeError:
                if not self._stop:
                    raise

    def stop(self) -> None:
        self._stop = True
        worker = self._worker
        if worker is not None:
            worker.terminate()
            self._worker = None
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def _capture_stderr(self, stream) -> None:
        for line in stream:
            text = line.rstrip()
            if text:
                self._stderr_lines.append(text)
                callback = self.request.status_callback
                if callback is not None:
                    callback(text)

    def _close_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


class _CoupledBackend:
    def __init__(
        self,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_project: str | Path | None = DEFAULT_COUPLED_CPU_PROJECT,
        julia_threads: str | int = "4",
        persistent_worker: bool = True,
        precision: str = "float64",
        bem_backend: str = "cpu",
    ):
        normalized_precision = str(precision).strip().lower()
        if normalized_precision not in {"float32", "float64"}:
            raise ValueError("Coupled precision must be float32 or float64.")
        normalized_bem_backend = str(bem_backend).strip().lower()
        if normalized_bem_backend not in COUPLED_BEM_BACKENDS:
            raise ValueError("Coupled BEM backend must be cpu or cuda.")
        self.julia_executable = julia_executable
        self.solver_script = Path(solver_script)
        self.julia_project = None if julia_project is None else Path(julia_project)
        self.julia_threads = julia_threads
        self.persistent_worker = persistent_worker
        self.precision = normalized_precision
        self.bem_backend = normalized_bem_backend

    def create_system_session(self, request: SystemSolveRequest) -> CoupledSession:
        validate_coupled_capabilities(request)
        solver_options = dict(request.solver_options)
        solver_options["precision"] = self.precision
        solver_options["bem_backend"] = self.bem_backend
        typed_request = replace(request, solver_options=solver_options)
        return CoupledSession(
            typed_request,
            julia_executable=self.julia_executable,
            solver_script=self.solver_script,
            julia_project=self.julia_project,
            julia_threads=self.julia_threads,
            persistent_worker=self.persistent_worker,
        )


def validate_coupled_capabilities(request: SystemSolveRequest) -> None:
    """Reject physical-model features that the current coupled backend cannot solve."""

    system = request.compiled_system
    unsupported_boundaries = [
        boundary.id for boundary in system.boundaries if boundary.kind not in COUPLED_BOUNDARY_KINDS
    ]
    if unsupported_boundaries:
        raise ValueError(
            "Coupled solver does not support the boundary assignments used by: "
            + ", ".join(unsupported_boundaries)
        )
    parameterized_boundaries = [boundary.id for boundary in system.boundaries if boundary.parameters]
    if parameterized_boundaries:
        raise ValueError(
            "Coupled solver does not yet support boundary parameters on: "
            + ", ".join(parameterized_boundaries)
        )
    lossy_regions = [region.id for region in system.regions if region.loss_model]
    if lossy_regions:
        raise ValueError(
            "Coupled solver does not yet support acoustic loss models on: " + ", ".join(lossy_regions)
        )
    unsupported_components = [
        component.id
        for component in system.components
        if component.kind != ComponentKind.IDEAL_VELOCITY_SOURCE
    ]
    if unsupported_components:
        raise ValueError(
            "Coupled solver currently supports only prescribed-velocity components; unsupported: "
            + ", ".join(unsupported_components)
        )
    for component in system.components:
        unsupported_parameters = set(component.parameters) - {"motion_profile"}
        if unsupported_parameters:
            raise ValueError(
                f"Coupled solver does not support component parameters on '{component.id}': "
                + ", ".join(sorted(unsupported_parameters))
            )
        motion_profile = component.parameters.get("motion_profile", "uniform")
        if motion_profile != "uniform":
            raise ValueError(
                f"Coupled solver supports only uniform prescribed motion; component "
                f"'{component.id}' requests {motion_profile!r}."
            )
    unsupported_ports = [
        port.id for port in system.excitation_ports if port.kind != ExcitationPortKind.NORMAL_VELOCITY
    ]
    if unsupported_ports:
        raise ValueError(
            "Coupled solver currently supports only normal-velocity excitation ports; unsupported: "
            + ", ".join(unsupported_ports)
        )


class CoupledReferenceBackend(_CoupledBackend):
    """Backend-only double-precision CPU correctness reference."""

    backend_id = "coupled_reference"
    label = "Coupled FEM-BEM Reference (Julia CPU FP64)"

    def __init__(
        self,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_project: str | Path | None = DEFAULT_COUPLED_CPU_PROJECT,
        julia_threads: str | int = "4",
        persistent_worker: bool = True,
    ):
        super().__init__(
            julia_executable=julia_executable,
            solver_script=solver_script,
            julia_project=julia_project,
            julia_threads=julia_threads,
            persistent_worker=persistent_worker,
            precision="float64",
            bem_backend="cpu",
        )


class CoupledProductionBackend(_CoupledBackend):
    """FP32 coupled backend used by interactive BEAT Engine CPU/CUDA solves."""

    backend_id = "coupled_production"
    label = "Coupled FEM-BEM (FP32)"

    def __init__(
        self,
        *,
        bem_backend: str,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_threads: str | int | None = None,
        persistent_worker: bool = True,
    ):
        normalized_bem_backend = str(bem_backend).strip().lower()
        resolved_threads = (4 if normalized_bem_backend == "cuda" else 8) if julia_threads is None else julia_threads
        julia_project = (
            DEFAULT_BEAT_ENGINE_CUDA_PROJECT
            if normalized_bem_backend == "cuda"
            else DEFAULT_COUPLED_CPU_PROJECT
        )
        super().__init__(
            julia_executable=julia_executable,
            solver_script=solver_script,
            julia_project=julia_project,
            julia_threads=resolved_threads,
            persistent_worker=persistent_worker,
            precision="float32",
            bem_backend=normalized_bem_backend,
        )
