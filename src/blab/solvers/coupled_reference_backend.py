"""Double-precision Julia reference backend for directly coupled FEM-BEM systems."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Callable, Iterator

from blab.system_contract import (
    SystemFrequencyResult,
    SystemSolveMetadata,
    SystemSolveRequest,
    system_frequency_result_from_dict,
    system_solve_request_to_dict,
)

DEFAULT_COUPLED_REFERENCE_SCRIPT = Path(__file__).with_name("julia_local") / "coupled_solver.jl"
DEFAULT_COUPLED_REFERENCE_PROJECT = DEFAULT_COUPLED_REFERENCE_SCRIPT.parent


class CoupledReferenceSession:
    def __init__(
        self,
        request: SystemSolveRequest,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_COUPLED_REFERENCE_SCRIPT,
        julia_project: str | Path | None = DEFAULT_COUPLED_REFERENCE_PROJECT,
    ):
        self.request = request
        self.julia_executable = julia_executable.strip() or "julia"
        self.solver_script = Path(solver_script).resolve()
        self.julia_project = None if julia_project is None else Path(julia_project).resolve()
        self._process: subprocess.Popen[str] | None = None
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
        command = [self.julia_executable]
        if self.julia_project is not None:
            command.append(f"--project={self.julia_project}")
        command.append(str(self.solver_script))
        environment = os.environ.copy()
        environment.setdefault("JULIA_NUM_THREADS", "1")
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
                raise RuntimeError(detail or f"Coupled Julia reference solver exited with status {return_code}.")
        finally:
            self._close_process()

    def stop(self) -> None:
        self._stop = True
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


class CoupledReferenceBackend:
    backend_id = "coupled_reference"
    label = "Coupled FEM-BEM Reference (Julia CPU)"

    def __init__(
        self,
        *,
        julia_executable: str = "julia",
        solver_script: str | Path = DEFAULT_COUPLED_REFERENCE_SCRIPT,
        julia_project: str | Path | None = DEFAULT_COUPLED_REFERENCE_PROJECT,
    ):
        self.julia_executable = julia_executable
        self.solver_script = Path(solver_script)
        self.julia_project = None if julia_project is None else Path(julia_project)

    def create_system_session(self, request: SystemSolveRequest) -> CoupledReferenceSession:
        return CoupledReferenceSession(
            request,
            julia_executable=self.julia_executable,
            solver_script=self.solver_script,
            julia_project=self.julia_project,
        )
