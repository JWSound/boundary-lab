"""Model-independent Julia worker transport, using only the Python standard library.

Callers supply executable/script paths and a complete child-process environment.
Requests and events are opaque JSON documents; no acoustic or application models
are imported here. WorkerPool owns only workers obtained through that pool.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path


class WorkerProcess:
    def __init__(
        self,
        *,
        julia_executable: str,
        solver_script: Path,
        julia_threads: str | int,
        julia_project: Path | None,
        julia_sysimage: Path | None = None,
        environment: Mapping[str, str] | None = None,
        backend_label: str = "the selected BEAT Engine backend",
    ):
        self.julia_executable = julia_executable
        self.solver_script = solver_script
        self.julia_threads = julia_threads
        self.julia_project = julia_project
        self.julia_sysimage = julia_sysimage
        self.environment = dict(os.environ if environment is None else environment)
        self.environment["JULIA_NUM_THREADS"] = resolve_julia_threads(julia_threads)
        self.backend_label = backend_label
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._status_callback: Callable[[str], None] | None = None
        self._worker_info: dict | None = None

    @property
    def worker_info(self) -> dict | None:
        """A copy of the current process's ready announcement, if any."""
        return copy.deepcopy(self._worker_info)

    def _accept_ready(self, event: dict) -> None:
        self._worker_info = copy.deepcopy(event)

    def _prepare_submission(self, request_path: Path, operation: str) -> dict:
        return {"request": str(request_path), "operation": str(operation)}

    def _accept_event(self, event: dict) -> None:
        """Optional protocol validation before exposing a submission event."""

    def submit(
        self,
        request_path: Path,
        *,
        status_callback: Callable[[str], None] | None = None,
        operation: str = "solve",
    ) -> Iterator[dict]:
        self._lock.acquire()
        self._status_callback = status_callback
        try:
            self._ensure_started()
            command = self._prepare_submission(request_path, operation)
            process = self._process
            if process is None or process.stdin is None:
                raise RuntimeError("Warm BEAT Engine solver did not provide stdin.")
            self._emit_status("Submitting solve request" if operation == "solve" else "Submitting field request")
            process.stdin.write(
                json.dumps(
                    command,
                    separators=(",", ":"),
                )
                + "\n"
            )
            process.stdin.flush()
            return self._iter_events_for_submission()
        except Exception:
            self._status_callback = None
            self._lock.release()
            raise

    def ensure_started(self, *, status_callback: Callable[[str], None] | None = None) -> None:
        with self._lock:
            previous_callback = self._status_callback
            self._status_callback = status_callback
            try:
                self._ensure_started()
            finally:
                self._status_callback = previous_callback

    def terminate(self) -> None:
        self._discard_process()
        if self._lock.locked():
            try:
                self._lock.release()
            except RuntimeError:
                pass

    def _discard_process(self) -> None:
        process = self._process
        self._process = None
        self._worker_info = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._emit_status("BEAT Engine ready")
            return

        self._stderr_lines.clear()
        self._worker_info = None
        command = julia_worker_command(
            self.julia_executable,
            self.solver_script,
            julia_project=self.julia_project,
            julia_sysimage=self.julia_sysimage,
        )
        self._emit_status("Initializing BEAT Engine")
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self.solver_script.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self.environment,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Julia executable was not found. Configure its executable path or add Julia to PATH."
            ) from exc

        self._stderr_thread = threading.Thread(target=self._collect_stderr, daemon=True)
        self._stderr_thread.start()

        for event in self._read_events():
            event_type = str(event.get("type", ""))
            if event_type == "ready":
                try:
                    self._accept_ready(event)
                except Exception:
                    self._discard_process()
                    raise
                self._emit_status("BEAT Engine ready")
                return
            if event_type == "failed":
                self._discard_process()
                raise RuntimeError(
                    format_julia_error(
                        str(event.get("error", "BEAT Engine solver failed during startup.")),
                        julia_project=self.julia_project,
                        backend_label=self.backend_label,
                    )
                )

        self._discard_process()
        raise RuntimeError(self._process_error("Warm BEAT Engine solver ended before startup completed."))

    def _iter_events_for_submission(self) -> Iterator[dict]:
        terminal = False
        try:
            for event in self._read_events():
                self._accept_event(event)
                terminal = str(event.get("type", "")) in {"completed", "cancelled", "failed"}
                yield event
                if terminal:
                    return
            raise RuntimeError(self._process_error("Warm BEAT Engine solver ended before job completion."))
        finally:
            if not terminal:
                # Unread events belong to this job and cannot become the next
                # submission's results. Restart and renegotiate after abandonment.
                self._discard_process()
            self._status_callback = None
            if self._lock.locked():
                self._lock.release()

    def _read_events(self) -> Iterator[dict]:
        process = self._process
        if process is None or process.stdout is None:
            return

        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            parse_started = time.perf_counter()
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                yield {"type": "status", "message": text}
                continue
            if isinstance(event, dict):
                if str(event.get("type", "")) == "result":
                    event["_transport"] = {
                        "julia_stdout_bytes": len(text.encode("utf-8")),
                        "python_julia_json_parse_s": time.perf_counter() - parse_started,
                    }
                yield event

        exit_code = process.wait()
        self._process = None
        self._worker_info = None
        if exit_code != 0:
            raise RuntimeError(self._process_error(f"Warm BEAT Engine solver exited with code {exit_code}."))

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
        return format_julia_error(
            message,
            julia_project=self.julia_project,
            detection_text="\n".join(self._stderr_lines),
            backend_label=self.backend_label,
        )

    def _emit_status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(message)


def format_julia_error(
    message: str,
    *,
    julia_project: str | Path | None,
    backend_label: str = "the selected BEAT Engine backend",
    detection_text: str | None = None,
) -> str:
    if julia_project is None:
        return message

    text = f"{detection_text or message}\n{message}".lower()
    missing_dependency_markers = (
        "argumenterror: package",
        "not found in current path",
        "run `import pkg; pkg.add",
        "could not load project",
        "failed to precompile",
    )
    julia_load_markers = (
        "loading.jl",
        "require(into::module",
        "require(uuidkey::base.pkgid",
    )
    cuda_load_markers = (
        "cuda.jl could not be loaded",
        "package cuda",
        "using cuda",
        "import cuda",
    )
    rocm_load_markers = (
        "amdgpu.jl could not be loaded",
        "package amdgpu",
        "using amdgpu",
        "import amdgpu",
    )
    looks_like_dependency_error = any(marker in text for marker in missing_dependency_markers)
    looks_like_julia_load_error = any(marker in text for marker in julia_load_markers)
    looks_like_cuda_error = any(marker in text for marker in cuda_load_markers)
    looks_like_rocm_error = any(marker in text for marker in rocm_load_markers)
    if not (
        looks_like_dependency_error or looks_like_julia_load_error or looks_like_cuda_error or looks_like_rocm_error
    ):
        return message

    project_path = Path(julia_project)
    install_command = f'julia --project={project_path} -e "using Pkg; Pkg.instantiate()"'
    return (
        f"BEAT Engine could not load the Julia dependencies for {backend_label}.\n\n"
        "This usually means the selected BEAT Engine Julia environment has not been installed yet. "
        "To install that environment, run:\n\n"
        f"{install_command}\n\n"
        f"Julia reported:\n{message}"
    )


def resolve_julia_threads(julia_threads: str | int = "auto") -> str:
    if isinstance(julia_threads, int):
        return str(max(1, julia_threads))

    text = str(julia_threads or "auto").strip().lower()
    if text == "auto":
        return str(os.cpu_count() or 1)

    try:
        return str(max(1, int(text)))
    except ValueError:
        return str(os.cpu_count() or 1)


def julia_command(
    julia_executable: str,
    solver_script: Path,
    request_path: Path,
    *,
    julia_project: Path | None,
    julia_sysimage: Path | None = None,
) -> list[str]:
    command = [julia_executable]
    if julia_sysimage is not None:
        command.append(f"--sysimage={julia_sysimage}")
    if julia_project is not None:
        command.append(f"--project={julia_project}")
        command.append("--startup-file=no")
    command.extend([str(solver_script), "--request", str(request_path)])
    return command


def julia_worker_command(
    julia_executable: str,
    solver_script: Path,
    *,
    julia_project: Path | None,
    julia_sysimage: Path | None = None,
) -> list[str]:
    command = [julia_executable]
    if julia_sysimage is not None:
        command.append(f"--sysimage={julia_sysimage}")
    if julia_project is not None:
        command.append(f"--project={julia_project}")
        command.append("--startup-file=no")
    command.extend([str(solver_script), "--worker"])
    return command


class WorkerPool:
    """Reuse workers only when their execution settings and environments match."""

    def __init__(self, factory: Callable[..., WorkerProcess] = WorkerProcess):
        self._factory = factory
        self._lock = threading.Lock()
        self._workers: dict[tuple, WorkerProcess] = {}

    def get_worker(
        self,
        *,
        julia_executable: str,
        solver_script: Path,
        julia_threads: str | int,
        julia_project: Path | None,
        julia_sysimage: Path | None = None,
        environment: Mapping[str, str] | None = None,
        backend_label: str = "the selected BEAT Engine backend",
    ) -> WorkerProcess:
        threads = resolve_julia_threads(julia_threads)
        env = dict(os.environ if environment is None else environment)
        env["JULIA_NUM_THREADS"] = threads
        key = (
            julia_executable,
            str(solver_script.resolve()),
            "" if julia_project is None else str(julia_project.resolve()),
            "" if julia_sysimage is None else str(julia_sysimage.resolve()),
            threads,
            tuple(sorted(env.items())),
            backend_label,
        )
        with self._lock:
            worker = self._workers.get(key)
            if worker is None:
                worker = self._factory(
                    julia_executable=julia_executable,
                    solver_script=solver_script,
                    julia_threads=threads,
                    julia_project=julia_project,
                    julia_sysimage=julia_sysimage,
                    environment=env,
                    backend_label=backend_label,
                )
                self._workers[key] = worker
            return worker

    def shutdown(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.terminate()
