"""Launch the normal GUI with opt-in solve-startup tracing.

Usage: python scripts/profile_solve_startup.py --output runs/startup-timing.jsonl

Each JSON line is a completed span with monotonic start/end times, wall and
thread CPU durations, thread identity, and solve attempt number. Spans nest and
can overlap across threads; do not sum every duration. The trace includes real
plot redraws and worker request serialization, without changing solver inputs,
settings, progress messages, or numerical execution. Close the GUI to finish.

This developer diagnostic wraps private methods at runtime. No hooks are
installed when Boundary Lab is launched normally. Use a new output path for
each launch; timings are written as they become available, including failures.
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter, thread_time
from unittest.mock import patch


class StartupTrace:
    def __init__(self, stream):
        self.stream = stream
        self.attempt = 0
        self.lock = Lock()

    def record(self, stage, start, cpu_start, *, attempt, failed=False):
        end = perf_counter()
        cpu_end = thread_time()
        if not attempt:
            return
        row = {
            "attempt": attempt,
            "stage": stage,
            "thread": get_ident(),
            "start_s": start,
            "end_s": end,
            "wall_ms": (end - start) * 1000,
            "thread_cpu_ms": (cpu_end - cpu_start) * 1000,
            "failed": failed,
        }
        with self.lock:
            self.stream.write(json.dumps(row) + "\n")
            self.stream.flush()

    def wrap(self, owner, name, stage, *, begins_attempt=False):
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args, **kwargs):
            if begins_attempt:
                self.attempt += 1
            attempt = self.attempt
            start, cpu_start = perf_counter(), thread_time()
            phase = None
            if stage == "prepare.total":
                progress = kwargs.get("progress")

                def report(message):
                    nonlocal phase
                    if phase is not None:
                        self.record(phase[0], phase[1], phase[2], attempt=attempt)
                    phase = ("prepare.phase:" + message, perf_counter(), thread_time())
                    if progress is not None:
                        progress(message)

                kwargs["progress"] = report
            failed = True
            try:
                result = original(*args, **kwargs)
                failed = False
                return result
            finally:
                if phase is not None:
                    self.record(phase[0], phase[1], phase[2], attempt=attempt, failed=failed)
                label = stage
                if stage == "plot.render":
                    label += ":" + str(getattr(args[0], "title", type(args[0]).__name__))
                self.record(label, start, cpu_start, attempt=attempt, failed=failed)

        return patch.object(owner, name, measured)

    def install(self, stack):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        import blab.solvers.coupled_backend as backend
        import blab.system_contract as contract
        import blab.system_solve as core
        import blab.ui.main_window.solve_workflow as workflow
        import blab.ui.preparation_worker as preparation
        from blab.ui.main_window.window import MainWindow
        from blab.ui.mesh_preview import MeshPreview
        from blab.ui.operation_controllers import SolveController
        from blab.ui.system_solve import SystemSolveWorker

        stack.enter_context(
            self.wrap(
                workflow.SolveWorkflowController,
                "_prepare_system_solve",
                "gui.capture_and_submit",
                begins_attempt=True,
            )
        )
        targets = [
            (core.PhysicalSystemCompiler, "compile", "prepare.compile"),
            (core, "validate_solve_plan", "prepare.validate"),
            (core, "_polar_observation_points", "domains.polars"),
            (core, "_fibonacci_sphere_points", "domains.sphere"),
            (core, "bem_boundary_result_domain", "domains.bem"),
            (core, "fem_volume_result_domain", "domains.fem"),
            (workflow, "prepare_system_solve", "prepare.total"),
            (workflow, "inspect_system_meshes", "prepare.inspect"),
            (workflow, "sync_physical_system_meshes", "prepare.sync"),
            (preparation._Job, "run", "preparation.worker"),
            (preparation.PreparationController, "_complete", "gui.complete"),
            (preparation.PreparationController, "_progress", "gui.progress"),
            (workflow.SolveWorkflowController, "_start_prepared_system_solve", "gui.handoff"),
            (workflow.SolveWorkflowController, "_begin_run", "gui.begin_run"),
            (workflow.SolveWorkflowController, "_on_solver_initialized", "gui.initialized"),
            (workflow, "analyze_exterior_mesh_topology", "gui.topology"),
            (workflow.SolvedSystemBuilder, "__init__", "gui.result_builder"),
            (MainWindow, "clear_plots", "gui.clear_plots"),
            (MainWindow, "apply_last_completed_comparison", "gui.comparison"),
            (MainWindow, "show_mesh_topology_issues", "gui.preview"),
            (MeshPreview, "set_topology_report", "preview.topology_render"),
            (MainWindow, "set_workflow_phase", "gui.phase"),
            (MainWindow, "show_status", "gui.status"),
            (SolveController, "start", "gui.start_worker"),
            (SystemSolveWorker, "run", "solve.worker"),
            (backend.PhysicalSystemProductionBackend, "create_system_session", "worker.create_session"),
            (backend, "system_solve_request_to_dict", "worker.request_to_dict"),
            (contract, "validate_solve_request", "worker.schema_validation"),
            (contract, "validate_system_solve_request", "request.validation"),
            (contract, "_validate_json_value", "request.json_validation"),
            (FigureCanvasQTAgg, "draw", "plot.render"),
        ]
        for owner, name, stage in targets:
            stack.enter_context(self.wrap(owner, name, stage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New JSONL trace file")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream, ExitStack() as stack:
        trace = StartupTrace(stream)
        trace.install(stack)
        print(f"Writing solve-startup timings to {args.output.resolve()}", flush=True)
        from blab.gui import main as gui_main

        gui_main()


if __name__ == "__main__":
    main()
