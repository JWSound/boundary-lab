"""Launch the GUI with startup and post-solve timing spans.

Usage: python scripts/profile_solve_completion.py --output runs/completion.jsonl

Uses normal settings and numerical execution. Close the GUI to finish. Each
JSONL row includes monotonic timestamps and wall/thread CPU times. Nested spans
overlap: do not sum them. A plot.render span measures rasterization; plot.paint
measures Qt painting, not monitor presentation. Use a new output filename.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from time import perf_counter, thread_time
from unittest.mock import patch

from profile_solve_startup import StartupTrace


class CompletionTrace(StartupTrace):
    def install(self, stack):
        super().install(stack)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        import blab.live as live
        import blab.postprocess as postprocess
        import blab.ui.result_projection as projection
        import blab.ui.system_solve as worker
        from blab.solve_results import SolvedSystemBuilder
        from blab.ui.main_window.observation_planes import ObservationPlaneController
        from blab.ui.main_window.solve_session import SolveSession
        from blab.ui.main_window.solve_workflow import SolveWorkflowController
        from blab.ui.main_window.window import MainWindow
        from blab.ui.observation_plane_viewport import ObservationPlaneViewport
        from blab.ui.operation_controllers import SolveController
        from blab.ui.plots import IsobarCanvas

        targets = [
            (worker, "canonicalize_observation_result", "result.canonicalize"),
            (SolveController, "_on_result", "gui.receive_result"),
            (SolveController, "_on_finished", "gui.controller_finished"),
            (SolveWorkflowController, "_on_frequency_result", "gui.accumulate_result"),
            (SolveWorkflowController, "_on_solve_finished", "completion.total"),
            (SolveSession, "finalize_results", "completion.finalize"),
            (SolvedSystemBuilder, "finalize", "results.builder_finalize"),
            (ObservationPlaneController, "sync_view", "completion.observation_planes"),
            (MainWindow, "refresh_plots", "plots.refresh"),
            (MainWindow, "prepared_live_dataset", "plots.prepare"),
            (MainWindow, "refresh_contour_controls", "plots.contour_controls"),
            (MainWindow, "set_plot_exports_available", "completion.export_availability"),
            (MainWindow, "_update_on_axis_plot", "plots.update_on_axis"),
            (ObservationPlaneViewport, "set_field_results", "preview.set_field_results"),
            (ObservationPlaneViewport, "set_planes", "preview.set_planes"),
            (ObservationPlaneViewport, "_render", "preview.render"),
            (projection.ResultProjectionService, "prepare", "projection.total"),
            (projection.VisualizationProjection, "snapshot", "completion.snapshot"),
            (live.LiveSolveDataset, "as_visualization_dataset", "projection.visualization_arrays"),
            (live.LiveSolveDataset, "as_balloon_raw_bundle", "projection.sphere_bundle"),
            (live.LiveSolveDataset, "as_group_delay_arrays", "projection.group_delay"),
            (live.LiveSolveDataset, "_polar_export_arrays", "projection.polar_synthesis"),
            (live.LiveSolveDataset, "_channel_on_axis_dataset", "projection.on_axis"),
            (live, "prepare_visualization_data_from_arrays", "projection.postprocess"),
            (postprocess, "_interpolate_isobar_heatmap", "projection.interpolate_isobar"),
            (projection, "compute_spinorama_from_planes", "projection.spinorama"),
            (IsobarCanvas, "update_plot", "plots.update_isobar"),
            (FigureCanvasQTAgg, "paintEvent", "plot.paint"),
        ]
        for owner, name, stage in targets:
            stack.enter_context(self.wrap(owner, name, stage))

        original_status = MainWindow.show_status

        @wraps(original_status)
        def status(window, message, *args, **kwargs):
            start, cpu_start = perf_counter(), thread_time()
            attempt = self.attempt
            try:
                return original_status(window, message, *args, **kwargs)
            finally:
                self.record("status:" + str(message), start, cpu_start, attempt=attempt)

        stack.enter_context(patch.object(MainWindow, "show_status", status))

        # Instrument consumption, rather than wrapping a generator's creation.
        from beat_engine.worker import _SubmissionEvents

        original_next = _SubmissionEvents.__next__

        @wraps(original_next)
        def next_event(events):
            start, cpu_start = perf_counter(), thread_time()
            attempt = self.attempt
            kind = "stop"
            failed = False
            try:
                event = original_next(events)
                kind = str(event.get("type", "unknown"))
                return event
            except StopIteration:
                raise
            except Exception:
                failed = True
                raise
            finally:
                self.record("worker.event:" + kind, start, cpu_start, attempt=attempt, failed=failed)

        stack.enter_context(patch.object(_SubmissionEvents, "__next__", next_event))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream, ExitStack() as stack:
        CompletionTrace(stream).install(stack)
        print(f"Writing solve timings to {args.output.resolve()}", flush=True)
        from blab.gui import main as gui_main

        gui_main()


if __name__ == "__main__":
    main()
