"""GeometryWorkflowController driven with no MainWindow at all.

Companion to test_solve_workflow_controller.py: geometry generation depends on
WorkflowView, PlotPresenter and GeometryInputs, so it can be exercised against
recording fakes rather than a real window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from blab.generators.base import GeneratedGeometry, GenerationCompleted, GenerationRequest  # noqa: E402
from blab.ui.application_state import OperationPhase, OperationState  # noqa: E402
from blab.ui.main_window.geometry_workflow import CANCEL_DELAY_MS, GeometryWorkflowController  # noqa: E402


class FakeView:
    def __init__(self) -> None:
        self.status: list[str] = []
        self.warnings: list[tuple[str, str]] = []
        self.errors: list[tuple[str, str]] = []
        self.phases: list[tuple[OperationPhase, bool]] = []
        self.cursor: list[bool] = []
        self.quality_warnings: list[object] = []
        self.plot_exports: list[bool] = []

    def show_status(self, message):
        self.status.append(message)

    def warn(self, title, message):
        self.warnings.append((title, message))

    def show_error(self, title, message):
        self.errors.append((title, message))

    def set_workflow_phase(self, phase, *, cancel_available=True):
        self.phases.append((phase, cancel_available))

    def set_busy_cursor(self, busy):
        self.cursor.append(busy)

    def show_mesh_quality_warning(self, result):
        self.quality_warnings.append(result)

    def set_plot_exports_available(self, available):
        self.plot_exports.append(available)


class FakePlots:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def refresh_contour_controls(self):
        self.calls.append("refresh_contour_controls")


class FakeInputs:
    def __init__(self, document=None) -> None:
        self._document = document
        self.recorded: list[tuple[str, object]] = []
        self.seeded = 0

    def active_generator_document(self):
        return self._document

    def apply_saved_source_config_to_result(self, result, mesh_name):
        return result

    def record_generated_geometry(self, document_id, result):
        self.recorded.append((document_id, result))

    def ensure_seeded_exterior_system(self):
        self.seeded += 1


class FakeOperation:
    def __init__(self, *, active=False, phase=OperationPhase.IDLE) -> None:
        self.active = active
        self.state = OperationState(phase=phase)
        self.started: list[object] = []

    def start(self, request):
        self.started.append(request)
        return True


def _geometry() -> GeneratedGeometry:
    return GeneratedGeometry(
        provider_id="ath",
        output_dir=Path("runs/case"),
        mesh_path=Path("runs/case/case.msh"),
        source_path=Path("runs/case/case.cfg"),
        radiators=(),
    )


@pytest.fixture
def controller(qapp):
    view, plots, inputs = FakeView(), FakePlots(), FakeInputs()
    geometry, solve = FakeOperation(), FakeOperation()
    built = GeometryWorkflowController(
        None,
        view=view,
        plots=plots,
        inputs=inputs,
        geometry_controller=geometry,
        solve_controller=solve,
    )
    built.view, built.plots, built.inputs = view, plots, inputs
    built.geometry, built.solve = geometry, solve
    return built


def test_generating_without_a_design_warns_instead_of_starting(controller) -> None:
    controller.generate_geometry()

    assert controller.view.warnings == [("No waveguide design", "Add a waveguide design before generating.")]
    assert controller.geometry.started == []


def test_a_running_solve_blocks_a_generation(controller) -> None:
    controller.solve.active = True

    controller.generate_geometry()

    assert controller.geometry.started == []
    assert controller.view.warnings == [], "a busy run is not a user error"


def test_stop_is_withheld_for_the_first_seconds_of_a_generation() -> None:
    """Regression: reading 'cancel disabled' as 'idle' left Generate and Solve live."""
    assert CANCEL_DELAY_MS == 3000


def test_cancel_becomes_available_once_the_run_is_still_going(controller) -> None:
    controller.geometry.state = OperationState(phase=OperationPhase.RUNNING)

    controller._enable_geometry_cancel_if_active()

    assert controller.view.phases == [(OperationPhase.RUNNING, True)]


def test_cancel_is_not_offered_for_a_run_that_already_ended(controller) -> None:
    controller.geometry.state = OperationState(phase=OperationPhase.COMPLETED)

    controller._enable_geometry_cancel_if_active()

    assert controller.view.phases == [], "a finished run must not re-enable Stop"


def test_finished_geometry_clears_the_wait_cursor_and_returns_to_idle(controller) -> None:
    controller._on_geometry_generation_finished()

    assert controller.view.cursor == [False]
    assert controller.view.phases == [(OperationPhase.IDLE, True)]


def test_a_failed_generation_is_an_error_not_a_warning(controller) -> None:
    controller._on_geometry_generation_failed("gmsh exploded")

    assert controller.view.errors == [("Geometry generation failed", "gmsh exploded")]
    assert controller.view.warnings == []
    assert "Generate failed" in controller.view.status


def test_a_cancelled_generation_says_so(controller) -> None:
    controller._on_geometry_generation_cancelled()

    assert controller.view.status == ["Geometry generation stopped"]


def test_generated_geometry_is_recorded_seeded_and_quality_checked(controller) -> None:
    result = _geometry()
    request = GenerationRequest(
        provider_id="ath",
        document_id="doc-1",
        mesh_name="waveguide",
        source=None,
        run_root=Path("runs"),
        case_name="case",
        provider_options={},
    )
    emitted: list[str] = []
    controller.mesh_state_changed.connect(emitted.append)

    controller._on_geometry_generated(GenerationCompleted(request=request, result=result))

    assert controller.inputs.recorded == [("doc-1", result)]
    assert controller.inputs.seeded == 1
    assert emitted == ["geometry_generated"]
    assert controller.view.quality_warnings == [result]
