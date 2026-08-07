"""SolveWorkflowController driven with no MainWindow at all.

This is the payoff of the seam work: the solve workflow depends on WorkflowView,
PlotPresenter and SolveInputs, so it can be exercised against plain recording
fakes. A UI revamp replaces those three implementations and this suite still
describes the behaviour that must survive.
"""

from __future__ import annotations

import pytest

from blab.ui.application_state import OperationPhase, SolveCompletion  # noqa: E402
from blab.ui.main_window.solve_session import SolveSession  # noqa: E402
from blab.ui.main_window.solve_workflow import SolveWorkflowController  # noqa: E402
from blab.ui.main_window.workflow_view import FrequencyRange  # noqa: E402
from blab.ui.project_state import new_project_document  # noqa: E402
from blab.ui.settings import GuiPreferences  # noqa: E402


class FakeView:
    """Records what the workflow asked the UI to do."""

    def __init__(self) -> None:
        self.status: list[str] = []
        self.warnings: list[tuple[str, str]] = []
        self.errors: list[tuple[str, str]] = []
        self.stitch_errors: list[tuple[str, Exception]] = []
        self.phases: list[tuple[OperationPhase, bool]] = []
        self.plot_exports: list[bool] = []
        self.polar_exports: list[bool] = []
        self.on_axis_exports: list[bool] = []
        self.balloon: list[bool] = []

    def show_status(self, message):
        self.status.append(message)

    def warn(self, title, message):
        self.warnings.append((title, message))

    def show_error(self, title, message):
        self.errors.append((title, message))

    def show_stitch_or_generic_error(self, title, exc):
        self.stitch_errors.append((title, exc))

    def set_workflow_phase(self, phase, *, cancel_available=True):
        self.phases.append((phase, cancel_available))

    def frequency_range(self):
        return FrequencyRange(min_hz=200, max_hz=20_000, count=41)

    def set_plot_exports_available(self, available):
        self.plot_exports.append(available)

    def set_polar_export_available(self, available):
        self.polar_exports.append(available)

    def set_on_axis_export_available(self, available):
        self.on_axis_exports.append(available)

    def set_balloon_plot_available(self, available):
        self.balloon.append(available)


class FakePlots:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name):
        def record(*_args, **_kwargs):
            self.calls.append(name)
            return None

        return record


class FakeInputs:
    def __init__(self, *, has_meshes=True, radiators=()) -> None:
        self._has_meshes = has_meshes
        self._radiators = radiators
        self.seeded = 0

    def has_solver_meshes(self):
        return self._has_meshes

    def ensure_seeded_exterior_system(self):
        self.seeded += 1

    def all_radiators(self):
        return self._radiators

    def reconcile_symmetry_with_backend(self):
        return False

    def channel_configs(self):
        return ()


class FakeOperation:
    def __init__(self, *, active=False) -> None:
        self.active = active
        self.started: list[object] = []
        self.cancelled = 0

    def start(self, request):
        self.started.append(request)
        return True

    def cancel(self):
        self.cancelled += 1


@pytest.fixture
def controller(qapp):
    """A controller built entirely from fakes — no window, no widgets."""
    view, plots, inputs = FakeView(), FakePlots(), FakeInputs()
    session = SolveSession()
    geometry, solve = FakeOperation(), FakeOperation()
    built = SolveWorkflowController(
        None,
        view=view,
        plots=plots,
        inputs=inputs,
        session=session,
        project=lambda: new_project_document(),
        preferences=lambda: GuiPreferences(),
        assembler=None,
        geometry_controller=geometry,
        solve_controller=solve,
    )
    built.view, built.plots, built.inputs = view, plots, inputs
    built.session, built.geometry, built.solve = session, geometry, solve
    return built


def test_the_controller_needs_no_main_window(controller) -> None:
    """The whole point: constructed from protocols, not from a QMainWindow."""
    assert isinstance(controller, SolveWorkflowController)


def test_beginning_a_solve_withdraws_every_export_entry_point(controller) -> None:
    controller._begin_run("Initializing solver...")

    assert controller.view.plot_exports == [False]
    assert controller.view.polar_exports == [False]
    assert controller.view.on_axis_exports == [False]
    assert controller.view.balloon == [False]


def test_solving_without_a_mesh_warns_instead_of_starting(controller) -> None:
    controller.inputs._has_meshes = False

    controller.start_solve()

    assert controller.view.warnings == [
        ("No mesh", "Enable at least one generated or imported mesh before solving.")
    ]
    assert controller.solve.started == []


def test_a_busy_geometry_run_blocks_a_second_start(controller) -> None:
    controller.geometry.active = True

    controller.start_solve()

    assert controller.solve.started == []
    assert controller.view.warnings == [], "a busy run is not a user error"


def test_solving_with_no_driven_surfaces_explains_what_to_add(controller) -> None:
    controller.start_solve()

    assert controller.view.warnings == [
        (
            "No driven surfaces",
            "Open System and add a prescribed-velocity component to a moving boundary.",
        )
    ]
    assert controller.inputs.seeded == 1, "seeding is attempted before the radiator check"


def test_cancelling_prefers_the_geometry_run_when_one_is_active(controller) -> None:
    controller.geometry.active = True

    controller.cancel_current_operation()

    assert controller.geometry.cancelled == 1
    assert controller.solve.cancelled == 0
    assert (OperationPhase.CANCELLING, True) in controller.view.phases


def test_cancelling_falls_through_to_the_solve(controller) -> None:
    controller.solve.active = True

    controller.cancel_current_operation()

    assert controller.solve.cancelled == 1
    assert controller.geometry.cancelled == 0


def test_cancelling_an_idle_workflow_does_nothing(controller) -> None:
    controller.cancel_current_operation()

    assert controller.solve.cancelled == 0
    assert controller.geometry.cancelled == 0


def test_a_failed_solve_is_reported_as_an_error_not_a_warning(controller) -> None:
    controller._on_solve_failed("backend exploded")

    assert controller.view.errors == [("Solve failed", "backend exploded")]
    assert controller.view.warnings == []


def test_finishing_with_no_results_restores_idle_without_offering_exports(controller) -> None:
    controller._on_solve_finished(
        SolveCompletion(phase=OperationPhase.COMPLETED, solved_count=0, expected_count=0, elapsed_s=1.0)
    )

    assert (OperationPhase.IDLE, True) in controller.view.phases
    assert controller.view.plot_exports == [], "nothing was solved, so nothing is exportable"


def test_a_cancelled_empty_solve_says_so(controller) -> None:
    controller._on_solve_finished(
        SolveCompletion(phase=OperationPhase.CANCELLED, solved_count=0, expected_count=41, elapsed_s=2.0)
    )

    assert "Solve stopped" in controller.view.status
