"""Phase-to-control mapping. Pure logic: no QApplication required."""

from __future__ import annotations

import pytest

from blab.ui.application_state import OperationPhase
from blab.ui.main_window.workflow_view import FrequencyRange, WorkflowControls, workflow_controls

MUTATING = ("generate", "solve", "mesh_config", "system_config", "channel_config", "speaker_package")


@pytest.mark.parametrize("phase", [OperationPhase.RUNNING, OperationPhase.CANCELLING])
def test_a_running_operation_locks_every_input(phase) -> None:
    controls = workflow_controls(phase, has_solver_meshes=True)

    for name in MUTATING:
        assert getattr(controls, name) is False, name


def test_cancel_is_offered_while_running_and_withdrawn_once_requested() -> None:
    assert workflow_controls(OperationPhase.RUNNING, has_solver_meshes=True).cancel is True
    assert workflow_controls(OperationPhase.CANCELLING, has_solver_meshes=True).cancel is False


@pytest.mark.parametrize(
    "phase",
    [OperationPhase.IDLE, OperationPhase.COMPLETED, OperationPhase.FAILED, OperationPhase.CANCELLED],
)
def test_settled_phases_restore_the_inputs_and_hide_cancel(phase) -> None:
    controls = workflow_controls(phase, has_solver_meshes=True)

    assert controls.cancel is False
    for name in MUTATING:
        assert getattr(controls, name) is True, name


def test_system_config_stays_gated_on_having_meshes() -> None:
    assert workflow_controls(OperationPhase.IDLE, has_solver_meshes=False).system_config is False
    assert workflow_controls(OperationPhase.IDLE, has_solver_meshes=True).system_config is True
    assert workflow_controls(OperationPhase.IDLE, has_solver_meshes=False).speaker_package is False
    assert workflow_controls(OperationPhase.IDLE, has_solver_meshes=True).speaker_package is True


def test_busy_state_ignores_mesh_availability() -> None:
    with_meshes = workflow_controls(OperationPhase.RUNNING, has_solver_meshes=True)
    without = workflow_controls(OperationPhase.RUNNING, has_solver_meshes=False)

    assert with_meshes == without


def test_a_range_entered_the_right_way_round_is_left_alone() -> None:
    entered = FrequencyRange(min_hz=200, max_hz=20_000, count=41)

    assert entered.normalized() == entered


def test_reversed_endpoints_are_swapped_for_the_solver() -> None:
    # The UI lets min be dragged above max; the solver must never see that.
    reversed_range = FrequencyRange(min_hz=20_000, max_hz=200, count=41)

    normalized = reversed_range.normalized()

    assert (normalized.min_hz, normalized.max_hz) == (200, 20_000)
    assert normalized.count == 41, "normalizing must not touch the count"


def test_span_is_measured_after_normalizing() -> None:
    assert FrequencyRange(min_hz=200, max_hz=20_000, count=41).span_hz == 19_800
    assert FrequencyRange(min_hz=20_000, max_hz=200, count=41).span_hz == 19_800


def test_a_degenerate_range_has_no_span() -> None:
    assert FrequencyRange(min_hz=1_000, max_hz=1_000, count=3).span_hz == 0


def test_frequency_range_is_immutable() -> None:
    value = FrequencyRange(min_hz=200, max_hz=20_000, count=41)

    with pytest.raises(AttributeError):
        value.min_hz = 20  # type: ignore[misc]


def test_controls_are_immutable() -> None:
    controls = workflow_controls(OperationPhase.IDLE, has_solver_meshes=True)

    assert isinstance(controls, WorkflowControls)
    with pytest.raises(AttributeError):
        controls.solve = False  # type: ignore[misc]


def test_withholding_cancel_never_re_enables_the_inputs() -> None:
    """Regression: geometry generation starts running with Stop withheld.

    Reading "cancel disabled" as "idle" would enable Generate and Solve during
    the first seconds of a generation.
    """
    controls = workflow_controls(OperationPhase.RUNNING, has_solver_meshes=True, cancel_available=False)

    assert controls.cancel is False
    for name in MUTATING:
        assert getattr(controls, name) is False, name


def test_cancel_available_defaults_to_offered_while_running() -> None:
    assert workflow_controls(OperationPhase.RUNNING, has_solver_meshes=True).cancel is True


def test_cancel_available_cannot_resurrect_cancel_once_cancelling() -> None:
    controls = workflow_controls(OperationPhase.CANCELLING, has_solver_meshes=True, cancel_available=True)

    assert controls.cancel is False, "a requested cancellation stays withdrawn"


def test_cancel_available_is_ignored_when_settled() -> None:
    controls = workflow_controls(OperationPhase.IDLE, has_solver_meshes=True, cancel_available=False)

    assert controls.cancel is False
    assert controls.solve is True, "settled phases still restore the inputs"
