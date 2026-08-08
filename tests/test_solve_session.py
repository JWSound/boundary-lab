"""SolveSession derivations. Pure logic: no Qt, no MainWindow."""

from __future__ import annotations

from dataclasses import dataclass

from blab.ui.main_window.solve_session import SolveSession


@dataclass
class _Dataset:
    """Stands in for LiveSolveDataset, which needs solver output to build."""

    solved_count: int = 0


def test_a_fresh_session_holds_nothing() -> None:
    session = SolveSession()

    assert session.live_dataset is None
    assert session.last_completed_visualization is None
    assert session.use_final_isobar_resolution is False
    assert session.final_isobar_plots_rendered is False


def test_solved_count_is_zero_before_a_solve_starts() -> None:
    assert SolveSession().solved_count == 0
    assert SolveSession().has_solved_data() is False


def test_a_started_but_unsolved_dataset_still_counts_as_no_data() -> None:
    """The dataset exists from solver init, before any frequency completes."""
    session = SolveSession(live_dataset=_Dataset(solved_count=0))

    assert session.solved_count == 0
    assert session.has_solved_data() is False


def test_solved_data_is_reported_once_a_frequency_lands() -> None:
    session = SolveSession(live_dataset=_Dataset(solved_count=1))

    assert session.solved_count == 1
    assert session.has_solved_data() is True


def test_beginning_a_run_discards_the_previous_results() -> None:
    session = SolveSession(
        live_dataset=_Dataset(solved_count=12),
        use_final_isobar_resolution=True,
        final_isobar_plots_rendered=True,
    )

    session.begin()

    assert session.live_dataset is None
    assert session.use_final_isobar_resolution is False
    assert session.final_isobar_plots_rendered is False
    assert session.has_solved_data() is False


def test_beginning_a_run_keeps_the_comparison_overlay() -> None:
    """The whole point of the overlay is to survive into the next solve."""
    previous = object()
    session = SolveSession(live_dataset=_Dataset(solved_count=12), last_completed_visualization=previous)

    session.begin()

    assert session.last_completed_visualization is previous


def test_forgetting_the_comparison_leaves_the_live_run_alone() -> None:
    running = _Dataset(solved_count=3)
    session = SolveSession(live_dataset=running, last_completed_visualization=object())

    session.forget_comparison()

    assert session.last_completed_visualization is None
    assert session.live_dataset is running


def test_final_resolution_flags_are_independent() -> None:
    """A completed solve earns final resolution before the plots are drawn."""
    session = SolveSession(use_final_isobar_resolution=True)

    assert session.final_isobar_plots_rendered is False
