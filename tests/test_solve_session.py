"""SolveSession derivations. Pure logic: no Qt, no MainWindow."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from blab.live import LiveSolveDataset
from blab.physical_model import PhysicalSolveKind
from blab.solve_results import (
    HORIZONTAL_POLAR_PRESSURE_ID,
    VERTICAL_POLAR_PRESSURE_ID,
    SolvedSystemBuilder,
    SolveProvenance,
)
from blab.solve_results.live_projection import LiveResultProjector
from blab.system_contract import QuantityResult, SystemFrequencyResult
from blab.ui.main_window.solve_session import SolveSession


@dataclass
class _Dataset:
    """Stands in for LiveSolveDataset, which needs solver output to build."""

    solved_count: int = 0


def test_canonical_stream_retains_excitation_basis_and_projects_channel_sum() -> None:
    pressure = np.asarray([[1 + 2j], [3 - 4j]], dtype=np.complex64)
    result = SystemFrequencyResult(
        freq_hz=500.0,
        excitation_port_ids=("port:a", "port:b"),
        quantities=tuple(
            QuantityResult(quantity_id, "exterior_pressure", "Pa", pressure, axes=("excitation", "observation"))
            for quantity_id in (HORIZONTAL_POLAR_PRESSURE_ID, VERTICAL_POLAR_PRESSURE_ID)
        ),
        diagnostics={"timings": {"assembly_s": 1.0, "mesh_setup_s": 0.2, "solve_s": 0.3}},
    )
    prepared = SimpleNamespace(
        solve_kind=PhysicalSolveKind.EXTERIOR_BEM,
        excitation_channel_names=np.asarray(["main", "main"]),
        excitation_component_names=np.asarray(["A", "B"]),
        polar_angle_deg=np.asarray([0.0]),
    )
    session = SolveSession(
        result_builder=SolvedSystemBuilder(
            frequencies_hz=(500.0,),
            excitation_ids=result.excitation_port_ids,
            provenance=SolveProvenance("beat_cpu", "exterior_bem"),
        ),
        live_dataset=LiveSolveDataset(prepared.polar_angle_deg),
        live_projector=LiveResultProjector(prepared),
    )

    session.add_result(result)

    snapshot = session.result_builder.snapshot(status="partial")
    np.testing.assert_array_equal(snapshot.quantities[HORIZONTAL_POLAR_PRESSURE_ID].values[0], pressure)
    projected = session.live_dataset.ordered_results()[0]
    np.testing.assert_array_equal(projected.horizontal_pressure, [[4 - 2j]])
    assert projected.channel_names.tolist() == ["main"]
    assert projected.timings.assembly_s == pytest.approx(1.2)
    assert session.result_builder.solved_count == session.solved_count == 1

    # Replacing a frequency updates both representations without counting it twice.
    session.add_result(result)
    assert session.result_builder.solved_count == session.solved_count == 1


def test_canonical_result_requires_initialized_session() -> None:
    with pytest.raises(RuntimeError, match="before its session was initialized"):
        SolveSession().add_result(SystemFrequencyResult(500.0, ()))


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
        result_builder=object(),
        transducer_motion=object(),
        electrical_impedance=object(),
        acoustic_load_impedance=object(),
        solved_system=object(),
        use_final_isobar_resolution=True,
        final_isobar_plots_rendered=True,
    )

    session.begin()

    assert session.live_dataset is None
    assert session.result_builder is None
    assert session.transducer_motion is None
    assert session.electrical_impedance is None
    assert session.acoustic_load_impedance is None
    assert session.solved_system is None
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
