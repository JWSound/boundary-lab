import pytest

from blab.solve_eta import (
    SolveProgress,
    SolveTimeEstimator,
    format_elapsed,
    format_estimate,
    format_solve_status,
    format_warm_up_status,
)
from blab.warm_up_history import WarmUpHistory, WarmUpKey


def test_warm_up_status_shows_phase_and_elapsed_time() -> None:
    assert format_warm_up_status("Preparing BEAT CPU geometry caches", 42.7) == (
        "Preparing BEAT CPU geometry caches · 0:42"
    )


def test_warm_up_status_shows_predicted_total_until_overrun() -> None:
    assert format_warm_up_status("Warming up", 42.0, 87.0) == "Warming up · 0:42 / ~1:00"
    assert format_warm_up_status("Warming up", 42.0, 40.0) == "Warming up · 0:42 / longer than usual"


def test_warm_up_overrun_is_judged_against_the_displayed_prediction() -> None:
    assert format_warm_up_status("Warming up", 60.0, 87.0) == "Warming up · 1:00 / ~1:00"
    assert format_warm_up_status("Warming up", 61.0, 87.0) == "Warming up · 1:01 / longer than usual"
    assert format_warm_up_status("Warming up", 95.0, 90.0) == "Warming up · 1:35 / ~2:00"


def test_solve_progress_predicts_and_records_warm_up_and_reports_eta(tmp_path) -> None:
    key = WarmUpKey("cpu", "0.4.0", "8", True)
    history = WarmUpHistory(tmp_path / "history.json")
    history.record(key, 90.0)
    now = [100.0]
    progress = SolveProgress(10, clock=lambda: now[0], history=history)
    progress.start_warm_up(key)

    now[0] = 120.0
    assert progress.status_fields() == {"elapsed_s": 20.0, "predicted_warm_up_s": 90.0}
    for arrival_s in (130.0, 140.0, 142.0, 146.0):
        now[0] = arrival_s
        fields = progress.frequency_fields()

    assert fields == {"elapsed_s": 46.0, "eta_s": 18.0}
    assert progress.status_fields() == {}  # Warm-up is over.
    assert history.durations_s(key) == [90.0, 30.0]


@pytest.mark.parametrize(("seconds", "text"), [(0.4, "0:00"), (65.9, "1:05"), (3725.0, "1:02:05")])
def test_elapsed_time_is_clock_style(seconds: float, text: str) -> None:
    assert format_elapsed(seconds) == text


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(1.0, "0:05"), (47.0, "0:45"), (48.0, "0:50"), (58.0, "1:00"), (135.0, "2:00"), (150.0, "3:00")],
)
def test_estimate_rounds_to_five_seconds_under_a_minute_and_whole_minutes_above(seconds: float, text: str) -> None:
    assert format_estimate(seconds) == text


def test_solve_status_shows_count_elapsed_and_remaining_time() -> None:
    assert format_solve_status(37, 120, 65.0, 135.0) == "Solving 37/120 · 1:05 elapsed · ~2:00 left"
    assert format_solve_status(2, 120, 65.0, None) == "Solving 2/120 · 1:05 elapsed"
    assert format_solve_status(120, 120, 65.0, 0.0) == "Solving 120/120 · 1:05 elapsed"


def test_remaining_time_is_mean_frequency_solve_time_times_remaining_frequencies() -> None:
    estimator = SolveTimeEstimator(total=10)
    # The first gap (10 s) contains pipeline fill and is excluded.
    for arrival_s in (30.0, 40.0, 42.0, 46.0):
        estimator.record_result(arrival_s)

    assert estimator.completed == 4
    assert estimator.remaining_s() == 18.0  # mean(2, 4) * 6 remaining


def test_remaining_time_needs_two_frequency_solve_times() -> None:
    estimator = SolveTimeEstimator(total=10)
    assert estimator.remaining_s() is None
    estimator.record_result(30.0)
    estimator.record_result(40.0)
    estimator.record_result(42.0)
    assert estimator.remaining_s() is None
