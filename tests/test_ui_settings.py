from blab.ui.settings import (
    balloon_angle_precision_from_points,
    balloon_sampling_points,
    live_plot_angle_samples,
    live_plot_freq_samples,
    normalize_balloon_angle_precision_deg,
    normalize_live_plot_quality,
)


def test_live_plot_quality_sample_mapping() -> None:
    assert normalize_live_plot_quality("LOW") == "low"
    assert normalize_live_plot_quality("bogus") == "medium"

    assert live_plot_angle_samples("low") == 180
    assert live_plot_freq_samples("low") == 90
    assert live_plot_angle_samples("medium") == 250
    assert live_plot_freq_samples("medium") == 125
    assert live_plot_angle_samples("high") == 500
    assert live_plot_freq_samples("high") == 250


def test_balloon_angle_precision_point_conversion() -> None:
    assert normalize_balloon_angle_precision_deg(0.1) == 0.5
    assert normalize_balloon_angle_precision_deg(30.0) == 15.0

    assert balloon_sampling_points(5.0) == 1650
    assert balloon_sampling_points(2.5) == 6600
    assert balloon_sampling_points(1.0) == 41253
    assert round(balloon_angle_precision_from_points(6000), 1) == 2.6
