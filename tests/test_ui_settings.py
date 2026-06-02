from blab.ui.settings import (
    balloon_angle_precision_from_points,
    balloon_sampling_points,
    live_plot_angle_samples,
    live_plot_freq_samples,
    normalize_balloon_angle_precision_deg,
    normalize_live_plot_quality,
    preferences_require_solve_invalidation,
    preferences_require_visualization_refresh,
    GuiPreferences,
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


def test_preference_change_classification() -> None:
    baseline = GuiPreferences()

    assert baseline.spin_horizontal_reference_angle == 0.0
    assert baseline.spin_vertical_reference_angle == 0.0

    assert preferences_require_solve_invalidation(
        baseline,
        GuiPreferences(gmres_tolerance=5e-4),
    )
    assert preferences_require_solve_invalidation(
        baseline,
        GuiPreferences(spherical_sampling_enabled=True),
    )
    assert preferences_require_solve_invalidation(
        baseline,
        GuiPreferences(polar_observation_distance_m=3.5),
    )
    assert not preferences_require_visualization_refresh(
        baseline,
        GuiPreferences(gmres_tolerance=5e-4),
    )

    assert preferences_require_visualization_refresh(
        baseline,
        GuiPreferences(polar_smoothing=24),
    )
    assert preferences_require_visualization_refresh(
        baseline,
        GuiPreferences(spl_min_db=-40.0),
    )
    assert preferences_require_visualization_refresh(
        baseline,
        GuiPreferences(spin_horizontal_reference_angle=15.0),
    )
    assert preferences_require_visualization_refresh(
        baseline,
        GuiPreferences(spin_vertical_reference_angle=-10.0),
    )
    assert not preferences_require_solve_invalidation(
        baseline,
        GuiPreferences(polar_smoothing=24),
    )
    assert not preferences_require_solve_invalidation(
        baseline,
        GuiPreferences(spin_horizontal_reference_angle=15.0),
    )

    assert not preferences_require_solve_invalidation(
        baseline,
        GuiPreferences(theme="dark", worker_count=4, solve_server_url="http://127.0.0.1:9999"),
    )
    assert not preferences_require_visualization_refresh(
        baseline,
        GuiPreferences(theme="dark", worker_count=4, solve_server_url="http://127.0.0.1:9999"),
    )
