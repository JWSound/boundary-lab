from blab.ui.settings import (
    GuiPreferences,
    balloon_angle_precision_from_points,
    balloon_sampling_points,
    gui_preferences_with_project_preferences,
    live_plot_angle_samples,
    live_plot_freq_samples,
    load_gui_preferences,
    normalize_balloon_angle_precision_deg,
    normalize_live_plot_quality,
    preferences_require_solve_invalidation,
    preferences_require_visualization_refresh,
    project_preferences_from_gui,
    save_gui_preferences,
)


class MemorySettings:
    def __init__(self, values: dict[str, object] | None = None):
        self.values = dict(values or {})

    def contains(self, key: str) -> bool:
        return key in self.values

    def value(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def setValue(self, key: str, value: object) -> None:
        self.values[key] = value


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


def test_hidden_solver_preferences_always_use_backend_defaults() -> None:
    settings = MemorySettings(
        {
            "preferences/gmres_tolerance": 1e-8,
            "preferences/use_burton_miller": False,
        }
    )

    preferences = load_gui_preferences(settings)

    assert preferences.gmres_tolerance == 0.001
    assert preferences.use_burton_miller is True

    saved = MemorySettings()
    save_gui_preferences(
        saved,
        GuiPreferences(gmres_tolerance=1e-8, use_burton_miller=False),
    )
    assert "preferences/gmres_tolerance" not in saved.values
    assert "preferences/use_burton_miller" not in saved.values


def test_preference_change_classification() -> None:
    baseline = GuiPreferences()

    assert baseline.spin_horizontal_reference_angle == 0.0
    assert baseline.spin_vertical_reference_angle == 0.0
    assert baseline.isobar_contour_step_db == 3.0

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
    assert preferences_require_solve_invalidation(
        baseline,
        GuiPreferences(normalized_channel_correction=False),
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
        GuiPreferences(isobar_contour_step_db=1.5),
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
        GuiPreferences(isobar_contour_step_db=0.0),
    )
    assert not preferences_require_solve_invalidation(
        baseline,
        GuiPreferences(spin_horizontal_reference_angle=15.0),
    )

    assert not preferences_require_solve_invalidation(
        baseline,
        GuiPreferences(theme="dark", solve_server_url="http://127.0.0.1:9999"),
    )
    assert not preferences_require_visualization_refresh(
        baseline,
        GuiPreferences(theme="dark", solve_server_url="http://127.0.0.1:9999"),
    )
    assert not preferences_require_solve_invalidation(
        baseline,
        GuiPreferences(live_plot_streaming=False),
    )
    assert not preferences_require_visualization_refresh(
        baseline,
        GuiPreferences(live_plot_streaming=False),
    )


def test_applying_project_preferences_preserves_solver_and_application_choices() -> None:
    current = GuiPreferences(
        theme="dark",
        solve_backend="server",
        solve_server_url="http://solver.example:8765",
        live_plot_streaming=False,
        live_plot_quality="high",
        gmres_tolerance=1e-7,
        use_burton_miller=False,
    )
    project = project_preferences_from_gui(
        GuiPreferences(
            polar_angle_step_deg=5.0,
            normalized_channel_correction=False,
            spherical_sampling_enabled=True,
        ),
        freq_min_hz=80,
        freq_max_hz=16000,
        freq_count=61,
    )

    applied = gui_preferences_with_project_preferences(current, project)

    assert applied.polar_angle_step_deg == 5.0
    assert applied.normalized_channel_correction is False
    assert applied.spherical_sampling_enabled is True
    assert applied.solve_backend == "server"
    assert applied.solve_server_url == "http://solver.example:8765"
    assert applied.gmres_tolerance == 1e-7
    assert applied.use_burton_miller is False
    assert applied.theme == "dark"
    assert applied.live_plot_streaming is False
    assert applied.live_plot_quality == "high"
