"""Persistent GUI settings and preference models."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QSettings

from blab.solvers.registry import normalize_backend_id


SETTINGS_ORG = "Boundary Lab"
SETTINGS_APP = "Ath4LiveBEM"
LIVE_PLOT_QUALITY_ANGLE_SAMPLES = {
    "low": 180,
    "medium": 250,
    "high": 500,
}
BALLOON_ANGLE_PRECISION_MIN_DEG = 0.5
BALLOON_ANGLE_PRECISION_MAX_DEG = 15.0


@dataclass
class GuiPreferences:
    theme: str = "system"
    solve_backend: str = "local"
    solve_server_url: str = "http://127.0.0.1:8765"
    live_plot_streaming: bool = True
    live_plot_quality: str = "medium"
    gmres_tolerance: float = 1e-3
    polar_angle_step_deg: float = 10.0
    polar_observation_distance_m: float = 2.0
    use_burton_miller: bool = True
    polar_smoothing: int | None = 48
    horizontal_normalization_angle: float = 10.0
    vertical_normalization_angle: float = 10.0
    spin_horizontal_reference_angle: float = 0.0
    spin_vertical_reference_angle: float = 0.0
    spl_max_db: float = 0.0
    spl_min_db: float = -30.0
    stitch_tolerance_mm: float = 2.0
    spherical_sampling_enabled: bool = False
    balloon_angle_precision_deg: float = 2.5


SOLVE_AFFECTING_PREFERENCE_FIELDS = (
    "solve_backend",
    "gmres_tolerance",
    "polar_angle_step_deg",
    "polar_observation_distance_m",
    "use_burton_miller",
    "stitch_tolerance_mm",
    "spherical_sampling_enabled",
    "balloon_angle_precision_deg",
)
VISUALIZATION_PREFERENCE_FIELDS = (
    "live_plot_quality",
    "polar_smoothing",
    "horizontal_normalization_angle",
    "vertical_normalization_angle",
    "spin_horizontal_reference_angle",
    "spin_vertical_reference_angle",
    "spl_max_db",
    "spl_min_db",
)


def preferences_require_solve_invalidation(previous: GuiPreferences, current: GuiPreferences) -> bool:
    return _preferences_changed(previous, current, SOLVE_AFFECTING_PREFERENCE_FIELDS)


def preferences_require_visualization_refresh(previous: GuiPreferences, current: GuiPreferences) -> bool:
    return _preferences_changed(previous, current, VISUALIZATION_PREFERENCE_FIELDS)


def load_gui_preferences(settings: QSettings) -> GuiPreferences:
    defaults = GuiPreferences()
    return GuiPreferences(
        theme=normalize_theme(settings_str(settings, "preferences/theme", defaults.theme)),
        solve_backend=normalize_backend_id(
            settings_str(settings, "preferences/solve_backend", defaults.solve_backend)
        ),
        solve_server_url=settings_str(settings, "preferences/solve_server_url", defaults.solve_server_url),
        live_plot_streaming=settings_bool(
            settings,
            "preferences/live_plot_streaming",
            defaults.live_plot_streaming,
        ),
        live_plot_quality=normalize_live_plot_quality(
            settings_str(settings, "preferences/live_plot_quality", defaults.live_plot_quality)
        ),
        gmres_tolerance=settings_float(settings, "preferences/gmres_tolerance", defaults.gmres_tolerance),
        polar_angle_step_deg=settings_float(
            settings,
            "preferences/polar_angle_step_deg",
            defaults.polar_angle_step_deg,
        ),
        polar_observation_distance_m=settings_float(
            settings,
            "preferences/polar_observation_distance_m",
            defaults.polar_observation_distance_m,
        ),
        use_burton_miller=settings_bool(
            settings,
            "preferences/use_burton_miller",
            defaults.use_burton_miller,
        ),
        polar_smoothing=settings_optional_int(
            settings,
            "preferences/polar_smoothing",
            defaults.polar_smoothing,
        ),
        horizontal_normalization_angle=settings_float(
            settings,
            "preferences/horizontal_normalization_angle",
            defaults.horizontal_normalization_angle,
        ),
        vertical_normalization_angle=settings_float(
            settings,
            "preferences/vertical_normalization_angle",
            defaults.vertical_normalization_angle,
        ),
        spin_horizontal_reference_angle=settings_float(
            settings,
            "preferences/spin_horizontal_reference_angle",
            defaults.spin_horizontal_reference_angle,
        ),
        spin_vertical_reference_angle=settings_float(
            settings,
            "preferences/spin_vertical_reference_angle",
            defaults.spin_vertical_reference_angle,
        ),
        spl_max_db=settings_float(settings, "preferences/spl_max_db", defaults.spl_max_db),
        spl_min_db=settings_float(settings, "preferences/spl_min_db", defaults.spl_min_db),
        stitch_tolerance_mm=settings_float(
            settings,
            "preferences/stitch_tolerance_mm",
            defaults.stitch_tolerance_mm,
        ),
        spherical_sampling_enabled=settings_bool(
            settings,
            "preferences/spherical_sampling_enabled",
            defaults.spherical_sampling_enabled,
        ),
        balloon_angle_precision_deg=load_balloon_angle_precision_deg(settings, defaults),
    )


def save_gui_preferences(settings: QSettings, preferences: GuiPreferences) -> None:
    settings.setValue("preferences/theme", preferences.theme)
    settings.setValue("preferences/solve_backend", preferences.solve_backend)
    settings.setValue("preferences/solve_server_url", preferences.solve_server_url)
    settings.setValue("preferences/live_plot_streaming", preferences.live_plot_streaming)
    settings.setValue("preferences/live_plot_quality", preferences.live_plot_quality)
    settings.setValue("preferences/gmres_tolerance", preferences.gmres_tolerance)
    settings.setValue("preferences/polar_angle_step_deg", preferences.polar_angle_step_deg)
    settings.setValue(
        "preferences/polar_observation_distance_m",
        preferences.polar_observation_distance_m,
    )
    settings.setValue("preferences/use_burton_miller", preferences.use_burton_miller)
    settings.setValue("preferences/polar_smoothing", preferences.polar_smoothing)
    settings.setValue(
        "preferences/horizontal_normalization_angle",
        preferences.horizontal_normalization_angle,
    )
    settings.setValue(
        "preferences/vertical_normalization_angle",
        preferences.vertical_normalization_angle,
    )
    settings.setValue(
        "preferences/spin_horizontal_reference_angle",
        preferences.spin_horizontal_reference_angle,
    )
    settings.setValue(
        "preferences/spin_vertical_reference_angle",
        preferences.spin_vertical_reference_angle,
    )
    settings.setValue("preferences/spl_max_db", preferences.spl_max_db)
    settings.setValue("preferences/spl_min_db", preferences.spl_min_db)
    settings.setValue("preferences/stitch_tolerance_mm", preferences.stitch_tolerance_mm)
    settings.setValue("preferences/spherical_sampling_enabled", preferences.spherical_sampling_enabled)
    settings.setValue(
        "preferences/balloon_angle_precision_deg",
        preferences.balloon_angle_precision_deg,
    )


def load_balloon_angle_precision_deg(settings: QSettings, defaults: GuiPreferences) -> float:
    if settings.contains("preferences/balloon_angle_precision_deg"):
        return normalize_balloon_angle_precision_deg(
            settings_float(
                settings,
                "preferences/balloon_angle_precision_deg",
                defaults.balloon_angle_precision_deg,
            )
        )
    if settings.contains("preferences/spherical_sampling_points"):
        return balloon_angle_precision_from_points(
            settings_int(
                settings,
                "preferences/spherical_sampling_points",
                balloon_sampling_points(defaults.balloon_angle_precision_deg),
            )
        )
    return defaults.balloon_angle_precision_deg


def normalize_theme(theme: object) -> str:
    normalized = str(theme).strip().lower()
    return normalized if normalized in {"system", "light", "dark"} else "system"


def settings_bool(settings: QSettings, key: str, default: bool) -> bool:
    value = settings.value(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def settings_int(settings: QSettings, key: str, default: int) -> int:
    try:
        return int(settings.value(key, default))
    except (TypeError, ValueError):
        return default


def settings_float(settings: QSettings, key: str, default: float) -> float:
    try:
        return float(settings.value(key, default))
    except (TypeError, ValueError):
        return default


def settings_str(settings: QSettings, key: str, default: str) -> str:
    value = settings.value(key, default)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def normalize_live_plot_quality(value: object) -> str:
    quality = str(value or "medium").strip().lower()
    return quality if quality in LIVE_PLOT_QUALITY_ANGLE_SAMPLES else "medium"


def live_plot_angle_samples(quality: object) -> int:
    return LIVE_PLOT_QUALITY_ANGLE_SAMPLES[normalize_live_plot_quality(quality)]


def live_plot_freq_samples(quality: object) -> int:
    return max(1, live_plot_angle_samples(quality) // 2)


def normalize_balloon_angle_precision_deg(value: object) -> float:
    try:
        angle_deg = float(value)
    except (TypeError, ValueError):
        angle_deg = 2.5
    if not math.isfinite(angle_deg):
        angle_deg = 2.5
    return min(max(angle_deg, BALLOON_ANGLE_PRECISION_MIN_DEG), BALLOON_ANGLE_PRECISION_MAX_DEG)


def balloon_sampling_points(angle_precision_deg: object) -> int:
    angle_deg = normalize_balloon_angle_precision_deg(angle_precision_deg)
    angle_rad = math.radians(angle_deg)
    return max(1, int(round((4.0 * math.pi) / (angle_rad * angle_rad))))


def balloon_angle_precision_from_points(point_count: object) -> float:
    try:
        points = int(point_count)
    except (TypeError, ValueError):
        points = balloon_sampling_points(2.5)
    points = max(1, points)
    angle_rad = math.sqrt((4.0 * math.pi) / float(points))
    return normalize_balloon_angle_precision_deg(math.degrees(angle_rad))


def settings_optional_int(settings: QSettings, key: str, default: int | None) -> int | None:
    value = settings.value(key, default)
    if value is None or str(value).strip().lower() in {"", "none", "off"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _preferences_changed(previous: GuiPreferences, current: GuiPreferences, fields: tuple[str, ...]) -> bool:
    return any(getattr(previous, field) != getattr(current, field) for field in fields)
