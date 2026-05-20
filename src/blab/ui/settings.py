"""Persistent GUI settings and preference models."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings


SETTINGS_ORG = "Boundary Lab"
SETTINGS_APP = "Ath4LiveBEM"


@dataclass
class GuiPreferences:
    gmres_tolerance: float = 1e-3
    polar_angle_step_deg: float = 10.0
    use_burton_miller: bool = True
    worker_count: int = 1
    polar_smoothing: int | None = 48
    horizontal_normalization_angle: float = 10.0
    vertical_normalization_angle: float = 10.0
    spl_max_db: float = 0.0
    spl_min_db: float = -30.0
    stitch_imported_meshes: bool = False
    stitch_tolerance_mm: float = 2.0
    spherical_sampling_enabled: bool = False
    spherical_sampling_points: int = 6000


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


def settings_optional_int(settings: QSettings, key: str, default: int | None) -> int | None:
    value = settings.value(key, default)
    if value is None or str(value).strip().lower() in {"", "none", "off"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
