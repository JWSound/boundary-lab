"""Qt-free design source: the STEP file, axis mapping and sizing, turned into pipeline settings."""

from pathlib import Path

from blab.cad_import import CadImportSettings

DEFAULT_SOUND_SPEED_M_PER_S = 343.0


def sound_speed(configuration):
    """Sound speed of the project's exterior air region, else the default."""
    system = configuration.get("physical_system") or {}
    for region in system.get("regions", []):
        if region.get("kind") == "unbounded_air":
            return float(region.get("sound_speed_m_per_s", DEFAULT_SOUND_SPEED_M_PER_S))
    return DEFAULT_SOUND_SPEED_M_PER_S


def settings(source, configuration):
    """Pipeline settings for ``source`` with c, fmax and symmetry from the project configuration."""
    if not isinstance(source.get("face_groups", []), list):
        raise ValueError("Face groups must be a list.")
    try:
        return CadImportSettings(
            fmax_hz=float(configuration["frequency"]["freq_max_hz"]),
            sound_speed_m_per_s=sound_speed(configuration),
            elements_per_wavelength=float(source["elements_per_wavelength"]),
            forward=source["forward"],
            up=source["up"],
            body_sizes_m=dict(source.get("body_sizes_m", {})),
            face_group_sizes_m=dict(source.get("face_group_sizes_m", {})),
            curvature_elements_per_circle=float(source["curvature_elements_per_circle"]),
            growth_rate=float(source["growth_rate"]),
            symmetry=configuration.get("symmetry_config", {}).get("mode", "off"),
            closed_shells=dict(source.get("closed_shells", {})),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid CAD import source: {exc}") from exc


def step_path(source):
    path = str(source.get("step_path") or "")
    if not path:
        raise ValueError("Choose a STEP file first.")
    if not Path(path).is_file():
        raise ValueError(f"STEP file not found: {path}")
    return Path(path)
