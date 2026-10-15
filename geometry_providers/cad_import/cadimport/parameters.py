"""Qt-free design source: the STEP file, placement and sizing, turned into pipeline settings."""

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


def face_groups(source):
    """``{name: faces}`` from the source's list of ``{"name", "faces": [CadFaceRef fields]}``."""
    groups = source.get("face_groups", [])
    if not isinstance(groups, list):
        raise ValueError("Face groups must be a list.")
    result = {}
    for group in groups:
        if group["name"] in result:
            raise ValueError(f"Face group '{group['name']}' is defined twice.")
        result[group["name"]] = tuple(group["faces"])
    return result


def translation_m(source):
    """The source's translation (millimetres) in metres; values that are no numbers are left for the check."""
    return tuple(v / 1000 if isinstance(v, int | float) else v for v in source["translation_mm"])


def settings(source, configuration):
    """Pipeline settings for ``source`` with c, fmax and symmetry from the project configuration."""
    try:
        return CadImportSettings(
            face_groups=face_groups(source),
            fmax_hz=float(configuration["frequency"]["freq_max_hz"]),
            sound_speed_m_per_s=sound_speed(configuration),
            elements_per_wavelength=float(source["elements_per_wavelength"]),
            body_sizes_m=dict(source.get("body_sizes_m", {})),
            face_group_sizes_m=dict(source.get("face_group_sizes_m", {})),
            curvature_elements_per_circle=float(source["curvature_elements_per_circle"]),
            growth_rate=float(source["growth_rate"]),
            symmetry=configuration.get("symmetry_config", {}).get("mode", "off"),
            closed_shells=dict(source.get("closed_shells", {})),
            rotation_deg=tuple(source["rotation_deg"]),
            translation_m=translation_m(source),
            interface_names=dict(source.get("interface_names", {})),
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
