"""Qt-free design source: the STEP file, placement and sizing, turned into pipeline settings."""

from pathlib import Path

from blab.cad_import import CadImportSettings
from blab.generators.configuration import exterior_sound_speed


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


def size_overrides(source):
    """``(body_sizes, face_group_sizes, face_sizes)`` of the source, each override ``{"elements_per_wavelength": N}``
    or ``{"size_m": x}``; ``face_sizes`` a list of ``{"face": CadFaceRef fields, **override}``.

    Designs saved before levels existed stored edge lengths (m) as ``body_sizes_m`` and ``face_group_sizes_m``;
    they are kept as edge lengths.
    """
    sizes = []
    for key in ("body_sizes", "face_group_sizes"):
        legacy = {name: {"size_m": size} for name, size in source.get(f"{key}_m", {}).items()}
        sizes.append(legacy | source.get(key, {}))
    return sizes[0], sizes[1], list(source.get("face_sizes", []))


def translation_m(source):
    """The source's translation (millimetres) in metres; values that are no numbers are left for the check."""
    return tuple(v / 1000 if isinstance(v, int | float) else v for v in source["translation_mm"])


def settings(source, configuration):
    """Pipeline settings for ``source`` with c, fmax and symmetry from the project configuration."""
    try:
        body_sizes, face_group_sizes, face_sizes = size_overrides(source)
        return CadImportSettings(
            face_groups=face_groups(source),
            fmax_hz=float(configuration["frequency"]["freq_max_hz"]),
            sound_speed_m_per_s=exterior_sound_speed(configuration),
            elements_per_wavelength=float(source["elements_per_wavelength"]),
            body_sizes=body_sizes,
            face_group_sizes=face_group_sizes,
            face_sizes=tuple((entry["face"], {k: v for k, v in entry.items() if k != "face"}) for entry in face_sizes),
            volume_sizes=dict(source.get("volume_sizes", {})),
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
