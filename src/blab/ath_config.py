"""Small parser for Ath/ABEC solve metadata embedded in .cfg files."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AthPolarSettings:
    name: str
    distance_m: float | None = None
    inclination_deg: float | None = None
    map_min_angle_deg: float | None = None
    map_max_angle_deg: float | None = None
    map_sample_count: int | None = None
    norm_angle_deg: float | None = None

    @property
    def map_step_deg(self) -> float | None:
        if (
            self.map_min_angle_deg is None
            or self.map_max_angle_deg is None
            or self.map_sample_count is None
            or self.map_sample_count < 2
        ):
            return None
        return abs(self.map_max_angle_deg - self.map_min_angle_deg) / float(self.map_sample_count - 1)


@dataclass(frozen=True)
class AthSolveSettings:
    freq_min_hz: float | None = None
    freq_max_hz: float | None = None
    freq_count: int | None = None
    polar_min_angle_deg: float | None = None
    polar_max_angle_deg: float | None = None
    polar_step_deg: float | None = None
    polar_distance_m: float | None = None
    horizontal_norm_angle_deg: float | None = None
    vertical_norm_angle_deg: float | None = None


BARE_MESH_MODES = {"bare", "inner", "open"}
ENCLOSED_MESH_MODES = {"enclosure", "enclosed"}
FREESTANDING_MESH_MODES = {"free-standing", "freestanding", "free"}
INFINITE_BAFFLE_MESH_MODES = {"infinite-baffle", "infinitebaffle", "ib", "baffle"}
_ASSIGNMENT_RE = re.compile(r"^\s*(?P<key>[A-Za-z0-9_.:]+)\s*=\s*(?P<value>.*?)\s*$")
_POLAR_BLOCK_RE = re.compile(r"^\s*ABEC\.Polars:(?P<name>[^=\s]+)\s*=\s*\{\s*$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_ath_solve_settings(config_text: str) -> AthSolveSettings:
    top_level: dict[str, str] = {}
    polar_blocks = _parse_polar_blocks(config_text)

    current_polar: str | None = None
    for raw_line in config_text.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if current_polar is not None:
            if line.startswith("}"):
                current_polar = None
            continue
        block_match = _POLAR_BLOCK_RE.match(line)
        if block_match:
            current_polar = block_match.group("name")
            continue
        assignment = _ASSIGNMENT_RE.match(line)
        if assignment:
            top_level[assignment.group("key").strip().lower()] = assignment.group("value").strip()

    horizontal = _select_horizontal_polar(polar_blocks)
    vertical = _select_vertical_polar(polar_blocks)
    range_source = _first_polar_with_range(horizontal, vertical, *polar_blocks)
    distance_source = _first_polar_with_distance(horizontal, vertical, *polar_blocks)

    return AthSolveSettings(
        freq_min_hz=_parse_float(top_level.get("abec.f1")),
        freq_max_hz=_parse_float(top_level.get("abec.f2")),
        freq_count=_parse_positive_int(top_level.get("abec.numfrequencies")),
        polar_min_angle_deg=None if range_source is None else range_source.map_min_angle_deg,
        polar_max_angle_deg=None if range_source is None else range_source.map_max_angle_deg,
        polar_step_deg=None if range_source is None else range_source.map_step_deg,
        polar_distance_m=None if distance_source is None else distance_source.distance_m,
        horizontal_norm_angle_deg=None if horizontal is None else horizontal.norm_angle_deg,
        vertical_norm_angle_deg=None if vertical is None else vertical.norm_angle_deg,
    )


def native_check_open_edges_for_ath_config(config_text: str) -> bool:
    """Return whether native symmetric solves should enforce open-edge checks.

    HornLab Metal needs strict open-edge validation for closed reduced meshes.
    Only bare/open waveguide modes have a real mouth rim off the symmetry planes,
    so those are the only Ath-family configs that opt out.
    """
    return _parse_mesh_mode(config_text) != "bare"


def _parse_polar_blocks(config_text: str) -> tuple[AthPolarSettings, ...]:
    blocks: list[AthPolarSettings] = []
    current_name: str | None = None
    current_values: dict[str, str] = {}

    for raw_line in config_text.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if current_name is None:
            block_match = _POLAR_BLOCK_RE.match(line)
            if block_match:
                current_name = block_match.group("name").strip()
                current_values = {}
            continue
        if line.startswith("}"):
            blocks.append(_polar_settings_from_values(current_name, current_values))
            current_name = None
            current_values = {}
            continue
        assignment = _ASSIGNMENT_RE.match(line)
        if assignment:
            current_values[assignment.group("key").strip().lower()] = assignment.group("value").strip()

    return tuple(blocks)


def _parse_mesh_mode(config_text: str) -> str:
    top_level = _parse_top_level_assignments(config_text)
    raw_mode = _normalized_mode_value(_first_present(top_level, "mode", "mesh.mode"))
    if raw_mode in BARE_MESH_MODES:
        return "bare"
    if raw_mode in ENCLOSED_MESH_MODES or _positive_depth_present(top_level):
        return "enclosure"
    if raw_mode in INFINITE_BAFFLE_MESH_MODES:
        return "infinite-baffle"
    if raw_mode in FREESTANDING_MESH_MODES:
        return "freestanding"

    sim_type = _parse_positive_int(_first_present(top_level, "abec.simtype", "simtype", "sim_type", "mesh.simtype"))
    if sim_type == 1:
        return "infinite-baffle"
    return "freestanding"


def _parse_top_level_assignments(config_text: str) -> dict[str, str]:
    top_level: dict[str, str] = {}
    current_polar: str | None = None
    for raw_line in config_text.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if current_polar is not None:
            if line.startswith("}"):
                current_polar = None
            continue
        block_match = _POLAR_BLOCK_RE.match(line)
        if block_match:
            current_polar = block_match.group("name")
            continue
        assignment = _ASSIGNMENT_RE.match(line)
        if assignment:
            top_level[assignment.group("key").strip().lower()] = assignment.group("value").strip()
    return top_level


def _first_present(values: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _normalized_mode_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip('"').strip("'").lower().replace("_", "-")


def _positive_depth_present(values: dict[str, str]) -> bool:
    for key in ("enclosure.depth", "enclosure.depth_mm", "mesh.depth", "mesh.depth_mm", "encdepth", "depth_mm"):
        depth = _parse_float(values.get(key))
        if depth is not None and depth > 0.0:
            return True
    return False


def _polar_settings_from_values(name: str, values: dict[str, str]) -> AthPolarSettings:
    map_min, map_max, map_count = _parse_map_angle_range(values.get("mapanglerange"))
    return AthPolarSettings(
        name=name,
        distance_m=_parse_float(values.get("distance")),
        inclination_deg=_parse_float(values.get("inclination")),
        map_min_angle_deg=map_min,
        map_max_angle_deg=map_max,
        map_sample_count=map_count,
        norm_angle_deg=_parse_float(values.get("normangle")),
    )


def _select_horizontal_polar(polars: tuple[AthPolarSettings, ...]) -> AthPolarSettings | None:
    named = _first_matching_polar_name(polars, {"h", "hor", "horizontal", "splh", "spl_h"})
    if named is not None:
        return named
    for polar in polars:
        if polar.inclination_deg is None or abs(polar.inclination_deg) < 1e-6:
            return polar
    return polars[0] if polars else None


def _select_vertical_polar(polars: tuple[AthPolarSettings, ...]) -> AthPolarSettings | None:
    named = _first_matching_polar_name(polars, {"v", "vert", "vertical", "splv", "spl_v"})
    if named is not None:
        return named
    for polar in polars:
        if polar.inclination_deg is not None and abs(polar.inclination_deg - 90.0) < 1e-6:
            return polar
    return _select_horizontal_polar(polars)


def _first_matching_polar_name(
    polars: tuple[AthPolarSettings, ...],
    aliases: set[str],
) -> AthPolarSettings | None:
    for polar in polars:
        normalized = _normalized_polar_name(polar.name)
        if normalized in aliases:
            return polar
    return None


def _first_polar_with_range(*polars: AthPolarSettings | None) -> AthPolarSettings | None:
    for polar in polars:
        if polar is not None and polar.map_step_deg is not None:
            return polar
    return None


def _first_polar_with_distance(*polars: AthPolarSettings | None) -> AthPolarSettings | None:
    for polar in polars:
        if polar is not None and polar.distance_m is not None:
            return polar
    return None


def _parse_map_angle_range(value: str | None) -> tuple[float | None, float | None, int | None]:
    if value is None:
        return None, None, None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) < 3:
        return None, None, None
    start = _parse_float(parts[0])
    stop = _parse_float(parts[1])
    count = _parse_positive_int(parts[2])
    if start is None or stop is None or count is None:
        return None, None, None
    return min(start, stop), max(start, stop), count


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    match = _NUMBER_RE.search(value.strip().strip('"'))
    if match is None:
        return None
    return float(match.group(0))


def _parse_positive_int(value: str | None) -> int | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    rounded = int(round(parsed))
    return rounded if rounded > 0 else None


def _strip_comment(line: str) -> str:
    return line.split(";", 1)[0]


def _normalized_polar_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())
