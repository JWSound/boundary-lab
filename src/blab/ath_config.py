"""Small reader for the Ath ``.cfg`` metadata Boundary Lab needs before solving."""

from __future__ import annotations

import re

BARE_MESH_MODES = {"bare", "inner", "open"}
ENCLOSED_MESH_MODES = {"enclosure", "enclosed"}
FREESTANDING_MESH_MODES = {"free-standing", "freestanding", "free"}
INFINITE_BAFFLE_MESH_MODES = {"infinite-baffle", "infinitebaffle", "ib", "baffle"}
_ASSIGNMENT_RE = re.compile(r"^\s*(?P<key>[A-Za-z0-9_.:]+)\s*=\s*(?P<value>.*?)\s*$")
_POLAR_BLOCK_RE = re.compile(r"^\s*ABEC\.Polars:(?P<name>[^=\s]+)\s*=\s*\{\s*$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_ENCLOSURE_DEPTH_KEYS = (
    "enclosure.depth",
    "enclosure.depth_mm",
    "mesh.depth",
    "mesh.depth_mm",
    "encdepth",
    "depth_mm",
)


def native_check_open_edges_for_ath_config(config_text: str) -> bool:
    """Return whether a native symmetric solve should enforce open-edge checks.

    Backends with native symmetry validate that a reduced mesh's open edges all
    lie on a symmetry plane. Only bare/open waveguide modes have a real mouth rim
    away from those planes, so they are the only Ath-family configs that opt out.
    """
    return _parse_mesh_mode(config_text) != "bare"


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
    """Collect ``key = value`` pairs outside ``ABEC.Polars:<name> = { ... }`` blocks."""
    top_level: dict[str, str] = {}
    inside_polar_block = False
    for raw_line in config_text.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if inside_polar_block:
            inside_polar_block = not line.startswith("}")
            continue
        if _POLAR_BLOCK_RE.match(line):
            inside_polar_block = True
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
    for key in _ENCLOSURE_DEPTH_KEYS:
        depth = _parse_float(values.get(key))
        if depth is not None and depth > 0.0:
            return True
    return False


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
