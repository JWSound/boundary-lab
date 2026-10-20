"""Manufacturer-independent driver parameter text interchange.

Values use SI except Sd (cm²), Mmd (g), and inductances (mH).
Repeated Le and Rms labels belong to separate model sections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

BASE_FIELDS = (
    ("Bl", "bl_n_per_a", 1.0),
    ("Cms", "cms_m_per_n", 1.0),
    ("Rms", "rms_n_s_per_m", 1.0),
    ("Mmd", "mmd_kg", 0.001),
    ("Le", "le_h", 0.001),
    ("Re", "re_ohm", 1.0),
)
SEMI_FIELDS = (
    ("Re'", "re_prime_ohm", 1.0),
    ("Leb", "leb_h", 0.001),
    ("Le", "le_h", 0.001),
    ("Ke", "ke_semi_h", 1.0),
    ("Rss", "rss_ohm", 1.0),
)


@dataclass(frozen=True)
class DriverText:
    name: str
    parameters: dict
    sd_m2: float | None = None


def _number(text: str, label: str) -> float:
    try:
        value = float(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a number using a decimal point.") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return value


def parse_driver_text(text: str) -> DriverText:
    """Parse and validate the entire file before the editor applies any values."""
    lines = [line.strip() for line in text.lstrip("\ufeff").splitlines() if line.strip()]
    if not lines or "=" in lines[0]:
        raise ValueError("The first line must contain the driver name.")
    name = lines.pop(0)
    base: dict[str, float] = {}
    semi: dict[str, float] = {}
    extras: dict[str, float] = {}
    in_semi = False
    base_fields = {label: (key, scale) for label, key, scale in BASE_FIELDS}
    semi_fields = {label: (key, scale) for label, key, scale in SEMI_FIELDS}
    for line in lines:
        label, separator, raw = line.partition("=")
        label = label.strip()
        if not separator:
            raise ValueError(f"Expected parameter=value: {line}")
        if label in {"Pmax", "Xmax"}:
            continue  # Limits belong to channel configuration, not the driver.
        if label == "Mms":
            raise ValueError("Mms is not Mmd. Supply the dry moving mass as Mmd.")
        if label == "Re'":
            in_semi = True
        fields = semi_fields if in_semi else base_fields
        values = semi if in_semi else base
        if label in fields:
            key, scale = fields[label]
            if key in values:
                raise ValueError(f"Duplicate {label} in the same parameter section.")
            values[key] = _number(raw, label) * scale
        elif label in {"Sd", "SemiInductance", "LossyLe", "Damping", "Ams"} or (in_semi and label == "Rms"):
            if label in extras:
                raise ValueError(f"Duplicate {label}.")
            extras[label] = _number(raw, label)
        else:
            raise ValueError(f"Unsupported driver parameter: {label}")
    for label, key, _scale in BASE_FIELDS:
        if key not in base:
            raise ValueError(f"Missing driver parameter: {label}")
        if key not in {"le_h", "rms_n_s_per_m"} and base[key] <= 0:
            raise ValueError(f"{label} must be greater than zero.")
    for flag in ("SemiInductance", "LossyLe", "Damping"):
        if extras.get(flag, 0) not in (0, 1):
            raise ValueError(f"{flag} must be 0 or 1.")
    if extras.get("LossyLe", 0) or extras.get("Damping", 0):
        raise ValueError("Enabled LossyLe and Damping models are not supported.")
    if semi and "SemiInductance" not in extras:
        raise ValueError("The semi-inductance section requires a SemiInductance=0 or 1 flag.")
    enabled = bool(extras.get("SemiInductance", 0))
    if enabled:
        for label, key, _scale in SEMI_FIELDS:
            if semi.get(key, 0) <= 0:
                raise ValueError(f"Semi-inductance {label} must be greater than zero.")
    # Disabled zero placeholders are not editable model values.
    semi = {key: value for key, value in semi.items() if value > 0}
    if semi or enabled:
        base["semi_inductance"] = {"enabled": enabled, **semi}
    sd = extras.get("Sd")
    if sd is not None and sd <= 0:
        raise ValueError("Sd must be greater than zero.")
    return DriverText(name, base, None if sd is None else sd * 0.0001)


def format_driver_text(driver: DriverText) -> str:
    """Write supported driver values with stable, locale-independent units."""
    if not driver.name.strip() or any(char in driver.name for char in "\r\n="):
        raise ValueError("The driver name must be a single line without '='.")
    lines = [driver.name.strip(), ""]
    if driver.sd_m2 is not None:
        lines.append(f"Sd={driver.sd_m2 / 0.0001:.12g}")
    for label, key, scale in BASE_FIELDS:
        if key not in driver.parameters:
            raise ValueError(f"Missing driver parameter: {label}")
        lines.append(f"{label}={driver.parameters[key] / scale:.12g}")
    semi = driver.parameters.get("semi_inductance") or {}
    for label, key, scale in SEMI_FIELDS:
        lines.append(f"{label}={semi.get(key, 0) / scale:.12g}")
    lines.extend(("Rms=0", "Ams=0", "LossyLe=0", f"SemiInductance={int(bool(semi.get('enabled')))}", "Damping=0"))
    text = "\n".join(lines) + "\n"
    parse_driver_text(text)
    return text
