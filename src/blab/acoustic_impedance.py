"""Acoustic-impedance normalization metadata and numerical helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ACOUSTIC_IMPEDANCE_NORMALIZATION_METADATA_KEY = "acoustic_impedance_normalization"
ACOUSTIC_AREA_MISMATCH_WARNING_THRESHOLD = 0.10


@dataclass(frozen=True)
class AcousticImpedanceNormalization:
    """Effective driven area and optional opposing-side diagnostics."""

    component_id: str
    component_name: str
    effective_area_m2: float
    area_kind: str
    positive_side_area_m2: float = 0.0
    negative_side_area_m2: float = 0.0
    relative_side_mismatch: float | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "component_name": self.component_name,
            "effective_area_m2": self.effective_area_m2,
            "area_kind": self.area_kind,
            "positive_side_area_m2": self.positive_side_area_m2,
            "negative_side_area_m2": self.negative_side_area_m2,
            "relative_side_mismatch": self.relative_side_mismatch,
        }

    @classmethod
    def from_metadata(cls, component_id: str, raw: object) -> AcousticImpedanceNormalization:
        if not isinstance(raw, dict):
            raise ValueError(f"Acoustic normalization metadata for '{component_id}' must be an object.")
        mismatch = raw.get("relative_side_mismatch")
        record = cls(
            component_id=str(component_id),
            component_name=str(raw.get("component_name", component_id)),
            effective_area_m2=float(raw["effective_area_m2"]),
            area_kind=str(raw.get("area_kind", "unknown")),
            positive_side_area_m2=float(raw.get("positive_side_area_m2", 0.0)),
            negative_side_area_m2=float(raw.get("negative_side_area_m2", 0.0)),
            relative_side_mismatch=None if mismatch is None else float(mismatch),
        )
        if not np.isfinite(record.effective_area_m2) or record.effective_area_m2 <= 0.0:
            raise ValueError(f"Acoustic normalization area for '{component_id}' must be positive and finite.")
        return record


def normalization_records(metadata: object) -> dict[str, AcousticImpedanceNormalization]:
    """Decode compiler-produced normalization records from system metadata."""

    if not isinstance(metadata, dict):
        return {}
    raw_records = metadata.get(ACOUSTIC_IMPEDANCE_NORMALIZATION_METADATA_KEY, {})
    if not isinstance(raw_records, dict):
        return {}
    return {
        str(component_id): AcousticImpedanceNormalization.from_metadata(str(component_id), raw)
        for component_id, raw in raw_records.items()
    }


def normalize_generalized_impedance(
    values: np.ndarray,
    effective_area_m2: np.ndarray,
    density_kg_per_m3: float,
    sound_speed_m_per_s: float,
) -> np.ndarray:
    """Return dimensionless ``Z / (rho * c * Sd)`` for component-major values."""

    impedance = np.asarray(values)
    areas = np.asarray(effective_area_m2, dtype=float)
    density = float(density_kg_per_m3)
    sound_speed = float(sound_speed_m_per_s)
    if impedance.ndim < 1 or areas.ndim != 1 or impedance.shape[0] != areas.size:
        raise ValueError("Acoustic impedance and effective-area component axes must match.")
    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ValueError("Acoustic impedance effective areas must be positive and finite.")
    if not np.isfinite(density) or density <= 0.0 or not np.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("Acoustic impedance normalization requires positive finite density and sound speed.")
    scale = density * sound_speed * areas
    return impedance / scale.reshape((areas.size, *([1] * (impedance.ndim - 1))))


__all__ = [
    "ACOUSTIC_AREA_MISMATCH_WARNING_THRESHOLD",
    "ACOUSTIC_IMPEDANCE_NORMALIZATION_METADATA_KEY",
    "AcousticImpedanceNormalization",
    "normalization_records",
    "normalize_generalized_impedance",
]
