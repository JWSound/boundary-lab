"""Explicit conversions between solver-native and standard audio phasors."""

from __future__ import annotations

import numpy as np

LEGACY_PHASOR_CONVENTION = "exp(-i omega t)"
SOLVER_PHASOR_CONVENTION = "exp(+i omega t)"
STANDARD_AUDIO_PHASOR_CONVENTION = "exp(+i omega t)"


def convert_phasor(values, source: str, target: str = STANDARD_AUDIO_PHASOR_CONVENTION):
    """Convert explicitly labelled complex data, exactly once at ingestion."""
    supported = {LEGACY_PHASOR_CONVENTION, STANDARD_AUDIO_PHASOR_CONVENTION}
    if source not in supported or target not in supported:
        raise ValueError(f"Unsupported phasor conversion: {source!r} to {target!r}.")
    return np.conjugate(values) if source != target else np.asarray(values)


def solver_to_standard_phasor(values):
    """Canonical solver data already uses positive-time audio phasors."""
    return np.asarray(values)


def standard_to_solver_phasor(values):
    """The requested BEAT convention is the positive-time audio convention."""

    return np.asarray(values)


def solver_phase_deg(values: np.ndarray) -> np.ndarray:
    """Return standard-audio phase angles for solver-native phasors."""

    standard = solver_to_standard_phasor(np.asarray(values))
    phase = np.rad2deg(np.angle(standard)).astype(np.float32, copy=False)
    return np.where(np.isclose(phase, 0.0, atol=1.0e-6), 0.0, phase).astype(
        np.float32,
        copy=False,
    )


__all__ = [
    "LEGACY_PHASOR_CONVENTION",
    "convert_phasor",
    "SOLVER_PHASOR_CONVENTION",
    "STANDARD_AUDIO_PHASOR_CONVENTION",
    "solver_phase_deg",
    "solver_to_standard_phasor",
    "standard_to_solver_phasor",
]
