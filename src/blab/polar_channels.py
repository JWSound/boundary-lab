"""Retained, DSP-weighted channel pressures for polar display and export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolarChannelBasis:
    freq_hz: np.ndarray
    angle_deg: np.ndarray
    names: tuple[str, ...]
    horizontal: np.ndarray
    vertical: np.ndarray

    def snapshot(self) -> PolarChannelBasis:
        return PolarChannelBasis(
            self.freq_hz.copy(),
            self.angle_deg.copy(),
            self.names,
            self.horizontal.copy(),
            self.vertical.copy(),
        )

    def sum(self, selected: frozenset[str]) -> tuple[np.ndarray, np.ndarray]:
        """Sum complex pressures, preserving interference between selected channels."""
        mask = np.asarray([name in selected for name in self.names], dtype=bool)
        if not mask.any():
            raise ValueError("No polar channels selected.")
        return self.horizontal[:, mask].sum(axis=1), self.vertical[:, mask].sum(axis=1)


def finite_pressure_spl(pressure: np.ndarray) -> np.ndarray:
    """Keep exact nulls finite for normalization and interpolation."""
    return 20.0 * np.log10(np.maximum(np.abs(pressure), 1e-12) / 2e-5)
