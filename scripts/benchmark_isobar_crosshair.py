from __future__ import annotations

import argparse
import os
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from blab.ui.plots import FINAL_ISOBAR_ANGLE_SAMPLES, FINAL_ISOBAR_FREQ_SAMPLES, IsobarCanvas


def _synthetic_values(angles_deg: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    angle_component = np.sin(np.deg2rad(angles_deg * 3.0))[:, np.newaxis] * 8.0
    log_freq = np.log10(freqs_hz / freqs_hz[0])
    freq_component = np.cos(log_freq * np.pi * 3.0)[np.newaxis, :] * 5.0
    ripple = np.sin(np.deg2rad(angles_deg))[:, np.newaxis] * np.sin(log_freq * np.pi * 8.0)[np.newaxis, :] * 3.0
    return np.clip(angle_component + freq_component + ripple - 12.0, -30.0, 0.0).astype(np.float32)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Boundary Lab isobar crosshair updates.")
    parser.add_argument("--updates", type=int, default=500, help="Number of synthetic cursor positions.")
    parser.add_argument("--freq-samples", type=int, default=FINAL_ISOBAR_FREQ_SAMPLES)
    parser.add_argument("--angle-samples", type=int, default=FINAL_ISOBAR_ANGLE_SAMPLES)
    parser.add_argument("--shading", choices=("gouraud", "nearest"), default="gouraud")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    app = QApplication.instance() or QApplication([])
    canvas = IsobarCanvas("Crosshair Benchmark")

    freqs_hz = np.geomspace(20.0, 20000.0, max(2, args.freq_samples)).astype(np.float32)
    angles_deg = np.linspace(-180.0, 180.0, max(2, args.angle_samples)).astype(np.float32)
    values_db = _synthetic_values(angles_deg, freqs_hz)

    canvas.update_plot(freqs_hz, angles_deg, values_db, -30.0, 0.0, shading=args.shading)
    canvas.draw()
    app.processEvents()

    cursor_freqs = np.geomspace(25.0, 18000.0, max(1, args.updates))
    cursor_angles = np.linspace(-170.0, 170.0, max(1, args.updates))

    start = time.perf_counter()
    for freq_hz, angle_deg in zip(cursor_freqs, cursor_angles):
        canvas._set_crosshair_position(float(freq_hz), float(angle_deg))
        app.processEvents()
    elapsed = time.perf_counter() - start

    updates = max(1, args.updates)
    print(f"updates: {updates}")
    print(f"grid: {angles_deg.size} angles x {freqs_hz.size} freqs")
    print(f"shading: {args.shading}")
    print(f"total_ms: {elapsed * 1000.0:.3f}")
    print(f"ms_per_update: {elapsed * 1000.0 / updates:.3f}")
    print(f"updates_per_second: {updates / elapsed:.1f}")


if __name__ == "__main__":
    main()
