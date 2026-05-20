"""Shared default paths for the command-line workflow."""

from __future__ import annotations

from pathlib import Path


EXAMPLE_MESH_PATH = Path("waveguide.msh")
EXAMPLE_CLEAN_MESH_PATH = Path("waveguide_clean.msh")

SOLVER_OUTPUT_NPZ_BASE = Path("pressure_data_raw")
SOLVER_OUTPUT_NPZ = Path(f"{SOLVER_OUTPUT_NPZ_BASE}.npz")
FORMATTED_OUTPUT_NPZ = Path("pressure_data_formatted.npz")

PLOT_OUTPUT_DIR = Path(".")