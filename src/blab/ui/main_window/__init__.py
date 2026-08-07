"""Main Qt window and user workflow orchestration."""

from __future__ import annotations

from blab.ui.main_window.constants import (
    ADD_DESIGN_TAB_LABEL,
    APP_ROOT,
    ATH_BUNDLE_DIR,
    BEM_PREVIEW_DARK_ICON,
    BEM_PREVIEW_LIGHT_ICON,
    CAPTURE_CONTOURS_DARK_ICON,
    CAPTURE_CONTOURS_LIGHT_ICON,
    CLEAR_CONTOURS_DARK_ICON,
    CLEAR_CONTOURS_LIGHT_ICON,
    DEFAULT_DOCK_STATE_B64,
    DEFAULT_MESH_SCALE_FACTOR,
    FEM_PREVIEW_DARK_ICON,
    FEM_PREVIEW_LIGHT_ICON,
    GENERATED_GEOMETRY_ROOT,
    GMSH_BUNDLE_EXE,
    HELP_GUIDE_PDF,
    LIVE_PLOT_REFRESH_INTERVAL_MS,
    SAVE_DARK_ICON,
    SAVE_LIGHT_ICON,
)
from blab.ui.main_window.helpers import (
    _mesh_entries_with_file_overrides,
    _physical_system_preview_metadata,
)
from blab.ui.main_window.window import MainWindow
from blab.ui.main_window_widgets import AthScriptEditor, DockTitleBar, PlotEntry
from blab.ui.mesh_assembly import STITCH_FAILURE_MESSAGE, STITCHED_MESH_NAME

__all__ = [
    "AthScriptEditor",
    "DockTitleBar",
    "PlotEntry",
    "MainWindow",
    "STITCH_FAILURE_MESSAGE",
    "STITCHED_MESH_NAME",
    "_mesh_entries_with_file_overrides",
    "_physical_system_preview_metadata",
    "ADD_DESIGN_TAB_LABEL",
    "APP_ROOT",
    "ATH_BUNDLE_DIR",
    "BEM_PREVIEW_DARK_ICON",
    "BEM_PREVIEW_LIGHT_ICON",
    "CAPTURE_CONTOURS_DARK_ICON",
    "CAPTURE_CONTOURS_LIGHT_ICON",
    "CLEAR_CONTOURS_DARK_ICON",
    "CLEAR_CONTOURS_LIGHT_ICON",
    "DEFAULT_DOCK_STATE_B64",
    "DEFAULT_MESH_SCALE_FACTOR",
    "FEM_PREVIEW_DARK_ICON",
    "FEM_PREVIEW_LIGHT_ICON",
    "GENERATED_GEOMETRY_ROOT",
    "GMSH_BUNDLE_EXE",
    "HELP_GUIDE_PDF",
    "LIVE_PLOT_REFRESH_INTERVAL_MS",
    "SAVE_DARK_ICON",
    "SAVE_LIGHT_ICON",
]
