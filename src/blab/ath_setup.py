"""Qt-free validation and discovery for user-installed Ath/Wine on macOS."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ATH_PATH_KEY = "macos/ath_executable"
WINE_PATH_KEY = "macos/wine_executable"


class AthSetupError(ValueError):
    """The optional external generator tools have not been configured."""


def discover_wine() -> str:
    candidates = [shutil.which("wine"), "/opt/homebrew/bin/wine", "/usr/local/bin/wine"]
    for root in (Path("/Applications"), Path.home() / "Applications"):
        for name in ("Wine Stable.app", "Wine Devel.app", "Wine Staging.app"):
            candidates.append(str(root / name / "Contents/Resources/wine/bin/wine"))
    return next((p for p in candidates if p and Path(p).is_file() and os.access(p, os.X_OK)), "")


def validate_paths(ath: str, wine: str) -> tuple[Path, Path]:
    if not ath.strip() or not wine.strip():
        raise AthSetupError("Ath setup required. Open Edit → Ath Setup… and select Ath and Wine.")
    ath_path, wine_path = Path(ath).expanduser().resolve(), Path(wine).expanduser().resolve()
    if not ath_path.is_file() or ath_path.suffix.lower() != ".exe":
        raise AthSetupError("Select the downloaded Ath .exe file.")
    if not wine_path.is_file() or not os.access(wine_path, os.X_OK):
        raise AthSetupError("Select an executable Wine program, not its .app folder.")
    return ath_path, wine_path


def wine_prefix() -> Path:
    return Path.home() / "Library/Application Support/Boundary Lab/ath-wine"


# Small geometry-only probe: the same batch interface used for normal generation.
TEST_CONFIG = """Throat.Profile = 1
Throat.Diameter = 25.4
Throat.Angle = 0
Length = 50
Coverage.Angle = 45
Term.s = 0.5
Term.q = 0.99
Term.n = 4
OS.k = 1
Mesh.AngularSegments = 16
Mesh.LengthSegments = 4
Mesh.Quadrants = 1
ABEC.SimType = 1
"""
