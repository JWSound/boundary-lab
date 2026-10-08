"""The one filesystem anchor for bundled resources.

Every module that needs the repository root imports :data:`APP_ROOT` from here.
Deriving it locally with ``Path(__file__).resolve().parents[N]`` is how every
icon path once broke silently: a module moved one directory deeper, ``N`` was
no longer right, and the paths resolved into ``src/``. Nothing raised, because
a missing icon degrades to a null ``QIcon`` and a missing asset is only ever
read lazily. Four copies of that expression had accumulated, each with a
different ``N``.

``test_no_module_derives_its_own_app_root`` keeps it at one.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository root. This module is ``src/blab/paths.py``, so the root is two
#: levels up. Nothing else in the tree may compute this.
APP_ROOT = (
    Path(os.environ["BLAB_RESOURCE_ROOT"])
    if os.environ.get("BLAB_RESOURCE_ROOT")
    else Path(__file__).resolve().parents[2]
)

#: Bundled icons, images and SVGs.
ASSETS_DIR = APP_ROOT / "assets"

# Packaged launches set this to the Windows Known Folder Documents location.
# Source checkouts keep their existing working directories.
USER_ROOT = Path(os.environ["BLAB_USER_ROOT"]) if os.environ.get("BLAB_USER_ROOT") else None
RUNS_ROOT = USER_ROOT / "runs" if USER_ROOT is not None else APP_ROOT / "runs"
DOCS_ROOT = USER_ROOT / "documentation" if USER_ROOT is not None else APP_ROOT / "docs"


def working_runs_root() -> Path:
    return USER_ROOT / "runs" if USER_ROOT is not None else Path.cwd() / "runs"


def asset(name: str) -> Path:
    """Path to a bundled asset by file name."""
    return ASSETS_DIR / name
