"""The CAD import plugin's modules, obtained through the plugin loader like the Vented Enclosure tests do.

``from cad_plugin import CadImportSettings`` reads a name from the plugin package; ``module("sizing")`` is one of its
submodules.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from blab.generators.catalog import ProviderCatalog

ROOT = Path(__file__).resolve().parents[1] / "geometry_providers"
PROVIDER = "blab.cad_import"
PLUGIN_DIR = ROOT / "cad_import"


def _load():
    catalog = ProviderCatalog([ROOT], enabled=[PROVIDER])
    catalog.scan()
    backend = catalog.factory(PROVIDER, "backend")()
    return importlib.import_module(importlib.import_module(type(backend).__module__).__package__)


package = _load()


def module(name: str):
    """The plugin submodule ``name``, for example ``module("sizing")``."""
    return importlib.import_module(f"{package.__name__}.{name}")


def __getattr__(name: str):
    try:
        return getattr(package, name)
    except AttributeError:
        raise AttributeError(f"module 'cad_plugin' has no attribute {name!r}") from None
