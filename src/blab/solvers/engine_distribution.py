"""The required, independently released BEAT Engine dependency."""

try:
    import beat_engine
except ImportError as exc:
    raise RuntimeError("BEAT Engine is not installed. Reinstall Boundary Lab with its declared dependencies.") from exc

if beat_engine.__version__ != "0.1.3":
    raise RuntimeError(f"Boundary Lab requires beat-engine 0.1.3; found {beat_engine.__version__}.")

from beat_engine import backend_catalog as backend_catalog
from beat_engine import backend_info as engine_backend_info
from beat_engine import engine_paths as engine_paths

__all__ = ["backend_catalog", "engine_backend_info", "engine_paths"]
