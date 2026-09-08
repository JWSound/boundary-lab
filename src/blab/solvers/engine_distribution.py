"""The required, independently released BEAT Engine dependency."""

try:
    import beat_engine
except ImportError as exc:
    raise RuntimeError("BEAT Engine is not installed. Reinstall Boundary Lab with its declared dependencies.") from exc

if beat_engine.__version__ != "0.1.0rc1":
    raise RuntimeError(f"Boundary Lab requires beat-engine 0.1.0rc1; found {beat_engine.__version__}.")

from beat_engine import engine_paths as engine_paths
