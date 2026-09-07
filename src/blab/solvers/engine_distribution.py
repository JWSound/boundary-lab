"""Explicit, reversible selection while the first BEAT release is qualified."""

import os

ENGINE_DISTRIBUTION = os.environ.get("BLAB_BEAT_ENGINE_DISTRIBUTION", "bundled").strip().lower()
if ENGINE_DISTRIBUTION not in {"bundled", "external"}:
    raise ValueError("BLAB_BEAT_ENGINE_DISTRIBUTION must be bundled or external.")

if ENGINE_DISTRIBUTION == "external":
    try:
        import beat_engine
    except ImportError as exc:
        raise RuntimeError(
            "External BEAT was selected but is not installed. Install the beat-engine 0.1.0rc1 candidate."
        ) from exc
    if beat_engine.__version__ != "0.1.0rc1":
        raise RuntimeError(
            f"This Boundary Lab integration requires beat-engine 0.1.0rc1; found {beat_engine.__version__}."
        )
