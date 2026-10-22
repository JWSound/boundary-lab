"""The STEP file of a CAD import design source; the rest of the source is read by ``blab.cad_import.design``."""

from pathlib import Path


def step_path(source):
    path = str(source.get("step_path") or "")
    if not path:
        raise ValueError("Choose a STEP file first.")
    if not Path(path).is_file():
        raise ValueError(f"STEP file not found: {path}")
    return Path(path)
