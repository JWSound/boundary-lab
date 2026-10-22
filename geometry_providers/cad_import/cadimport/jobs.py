"""The worker jobs, preview and mesh: each runs ``worker.py`` in its own process and returns its result typed.

A rejection by the pipeline (``ValueError``) comes back as ``ValueError`` with the same message.
"""

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import blab
from blab.cad_import import CadImportResult, CadImportSettings, CadPreview, preview_from_json, result_from_json
from blab.generators.base import GenerationCancelledError

from .worker import ERROR, RESULT

BLAB_PATH = str(Path(blab.__file__).resolve().parents[1])
WORKER = Path(__file__).with_name("worker.py")


def preview_in_worker(step_path, output, *, rotation_deg, translation_m, closed_shells, cancelled) -> CadPreview:
    """Read ``step_path`` for display with that placement and those closed shell choices."""
    job = dict(
        mode="preview",
        step_path=str(step_path),
        rotation_deg=list(rotation_deg),
        translation_m=list(translation_m),
        closed_shells=dict(closed_shells),
    )
    return preview_from_json(_run(job, output, cancelled=cancelled))


def mesh_in_worker(
    step_path, settings: CadImportSettings, output, *, cancelled, started=lambda process: None, elapsed=lambda s: None
) -> CadImportResult:
    """Mesh ``step_path`` into ``output``; ``started`` gets the worker process, then ``None`` once it ended,
    and ``elapsed`` the seconds it has been running, polled about every 0.1 s."""
    job = dict(mode="mesh", step_path=str(step_path), settings=asdict(settings))
    return result_from_json(_run(job, output, cancelled=cancelled, started=started, elapsed=elapsed))


def _run(job, output, *, cancelled, started=lambda process: None, elapsed=lambda s: None):
    """Run ``worker.py`` on ``job`` in ``output`` and return its JSON result; ``cancelled()`` is polled and
    terminates it."""
    job_path = output / "job.json"
    job_path.write_text(json.dumps(job | {"blab_path": BLAB_PATH, "output": str(output)}, indent=2), encoding="utf-8")
    log_path = output / "meshing.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-X", "utf8", str(WORKER), str(job_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        started(process)
        begun = time.monotonic()
        try:
            while True:
                if cancelled():
                    raise GenerationCancelledError("CAD import cancelled.")
                elapsed(time.monotonic() - begun)
                try:
                    code = process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()
            started(None)
    if (output / ERROR).is_file():
        raise ValueError(json.loads((output / ERROR).read_text(encoding="utf-8"))["message"])
    if code:
        details = log_path.read_text(encoding="utf-8")
        raise ValueError(f"CAD import meshing failed. Details: {log_path}\n{details[-2000:]}")
    return json.loads((output / RESULT).read_text(encoding="utf-8"))
