"""Isolated Gmsh process: ``worker.py job.json`` runs a job :mod:`jobs` wrote.

Writes ``result.json``, or ``error.json`` with the message of a pipeline rejection. Runs with ``-I``, so the job
names the ``blab`` source folder of the host process and the plugin folder.
"""

import json
import sys
from pathlib import Path

RESULT, ERROR = "result.json", "error.json"  # what the worker writes in the output folder


def main(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    sys.path[:0] = [job["plugin_path"], job["blab_path"]]
    from cadimport.pipeline import CadImportSettings, mesh_step, open_step, preview_to_json, result_to_json

    output = Path(job["output"])
    try:
        if job["mode"] == "preview":
            placement = dict(rotation_deg=job["rotation_deg"], translation_m=job["translation_m"])
            result = preview_to_json(open_step(job["step_path"], **placement, closed_shells=job["closed_shells"]))
        else:
            result = result_to_json(mesh_step(job["step_path"], CadImportSettings(**job["settings"]), output))
    except ValueError as exc:
        (output / ERROR).write_text(json.dumps({"message": str(exc)}), encoding="utf-8")
        return 1
    (output / RESULT).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
