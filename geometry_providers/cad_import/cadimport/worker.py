"""Isolated Gmsh process: ``worker.py job.json`` meshes or previews a STEP file.

Runs with ``-I``, so the job names the ``blab`` source folder of the host process.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    sys.path.insert(0, job["blab_path"])
    from blab.cad_import import CadImportSettings, mesh_step, open_step, preview_to_json

    output = Path(job["output"])
    if job["mode"] == "preview":
        # Only the quick scan while a closed shell has no choice: the answers decide the full read.
        preview = open_step(
            job["step_path"],
            rotation_deg=job["rotation_deg"],
            translation_m=job["translation_m"],
            closed_shells=job["closed_shells"],
        )
        (output / "preview.json").write_text(json.dumps(preview_to_json(preview)), encoding="utf-8")
        return
    result = mesh_step(job["step_path"], CadImportSettings(**job["settings"]), output)
    info = {
        "target_size_m": result.target_size_m,
        "geometry": asdict(result.geometry),
        "meshes": [
            {
                "id": mesh.id,
                "purpose": mesh.purpose,
                "file": mesh.path.name,
                "volume_groups": list(mesh.volume_groups),
                "surface_groups": list(mesh.surface_groups),
            }
            for mesh in result.meshes
        ],
    }
    (output / "mesh_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
