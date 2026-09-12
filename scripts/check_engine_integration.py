"""Qualify the installed engine through Boundary Lab's headless project workflow."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from beat_engine import engine_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/engine-integration"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    request = args.output / "smoke-request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frequencies_hz": [500.0],
                "include_project_observations": False,
                "probes": [{"id": "on_axis", "coordinate_frame": "project", "points_m": [[0, 0, 2]]}],
                "retain": ["bem_boundary_traces"],
            }
        ),
        encoding="utf-8",
    )
    common = ["examples/Simple_Sealed/simple_sealed.blab.json", "--backend", "beat_cpu", "--request", str(request)]
    cli = [sys.executable, "-m", "blab.cli", "project"]
    subprocess.run(cli + ["validate"] + common + ["--json"], check=True)
    output = args.output / "result"
    subprocess.run(cli + ["solve"] + common + ["--julia-threads", "2", "--output", str(output)], check=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(manifest["completion_mask"]), manifest["status"]
    assert manifest["engine_runs"], "Missing engine provenance"
    for run in manifest["engine_runs"]:
        assert run["engine"]["version"] == "0.1.2", run
        assert len(run["engine"]["source_sha256"]) == 64, run
        assert Path(run["runtime"]["project_file"]).resolve() == (engine_paths().project / "Project.toml").resolve()
    with np.load(output / "frequencies/000000.npz") as result:
        assert result.files, "No result quantities"
        assert any(np.iscomplexobj(result[key]) for key in result.files), "Complex results lost"
        for key in result.files:
            assert np.isfinite(result[key]).all(), key
    print("Installed-engine integration passed:", engine_paths().project)


if __name__ == "__main__":
    main()
