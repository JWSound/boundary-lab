"""Compare a real local CPU solve with a separate HTTP server process."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from blab.remote import RemoteBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/remote-integration"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    request = args.output / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frequencies_hz": [500.0],
                "include_project_observations": False,
                "probes": [{"id": "on_axis", "coordinate_frame": "project", "points_m": [[0, 0, 2]]}],
                "retain": ["bem_boundary_traces"],
            }
        )
    )
    cli = [sys.executable, "-m", "blab.cli"]
    common = ["examples/Simple_Sealed/simple_sealed.blab.json", "--backend", "beat_cpu", "--request", str(request)]
    subprocess.run(cli + ["project", "validate"] + common + ["--json"], check=True)
    subprocess.run(
        cli + ["project", "solve"] + common + ["--julia-threads", "2", "--output", str(args.output / "local")],
        check=True,
    )
    log_path = args.output / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cli
            + [
                "server",
                "--root",
                str(args.output / "jobs"),
                "--port",
                "0",
                "--julia-threads",
                "2",
                "--shutdown-on-stdin-eof",
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            url = None
            while time.monotonic() < deadline:
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("Physical-system server listening on "):
                        url = line.split()[-1]
                if url:
                    break
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text())
                time.sleep(0.1)
            if url is None:
                raise RuntimeError("Server startup timed out.")
            RemoteBackend(url).check_capabilities()
            subprocess.run(
                cli + ["project", "solve"] + common + ["--server-url", url, "--output", str(args.output / "remote")],
                check=True,
            )
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=30)
    manifests = [json.loads((args.output / mode / "manifest.json").read_text()) for mode in ("local", "remote")]
    assert all(item["status"] == "complete" and all(item["completion_mask"]) for item in manifests)
    assert manifests[1]["remote_job_id"]
    assert all(item["engine_runs"] and item["worker"] for item in manifests)
    assert manifests[0]["excitation_port_ids"] == manifests[1]["excitation_port_ids"]
    assert manifests[0]["engine_runs"][0]["engine"] == manifests[1]["engine_runs"][0]["engine"]
    with (
        np.load(args.output / "local/frequencies/000000.npz") as local,
        np.load(args.output / "remote/frequencies/000000.npz") as remote,
    ):
        assert local.files == remote.files
        assert any(np.iscomplexobj(local[key]) for key in local.files)
        for key in local.files:
            assert np.isfinite(remote[key]).all()
            np.testing.assert_array_equal(local[key], remote[key])
        quantity_count = len(local.files)
    for mode in ("local", "remote"):
        metadata = json.loads((args.output / mode / "frequencies/000000.json").read_text())
        if mode == "local":
            local_metadata = metadata
        else:
            assert metadata["quantities"] == local_metadata["quantities"]
    report = {
        "status": "passed",
        "complex_arrays_equal": quantity_count,
        "remote_job_id": manifests[1]["remote_job_id"],
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
