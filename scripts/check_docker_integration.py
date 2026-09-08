"""Qualify a local image offline, retaining evidence under runs/ (never publishes)."""

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np

from blab.headless import HeadlessSolveSpec, load_headless_project, prepare_headless_solve
from blab.remote import RemoteBackend
from blab.remote_contract import build_job_bundle
from blab.system_contract import system_frequency_result_from_dict
from check_remote_integration import compare_results

# Test-only TCP relay: the solver stays on an internal network with no egress.
RELAY = """
import os, select, socket, socketserver
class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.create_connection((os.environ['TARGET'], 8765)) as upstream:
            sockets = [self.request, upstream]
            while True:
                readable, _, _ = select.select(sockets, [], [], 35)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is self.request else self.request).sendall(data)
socketserver.ThreadingTCPServer.allow_reuse_address = True
socketserver.ThreadingTCPServer(('0.0.0.0', 8765), Relay).serve_forever()
"""


def docker(*arguments, check=True):
    result = subprocess.run(["docker", *map(str, arguments)], capture_output=True, text=True, check=False)
    if check and result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result


def wait_ready(client):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            client.wait_ready()
            return client.check_capabilities()
        except OSError:
            time.sleep(1)
    raise TimeoutError("Container did not become ready")


def terminal(client, job_id):
    deadline = time.monotonic() + 90
    cursor = 0
    while time.monotonic() < deadline:
        response = client.call(f"/v1/jobs/{job_id}?cursor={cursor}")
        cursor = response["next_cursor"]
        for event in response["events"]:
            if event["type"] in {"completed", "failed", "cancelled"}:
                return event
        time.sleep(0.25)
    raise TimeoutError("Job did not terminate")


def check_transport_results(output):
    events = [json.loads(line) for line in (output / "server-events.ndjson").read_text().splitlines()]
    results = [system_frequency_result_from_dict(event["result"]) for event in events if event["type"] == "result"]
    assert len(results) == 1
    metadata = json.loads((output / "remote/frequencies/000000.json").read_text())
    # Headless preparation names the single probe and reorders retained outputs.
    keys = {
        quantity["metadata"].get("source_output_id", quantity["id"]): quantity["key"]
        for quantity in metadata["quantities"]
    }
    assert set(keys) == {quantity.id for quantity in results[0].quantities}
    with np.load(output / "remote/frequencies/000000.npz") as arrays:
        assert len(arrays.files) == len(results[0].quantities)
        for quantity in results[0].quantities:
            np.testing.assert_array_equal(quantity.values, arrays[keys[quantity.id]])
    return {"status": "passed", "arrays_exactly_equal": len(results[0].quantities)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    name = "blab-test-" + uuid.uuid4().hex[:12]
    volume, network = name + "-data", name + "-net"
    environment_key = "BLAB_SERVER_TOKEN"
    previous_key = os.environ.get(environment_key)
    os.environ[environment_key] = secrets.token_urlsafe(32)
    request = output / "request.json"
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
    project = root / "examples/Simple_Sealed/simple_sealed.blab.json"
    cli = [sys.executable, "-m", "blab.cli", "project"]
    with (output / "host-validate.log").open("w") as log:
        subprocess.run(
            cli + ["validate", str(project), "--backend", "beat_cpu", "--request", str(request), "--json"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    report = {"image": args.image, "expected_backend": args.backend}
    try:
        rejected = docker("run", "--rm", "--network", "none", args.image, check=False)
        assert rejected.returncode != 0 and "Hosted mode requires" in rejected.stderr
        report["hosted_requires_key"] = True
        docker("volume", "create", volume)
        docker("network", "create", "--internal", network)

        def start(*extra, private=False, gpu=True):
            docker(
                "run",
                "-d",
                "--name",
                name,
                "--network",
                network,
                "-v",
                f"{volume}:/data",
                "--mount",
                f"type=bind,source={root / 'examples'},target=/fixtures,readonly",
                *(["--gpus", "all"] if args.backend == "cuda" and gpu else []),
                *(["-e", "NVIDIA_VISIBLE_DEVICES=void"] if not gpu else []),
                "-e",
                environment_key + "=" if private else environment_key,
                args.image,
                *extra,
            )
            docker("rm", "-f", name + "-relay", check=False)
            docker(
                "run",
                "-d",
                "--name",
                name + "-relay",
                "--network",
                "bridge",
                "--network",
                network,
                "-p",
                "127.0.0.1::8765",
                "-e",
                f"TARGET={name}",
                "--entrypoint",
                "python",
                args.image,
                "-c",
                RELAY,
            )
            port = docker("port", name + "-relay", "8765/tcp").stdout.strip().rsplit(":", 1)[1]
            return RemoteBackend(f"http://127.0.0.1:{port}", token=None if private else os.environ[environment_key])

        client = start()
        report["capabilities"] = wait_ready(client)
        expected = "beat_" + args.backend
        assert report["capabilities"]["backends"][expected]["available"], report["capabilities"]
        for key in (None, "incorrect-key-" * 4):
            try:
                RemoteBackend(client.url, token=key).check_capabilities()
            except RuntimeError as error:
                assert "HTTP 401" in str(error)
            else:
                raise AssertionError("Unauthorized connection accepted")
        report["authentication"] = "passed"
        docker("exec", name, "python", "/opt/blab-container/healthcheck.py")
        docker("cp", request, f"{name}:/tmp/request.json")
        common = [
            "/fixtures/Simple_Sealed/simple_sealed.blab.json",
            "--backend",
            expected,
            "--request",
            "/tmp/request.json",
        ]
        for action in ("validate", "solve"):
            result = docker(
                "exec",
                name,
                "python",
                "-m",
                "blab.cli",
                "project",
                action,
                *common,
                *(["--json"] if action == "validate" else ["--output", "/data/reference"]),
            )
            (output / f"container-{action}.log").write_text(result.stdout + result.stderr)
        docker("cp", f"{name}:/data/reference", output / "local")
        with (output / "remote.log").open("w") as log:
            subprocess.run(
                cli
                + [
                    "solve",
                    str(project),
                    "--request",
                    str(request),
                    "--server-url",
                    client.url,
                    "--server-token-env",
                    environment_key,
                    "--output",
                    str(output / "remote"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        numerical_error = None
        try:
            compare_results(output)
        except AssertionError as error:
            numerical_error = str(error) or "Numerical comparison assertion failed"
        report["numerical_comparison"] = numerical_error or "passed"
        manifest = json.loads((output / "remote/manifest.json").read_text())
        assert manifest["backend_id"] == expected, manifest["backend_id"]
        report["automatic_selection"] = expected
        docker("cp", f"{name}:/data/jobs/{manifest['remote_job_id']}/events.ndjson", output / "server-events.ndjson")
        report["transport_integrity"] = check_transport_results(output)

        prepared = prepare_headless_solve(
            load_headless_project(project),
            HeadlessSolveSpec(frequencies_hz=(500.0,), include_project_observations=False),
            backend_id="beat_remote",
            include_observation_planes=False,
        )
        bundle = build_job_bundle(replace(prepared.request, frequencies_hz=tuple(range(500, 550))))
        job_id = client.call("/v1/jobs", data=bundle, method="POST")["job_id"]
        client.call(f"/v1/jobs/{job_id}", method="DELETE")
        assert terminal(client, job_id)["type"] == "cancelled"
        report["cancellation"] = "passed"
        # Stop with an active job to exercise SIGTERM and worker cleanup.
        active_id = client.call("/v1/jobs", data=bundle, method="POST")["job_id"]
        time.sleep(2)
        docker("stop", "--time", "30", name)
        state = json.loads(docker("inspect", name).stdout)[0]["State"]
        assert state["ExitCode"] == 0, state
        report["graceful_shutdown"] = True
        (output / "server.log").write_text(docker("logs", name).stdout + docker("logs", name).stderr)
        docker("rm", name)
        client = start()
        wait_ready(client)
        assert terminal(client, manifest["remote_job_id"])["type"] == "completed"
        assert terminal(client, active_id)["type"] in {"cancelled", "failed"}
        report["replacement_retains_jobs"] = True
        docker("stop", "--time", "30", name)
        docker("rm", name)
        client = start("--mode", "private-network", private=True, gpu=False)
        capabilities = wait_ready(client)
        assert capabilities["backends"]["beat_cpu"]["available"]
        assert not capabilities["backends"]["beat_cuda"]["available"]
        docker("exec", name, "python", "/opt/blab-container/healthcheck.py")
        report["private_network_without_key"] = True
        report["cpu_available_without_gpu"] = True
        report["offline_runtime"] = True
        report["image_inspect"] = json.loads(docker("image", "inspect", args.image).stdout)[0]
        report["status"] = "failed" if numerical_error else "passed"
        (output / "qualification.json").write_text(json.dumps(report, indent=2))
        if numerical_error:
            raise RuntimeError(f"Container lifecycle checks passed; numerical comparison failed: {numerical_error}")
        print(f"Container qualification passed: {output}")
    finally:
        logs = docker("logs", name, check=False)
        (output / "last-server.log").write_text(logs.stdout + logs.stderr)
        docker("stop", "--time", "30", name, check=False)
        docker("rm", name, check=False)
        docker("rm", "-f", name + "-relay", check=False)
        docker("volume", "rm", volume, check=False)
        docker("network", "rm", network, check=False)
        if previous_key is None:
            os.environ.pop(environment_key, None)
        else:
            os.environ[environment_key] = previous_key


if __name__ == "__main__":
    main()
