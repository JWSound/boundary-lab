import numpy as np
from fastapi.testclient import TestClient

import blab.cloud.api as cloud_api
from blab.cloud.bundle import write_solve_bundle
from blab.config import RadiatorConfig, SimulationConfig


def test_cloud_api_health_endpoint() -> None:
    client = TestClient(cloud_api.create_app())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_cloud_api_accepts_solve_bundle(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cloud_api, "_start_job_thread", lambda loop, store, job: None)
    mesh_path = tmp_path / "case.msh"
    mesh_path.write_text("mesh", encoding="utf-8")
    bundle_path = write_solve_bundle(
        tmp_path / "case.blabsolve.zip",
        config=SimulationConfig(
            mesh_file=str(mesh_path),
            freq_min=200.0,
            freq_max=1000.0,
            freq_count=2,
            radiators=(RadiatorConfig(name="throat", tag=2),),
        ),
        frequencies=np.array([200.0, 1000.0], dtype=np.float32),
    )
    client = TestClient(cloud_api.create_app())

    response = client.post(
        "/v1/solve-jobs/bundle",
        content=bundle_path.read_bytes(),
        headers={"content-type": "application/zip"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["frequency_count"] == 2
    assert payload["stream_path"].startswith("/v1/solve-jobs/job_")


def test_cloud_api_unknown_stream_sends_failure_event() -> None:
    client = TestClient(cloud_api.create_app())

    with client.websocket_connect("/v1/solve-jobs/job_missing/stream") as websocket:
        event = websocket.receive_json()

    assert event == {
        "type": "failed",
        "job_id": "job_missing",
        "message": "Unknown solve job.",
    }
