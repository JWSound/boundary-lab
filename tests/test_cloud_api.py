import numpy as np
from fastapi.testclient import TestClient

import blab.cloud.api as cloud_api
from blab.cloud.bundle import write_solve_bundle
from blab.cloud.events import LocalEventStore
from blab.config import RadiatorConfig, SimulationConfig
from blab.cloud.storage import LocalBundleStore, PresignedPutTarget, S3BundleStore


def test_cloud_api_health_endpoint() -> None:
    client = TestClient(cloud_api.create_app())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_cloud_api_accepts_solve_bundle(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cloud_api, "_start_job_thread", lambda *args: None)
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


def test_cloud_api_staged_upload_flow_starts_job(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cloud_api, "_start_job_thread", lambda *args: None)
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

    created = client.post("/v1/solve-jobs/upload-target")

    assert created.status_code == 200
    created_payload = created.json()
    assert created_payload["status"] == "created"
    assert created_payload["upload"]["method"] == "PUT"
    assert created_payload["upload"]["url"].endswith("/bundle")

    uploaded = client.put(
        created_payload["upload"]["url"],
        content=bundle_path.read_bytes(),
        headers=created_payload["upload"]["headers"],
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["status"] == "uploaded"

    started = client.post(created_payload["start_path"])
    assert started.status_code == 200
    started_payload = started.json()
    assert started_payload["status"] == "queued"
    assert started_payload["frequency_count"] == 2


class FakePresignedBundleStore:
    def bundle_key(self, job_id: str) -> str:
        return f"jobs/{job_id}/input/solve.blabsolve.zip"

    def create_upload_target(self, job_id: str, *, expires_in_s: int = 3600):
        return PresignedPutTarget(
            url=f"https://upload.example.test/{job_id}",
            headers={"Content-Type": "application/zip"},
            key=self.bundle_key(job_id),
        )

    def download_bundle(self, job_id: str, destination):
        raise AssertionError("download_bundle should not be called in this test")


def test_cloud_api_presigned_upload_target_shape() -> None:
    client = TestClient(cloud_api.create_app(bundle_store=FakePresignedBundleStore()))

    response = client.post("/v1/solve-jobs/upload-target")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "created"
    assert payload["upload"]["method"] == "PUT"
    assert payload["upload"]["url"].startswith("https://upload.example.test/job_")
    assert payload["upload"]["headers"] == {"Content-Type": "application/zip"}
    assert payload["start_path"].endswith("/start")


class FakeS3Client:
    def __init__(self, bundle_bytes: bytes):
        self.bundle_bytes = bundle_bytes

    def generate_presigned_url(self, **kwargs):
        return "https://upload.example.test/bundle"

    def download_file(self, bucket, key, destination):
        with open(destination, "wb") as handle:
            handle.write(self.bundle_bytes)


class FakeTaskLauncher:
    def __init__(self):
        self.calls = []

    def launch_worker(self, *, job_id: str, s3_bucket: str, s3_key: str):
        self.calls.append((job_id, s3_bucket, s3_key))
        return type(
            "Launch",
            (),
            {
                "task_arn": "arn:aws:ecs:us-east-1:123:task/cluster/task-id",
                "cluster": "cluster",
                "task_definition": "taskdef:1",
            },
        )()


def test_cloud_api_start_can_submit_ecs_task(tmp_path) -> None:
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
    task_launcher = FakeTaskLauncher()
    bundle_store = S3BundleStore("bucket", client=FakeS3Client(bundle_path.read_bytes()))
    client = TestClient(cloud_api.create_app(bundle_store=bundle_store, task_launcher=task_launcher))

    created = client.post("/v1/solve-jobs/upload-target").json()
    started = client.post(created["start_path"])

    assert started.status_code == 200
    payload = started.json()
    assert payload["status"] == "submitted"
    assert payload["launch"] == {
        "type": "ecs",
        "task_arn": "arn:aws:ecs:us-east-1:123:task/cluster/task-id",
        "cluster": "cluster",
        "task_definition": "taskdef:1",
    }
    assert task_launcher.calls == [
        (payload["job_id"], "bucket", f"jobs/{payload['job_id']}/input/solve.blabsolve.zip")
    ]


def test_cloud_api_websocket_streams_durable_worker_events(tmp_path) -> None:
    mesh_path = tmp_path / "case.msh"
    mesh_path.write_text("mesh", encoding="utf-8")
    bundle_path = write_solve_bundle(
        tmp_path / "case.blabsolve.zip",
        config=SimulationConfig(mesh_file=str(mesh_path), radiators=(RadiatorConfig(name="throat", tag=2),)),
        frequencies=np.array([200.0], dtype=np.float32),
    )
    event_store = LocalEventStore(tmp_path / "events")
    task_launcher = FakeTaskLauncher()
    bundle_store = S3BundleStore("bucket", client=FakeS3Client(bundle_path.read_bytes()))
    client = TestClient(
        cloud_api.create_app(
            bundle_store=bundle_store,
            task_launcher=task_launcher,
            event_store=event_store,
        )
    )
    created = client.post("/v1/solve-jobs/upload-target").json()
    started = client.post(created["start_path"]).json()
    job_id = started["job_id"]

    event_store.append(job_id, {"type": "status", "job_id": job_id, "message": "worker says hi"})
    event_store.append(job_id, {"type": "completed", "job_id": job_id})

    with client.websocket_connect(started["stream_path"]) as websocket:
        submitted = websocket.receive_json()
        worker_status = websocket.receive_json()
        completed = websocket.receive_json()

    assert submitted["type"] == "status"
    assert "ECS task submitted" in submitted["message"]
    assert worker_status["message"] == "worker says hi"
    assert completed["type"] == "completed"


def test_cloud_api_ecs_start_requires_s3_bundle_store(tmp_path) -> None:
    mesh_path = tmp_path / "case.msh"
    mesh_path.write_text("mesh", encoding="utf-8")
    bundle_path = write_solve_bundle(
        tmp_path / "case.blabsolve.zip",
        config=SimulationConfig(mesh_file=str(mesh_path), radiators=(RadiatorConfig(name="throat", tag=2),)),
        frequencies=np.array([200.0], dtype=np.float32),
    )
    client = TestClient(
        cloud_api.create_app(bundle_store=LocalBundleStore(tmp_path / "bundles"), task_launcher=FakeTaskLauncher())
    )
    created = client.post("/v1/solve-jobs/upload-target").json()
    client.put(
        created["upload"]["url"],
        content=bundle_path.read_bytes(),
        headers=created["upload"]["headers"],
    )

    response = client.post(created["start_path"])

    assert response.status_code == 200
    assert response.json()["status"] == "failed"


def test_cloud_api_unknown_stream_sends_failure_event() -> None:
    client = TestClient(cloud_api.create_app())

    with client.websocket_connect("/v1/solve-jobs/job_missing/stream") as websocket:
        event = websocket.receive_json()

    assert event == {
        "type": "failed",
        "job_id": "job_missing",
        "message": "Unknown solve job.",
    }
