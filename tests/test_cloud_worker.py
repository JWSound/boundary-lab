import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import blab.cloud.worker as cloud_worker
from blab.config import SimulationConfig
from blab.live import FrequencyResult


class RecordingSink:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event: dict) -> None:
        self.events.append(event)


class FakeLiveSolver:
    polar_angle_deg = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    radiator_names = np.array(["throat"])
    sphere_metadata = None

    def __init__(self, config: SimulationConfig):
        self.config = config

    def solve_stream(self, frequencies, *, stop_requested=None):
        for frequency in frequencies:
            yield FrequencyResult(
                freq_hz=float(frequency),
                horizontal_spl_norm_db=np.array([-6.0, 0.0, -6.0], dtype=np.float32),
                vertical_spl_norm_db=np.array([-8.0, 0.0, -8.0], dtype=np.float32),
                impedance=np.array([[1.0, 0.2]], dtype=np.float32),
            )


def test_run_solve_job_emits_stream_events(monkeypatch) -> None:
    monkeypatch.setattr(cloud_worker, "LiveSolver", FakeLiveSolver)
    sink = RecordingSink()

    cloud_worker.run_solve_job(
        job_id="job_test",
        config=SimulationConfig(mesh_file="case.msh"),
        frequencies=np.array([200.0, 1000.0], dtype=np.float32),
        sink=sink,
    )

    assert [event["type"] for event in sink.events] == [
        "status",
        "initialized",
        "frequency_result",
        "frequency_result",
        "completed",
    ]
    assert sink.events[1]["job_id"] == "job_test"
    assert sink.events[2]["frequency_hz"] == 200.0


def test_jsonl_event_sink_writes_events(tmp_path: Path) -> None:
    sink = cloud_worker.JsonlEventSink(tmp_path / "events.jsonl")

    sink.emit({"type": "status", "job_id": "job_test", "message": "ok"})

    assert json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8")) == {
        "type": "status",
        "job_id": "job_test",
        "message": "ok",
    }


def test_resolve_bundle_path_requires_local_or_s3_source(tmp_path: Path) -> None:
    args = Namespace(bundle=None, s3_bucket=None, s3_key=None)

    with pytest.raises(SystemExit, match="Provide --bundle"):
        cloud_worker._resolve_bundle_path(args, tmp_path)


def test_resolve_bundle_path_accepts_local_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "solve.blabsolve.zip"
    args = Namespace(bundle=bundle, s3_bucket=None, s3_key=None)

    assert cloud_worker._resolve_bundle_path(args, tmp_path) == bundle


def test_build_event_sink_can_add_local_event_store(tmp_path: Path) -> None:
    args = Namespace(
        events_jsonl=None,
        event_store="local",
        event_store_root=tmp_path / "events",
        dynamodb_events_table=None,
        job_id="job_test",
    )

    sink = cloud_worker._build_event_sink(args)
    sink.emit({"type": "completed", "job_id": "job_test"})

    assert (tmp_path / "events" / "jobs" / "job_test" / "events.jsonl").exists()
