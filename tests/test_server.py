import time
from pathlib import Path

import numpy as np

from blab.config import SimulationConfig
from blab.live import FrequencyResult
from blab.protocol import build_mesh_assets
from blab.server import JobOrchestrator


class FakeSolver:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.polar_angle_deg = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
        self.radiator_names = np.array(["throat"])
        self.sphere_metadata = None

    def solve_stream(self, frequencies, *, stop_requested=None):
        for freq in frequencies:
            if stop_requested is not None and stop_requested():
                return
            yield FrequencyResult(
                freq_hz=float(freq),
                horizontal_spl_norm_db=np.array([-6.0, 0.0, -3.0], dtype=np.float32),
                vertical_spl_norm_db=np.array([-8.0, 0.0, -4.0], dtype=np.float32),
                impedance=np.array([[1.0, 0.2]], dtype=np.float32),
                horizontal_spl_db=np.array([82.0, 88.0, 85.0], dtype=np.float32),
                vertical_spl_db=np.array([80.0, 88.0, 84.0], dtype=np.float32),
            )


def _wait_for_terminal(orchestrator: JobOrchestrator, job_id: str):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        job = orchestrator.get(job_id)
        if job is not None and job.status in {"completed", "cancelled", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


def test_orchestrator_streams_frequency_results_and_writes_artifact(tmp_path: Path) -> None:
    orchestrator = JobOrchestrator(
        max_running_jobs=1,
        artifact_root=tmp_path,
        solver_factory=FakeSolver,
    )
    try:
        job = orchestrator.submit(
            SimulationConfig(mesh_file="mesh.msh"),
            np.array([200.0, 1000.0], dtype=np.float32),
        )
        completed = _wait_for_terminal(orchestrator, job.job_id)

        events, terminal = orchestrator.events_since(job.job_id, 0)
        event_types = [event["type"] for event in events]

        assert terminal is True
        assert completed.status == "completed"
        assert completed.solved_count == 2
        assert event_types == ["queued", "started", "initialized", "result", "result", "completed"]
        assert completed.result_npz is not None
        assert completed.result_npz.exists()

        with np.load(completed.result_npz) as data:
            assert data["freq_hz"].tolist() == [200.0, 1000.0]
            assert data["horizontal_spl_norm_db"].shape == (2, 3)
            assert data["impedance_real"].shape == (1, 2)
    finally:
        orchestrator.shutdown()


def test_orchestrator_stages_uploaded_mesh_assets_before_solving(tmp_path: Path) -> None:
    source_mesh = tmp_path / "client_mesh.msh"
    source_mesh.write_text("mesh bytes", encoding="utf-8")
    seen_configs = []

    class CapturingSolver(FakeSolver):
        def __init__(self, config: SimulationConfig):
            seen_configs.append(config)
            super().__init__(config)

    orchestrator = JobOrchestrator(
        max_running_jobs=1,
        artifact_root=tmp_path / "jobs",
        solver_factory=CapturingSolver,
    )
    try:
        config = SimulationConfig(mesh_file=str(source_mesh))
        job = orchestrator.submit(
            config,
            np.array([1000.0], dtype=np.float32),
            assets=build_mesh_assets(config),
        )
        completed = _wait_for_terminal(orchestrator, job.job_id)

        assert completed.status == "completed"
        assert seen_configs
        staged_path = Path(seen_configs[0].mesh_file)
        assert staged_path != source_mesh
        assert staged_path.exists()
        assert staged_path.read_text(encoding="utf-8") == "mesh bytes"
        assert staged_path.parent == completed.artifact_dir / "assets"
    finally:
        orchestrator.shutdown()
