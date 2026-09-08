from types import SimpleNamespace

import pytest

from blab.remote import RemoteBackend
from blab.server_capabilities import BackendDiscovery


def test_discovery_uses_worker_availability_and_preserves_failure_reason(monkeypatch):
    class Worker:
        def __init__(self, **kwargs):
            self.kind = kwargs["julia_project"]
            self.worker_info = {
                "backends": {self.kind: {"available": self.kind == "cuda", "reason": "driver unavailable"}},
                "engine": {"version": "test"},
            }

        def ensure_started(self):
            if self.kind == "rocm":
                raise RuntimeError("rocSOLVER missing")

        def terminate(self):
            pass

    monkeypatch.setattr("blab.server_capabilities.BeatEngineWorkerProcess", Worker)
    backends = {
        "beat_" + key: SimpleNamespace(
            julia_executable="julia", solver_script="solver", julia_project=key, julia_threads=2
        )
        for key in ("cpu", "cuda", "rocm")
    }
    discovery = BackendDiscovery(backends)
    for thread in discovery.threads:
        thread.join(5)
    records = discovery.snapshot()
    assert records["beat_cuda"]["available"] is True
    assert records["beat_cpu"]["available"] is False
    assert "rocSOLVER missing" in records["beat_rocm"]["reason"]
    discovery.close()


@pytest.mark.parametrize(
    "requested,available,expected",
    [
        ("beat_auto", ["beat_cuda", "beat_cpu"], "beat_cuda"),
        ("beat_auto", ["beat_cpu", "beat_rocm"], "beat_cpu"),
        ("beat_rocm", ["beat_rocm"], "beat_rocm"),
    ],
)
def test_remote_selection_uses_server_status(monkeypatch, requested, available, expected):
    client = RemoteBackend("http://127.0.0.1:8765")
    monkeypatch.setattr(client, "check_capabilities", lambda: {"backend_ids": available})
    assert client.select_backend(requested) == expected


def test_explicit_gpu_does_not_fall_back(monkeypatch):
    client = RemoteBackend("http://127.0.0.1:8765")
    monkeypatch.setattr(
        client,
        "check_capabilities",
        lambda: {
            "backends": {"beat_cpu": {"available": True}, "beat_cuda": {"available": False, "reason": "No CUDA device"}}
        },
    )
    with pytest.raises(ValueError, match="No CUDA device"):
        client.select_backend("beat_cuda")


def test_pending_discovery_has_bounded_wait(monkeypatch):
    client = RemoteBackend("http://127.0.0.1:8765")
    monkeypatch.setattr(client, "check_capabilities", lambda: {"backends": {"beat_cpu": {"state": "checking"}}})
    with pytest.raises(TimeoutError):
        client.select_backend("beat_cpu", timeout=0)
