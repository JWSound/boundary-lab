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
    "policy,available,expected",
    [
        ("auto", ["beat_cuda", "beat_cpu"], "beat_cuda"),
        ("auto", ["beat_cpu", "beat_rocm"], "beat_rocm"),
        ("rocm", ["beat_rocm"], "beat_rocm"),
    ],
)
def test_server_owns_selection(tmp_path, policy, available, expected):
    from blab.physical_model import AcousticRegionKind
    from blab.server import SolveService

    service = SolveService(tmp_path, backends={key: object() for key in available}, backend_policy=policy)
    request = SimpleNamespace(
        compiled_system=SimpleNamespace(regions=[SimpleNamespace(kind=AcousticRegionKind.UNBOUNDED_AIR)])
    )
    assert service.select_backend(request) == expected


def test_server_forces_interior_fem_to_cpu(tmp_path):
    from blab.server import SolveService

    service = SolveService(tmp_path, backends={"beat_cpu": object(), "beat_cuda": object()}, backend_policy="cuda")
    request = SimpleNamespace(compiled_system=SimpleNamespace(regions=[]))
    assert service.select_backend(request) == "beat_cpu"
    service.backends.pop("beat_cuda")
    assert service.readiness() == "ready"
    assert service.select_backend(request) == "beat_cpu"


def test_server_pinned_gpu_does_not_fall_back(tmp_path):
    from blab.physical_model import AcousticRegionKind
    from blab.server import SolveService

    service = SolveService(tmp_path, backend=object(), backend_policy="cuda")
    request = SimpleNamespace(
        compiled_system=SimpleNamespace(regions=[SimpleNamespace(kind=AcousticRegionKind.UNBOUNDED_AIR)])
    )
    with pytest.raises(ValueError, match="cannot run"):
        service.select_backend(request)


def test_client_waits_on_readiness_without_hardware_selection(monkeypatch):
    client = RemoteBackend("http://127.0.0.1:8765")
    monkeypatch.setattr(client, "check_capabilities", lambda **kwargs: {"state": "starting"})
    with pytest.raises(TimeoutError):
        client.wait_ready(timeout=0)
    monkeypatch.setattr(client, "check_capabilities", lambda **kwargs: {"state": "ready"})
    assert client.wait_ready() is None
