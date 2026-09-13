import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import blab.solvers.beat_engine_runtime as runtime
import blab.solvers.coupled_backend as coupled_backend
from blab.phasor import SOLVER_PHASOR_CONVENTION
from blab.system_contract import OutputRequest, SystemSolveRequest


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, ("GPU-a", "GPU-b")),
        ({"CUDA_VISIBLE_DEVICES": "1, 3"}, ("1", "3")),
        ({"CUDA_VISIBLE_DEVICES": "GPU-a"}, ("GPU-a",)),
        ({"CUDA_VISIBLE_DEVICES": "-1"}, ()),
        ({"BLAB_MULTI_GPU": "0"}, ()),
    ],
)
def test_cuda_sweep_devices_honors_visibility_and_opt_out(monkeypatch, environ, expected):
    monkeypatch.setattr(runtime, "_nvidia_smi_gpu_uuids", lambda: ("GPU-a", "GPU-b"))

    assert runtime.cuda_sweep_devices(environ) == expected


def test_worker_pool_keeps_one_worker_per_cuda_device(tmp_path):
    options = {
        "julia_executable": "julia",
        "solver_script": tmp_path / "solver.jl",
        "julia_threads": 1,
        "julia_project": None,
    }
    first = runtime.get_beat_engine_worker(**options, cuda_visible_devices="GPU-a")
    second = runtime.get_beat_engine_worker(**options, cuda_visible_devices="GPU-b")

    assert first is not second
    assert first is runtime.get_beat_engine_worker(**options, cuda_visible_devices="GPU-a")
    assert second.environment["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    runtime.shutdown_beat_engine_workers()


class _FakeWorker:
    worker_info = {"engine": {"version": "test-engine"}}

    def __init__(self, device, *, fail=False):
        self.device = device
        self.fail = fail
        self.payload = None
        self.cancelled = False

    def submit(self, request_path, *, status_callback=None):
        self.payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
        status_callback("BEAT Engine ready")
        if self.fail:
            yield {"type": "failed", "error": "out of memory"}
            return
        if self.device == "wait-for-cancel":
            cancel_path = Path(self.payload["cancel_path"])
            deadline = time.monotonic() + 5.0
            while not cancel_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.cancelled = cancel_path.exists()
            yield {"type": "cancelled"}
            return
        for frequency in self.payload["frequencies_hz"]:
            yield {
                "type": "result",
                "result": {"freq_hz": frequency, "diagnostics": {"phasor_convention": SOLVER_PHASOR_CONVENTION}},
            }
        yield {"type": "completed"}

    def terminate(self):
        pass


def _session(monkeypatch, devices, workers, frequencies):
    request = SystemSolveRequest(
        compiled_system=SimpleNamespace(id="system:test", assumptions=()),
        frequencies_hz=frequencies,
        excitation_port_ids=("port",),
        outputs=(
            OutputRequest(
                id="field",
                quantity="exterior_pressure",
                options={"excitation_weights_sweep": [[f] for f in frequencies]},
            ),
        ),
        solver_options={"bem_backend": "cuda"},
    )
    monkeypatch.setattr(coupled_backend, "cuda_sweep_devices", lambda: devices)
    monkeypatch.setattr(
        coupled_backend,
        "get_beat_engine_worker",
        lambda **kwargs: workers[kwargs["cuda_visible_devices"]],
    )
    monkeypatch.setattr(
        coupled_backend,
        "system_solve_request_to_dict",
        lambda r: {
            "frequencies_hz": list(r.frequencies_hz),
            "weights": r.outputs[0].options["excitation_weights_sweep"],
        },
    )
    monkeypatch.setattr(coupled_backend, "system_frequency_result_from_dict", lambda raw: raw["freq_hz"])
    return coupled_backend.CoupledSession(request)


def test_sweep_round_robins_frequencies_across_cuda_workers(monkeypatch):
    workers = {device: _FakeWorker(device) for device in ("GPU-a", "GPU-b")}
    frequencies = (100.0, 200.0, 300.0, 400.0, 500.0)
    session = _session(monkeypatch, ("GPU-a", "GPU-b"), workers, frequencies)

    solved = list(session.solve_stream())

    assert sorted(solved) == list(frequencies)
    assert workers["GPU-a"].payload["frequencies_hz"] == [100.0, 300.0, 500.0]
    assert workers["GPU-b"].payload["frequencies_hz"] == [200.0, 400.0]
    assert workers["GPU-b"].payload["weights"] == [[200.0], [400.0]]
    assert session.worker_provenance["engine"]["version"] == "test-engine"
    assert session.worker_provenance["cuda_visible_devices"] == ["GPU-a", "GPU-b"]


def test_single_frequency_does_not_split(monkeypatch):
    session = _session(monkeypatch, ("GPU-a", "GPU-b"), {}, (100.0,))

    assert session._sweep_devices() == ("GPU-a",)


def test_worker_failure_cancels_the_other_devices(monkeypatch):
    workers = {"bad": _FakeWorker("bad", fail=True), "wait-for-cancel": _FakeWorker("wait-for-cancel")}
    session = _session(monkeypatch, ("wait-for-cancel", "bad"), workers, (100.0, 200.0))

    with pytest.raises(RuntimeError, match="GPU 2/2: out of memory"):
        list(session.solve_stream())

    assert workers["wait-for-cancel"].cancelled
