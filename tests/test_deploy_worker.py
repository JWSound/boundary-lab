from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from blab import deploy_worker


class _PackageCache:
    def load_package(self, _path: Path):
        return SimpleNamespace(frequencies=np.asarray([40.0, 20.0, 40.0]))


class _CoupledPackageCache:
    def __init__(self, representation: str):
        self.representation = representation

    def load_package(self, _path: Path):
        model = {"representation": self.representation, "frequency_band_hz": [20.0, 40.0]}
        return SimpleNamespace(frequencies=np.asarray([10.0, 20.0, 40.0, 80.0]), coupled_model=model)


class _SweepWorker:
    def submit(self, request_path: Path, **_kwargs):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        microphone_count = len(request["observation_points_m"])
        for frequency in request["frequencies_hz"]:
            yield {
                "type": "result",
                "result": {
                    "frequency_hz": frequency,
                    "spl_db": [frequency + index for index in range(microphone_count)],
                    "field_pressure": {
                        "real": [frequency / 100.0 for _ in range(microphone_count)],
                        "imag": [0.0 for _ in range(microphone_count)],
                    },
                },
            }
        yield {"type": "completed"}


def _payload() -> dict:
    return {
        "packagePath": "speaker.blabsp",
        "backend": "cuda",
        "sources": [{"id": "source"}],
        "microphones": [
            {"id": "mic-a", "positionX": 0.0, "positionHeightM": 1.2, "positionZ": 4.0},
            {"id": "mic-b", "positionX": 2.0, "positionHeightM": 1.2, "positionZ": 6.0},
        ],
    }


def test_microphone_sweep_uses_sorted_unique_package_frequencies(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []

    def prepare(payload, work_dir, **_kwargs):
        path = Path(work_dir) / "request.json"
        request = {
            "frequencies_hz": [20.0, 40.0],
            "observation_points_m": payload["observationPointsM"],
        }
        path.write_text(json.dumps(request), encoding="utf-8")
        return path, request

    monkeypatch.setattr(deploy_worker, "prepare_deploy_microphone_sweep_request", prepare)
    monkeypatch.setattr(
        deploy_worker,
        "_emit",
        lambda event_type, **values: events.append((event_type, values)) or {},
    )
    deploy_worker._microphone_sweep(
        7,
        _payload(),
        {"cuda": _SweepWorker()},
        _PackageCache(),
        threading.Event(),
    )

    progress = [values for event_type, values in events if event_type == "microphone-progress"]
    result = next(values["result"] for event_type, values in events if event_type == "result")
    assert [value["frequency_hz"] for value in progress] == [20.0, 40.0]
    assert result["frequencies_hz"] == [20.0, 40.0]
    assert result["microphone_ids"] == ["mic-a", "mic-b"]
    assert result["spl_db"] == [[20.0, 40.0], [21.0, 41.0]]
    assert events[-1][0] == "completed"


def test_microphone_sweep_honors_stop_before_first_frequency(monkeypatch) -> None:
    event_types: list[str] = []
    monkeypatch.setattr(
        deploy_worker,
        "_emit",
        lambda event_type, **_values: event_types.append(event_type) or {},
    )
    cancel = threading.Event()
    cancel.set()

    deploy_worker._microphone_sweep(
        8,
        _payload(),
        {"cuda": _SweepWorker()},
        _PackageCache(),
        cancel,
    )

    assert event_types == ["cancelled"]


def test_level_three_rom_uses_the_shared_exterior_worker() -> None:
    assert deploy_worker._worker_key({"backend": "cuda"}) == "cuda"
    assert deploy_worker._worker_key({"backend": "cuda", "fidelity": "coupled"}) == "cuda"


def test_level_three_execution_rejects_exact_and_routes_rom_packages() -> None:
    payload = {"packagePath": "speaker.blabsp", "backend": "cuda", "fidelity": "coupled"}
    with pytest.raises(ValueError, match="parity Petrov"):
        deploy_worker._execution_worker_key(
            payload, _CoupledPackageCache("exact_frequency_parametric_fem")
        )
    assert deploy_worker._execution_worker_key(
        payload, _CoupledPackageCache("parity_petrov_galerkin_rom")
    ) == "cuda"


def test_transducer_velocity_result_flattens_scene_instances() -> None:
    request = {
        "transducers": [
            {"id": "left:transducer:0", "name": "Left / Transducer 1"},
            {"id": "left:transducer:1", "name": "Left / Transducer 2"},
            {"id": "right:transducer:0", "name": "Right / Transducer 1"},
            {"id": "right:transducer:1", "name": "Right / Transducer 2"},
        ]
    }
    result = {
        "diagnostics": {
            "transducer_velocity": [
                {"real": [1.0, 2.0], "imag": [0.1, 0.2]},
                {"real": [3.0, 4.0], "imag": [0.3, 0.4]},
            ]
        }
    }

    velocity = deploy_worker._transducer_velocity_result(result, request)

    assert velocity["ids"] == [item["id"] for item in request["transducers"]]
    assert velocity["names"][2] == "Right / Transducer 1"
    assert velocity["real"] == [1.0, 2.0, 3.0, 4.0]
    assert velocity["imag"] == [0.1, 0.2, 0.3, 0.4]


def test_coupled_excursion_sweep_does_not_require_a_microphone(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    package = SimpleNamespace(
        frequencies=np.asarray([20.0, 40.0]),
        coupled_model={
            "representation": "parity_petrov_galerkin_rom",
            "arrays": {"frequencies_hz": np.asarray([20.0, 40.0])},
        },
    )
    cache = SimpleNamespace(load_package=lambda _path: package)

    def prepare(payload, work_dir, **_kwargs):
        assert payload["observationPointsM"] == [[0.0, 1.0, 1.0]]
        path = Path(work_dir) / "request.json"
        request = {
            "frequencies_hz": [20.0, 40.0],
            "transducers": [{"id": "source:transducer:0", "name": "Source / Transducer 1"}],
        }
        path.write_text(json.dumps(request), encoding="utf-8")
        return path, request

    class Worker:
        def submit(self, _request_path, **_kwargs):
            for frequency, velocity in ((20.0, 2.0), (40.0, 4.0)):
                yield {"type": "result", "result": {
                    "frequency_hz": frequency,
                    "spl_db": [80.0],
                    "field_pressure": {"real": [0.2], "imag": [0.0]},
                    "diagnostics": {"transducer_velocity": [{"real": [velocity], "imag": [0.0]}]},
                }}
            yield {"type": "completed"}

    monkeypatch.setattr(deploy_worker, "prepare_deploy_rom_microphone_sweep_request", prepare)
    monkeypatch.setattr(deploy_worker, "_emit", lambda event_type, **values: events.append((event_type, values)) or {})
    deploy_worker._microphone_sweep(
        9,
        {"packagePath": "speaker.blabsp", "backend": "cuda", "fidelity": "coupled", "sources": [{"id": "source"}], "microphones": []},
        {"cuda": Worker()},
        cache,
        threading.Event(),
    )

    result = next(values["result"] for event_type, values in events if event_type == "result")
    assert result["microphone_ids"] == []
    assert result["transducer_ids"] == ["source:transducer:0"]
    assert result["transducer_velocity"] == {"real": [[2.0, 4.0]], "imag": [[0.0, 0.0]]}
