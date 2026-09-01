from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from blab import deploy_worker
from blab.system_contract import (
    QuantityResult,
    SystemFrequencyResult,
    system_frequency_result_to_dict,
)


class _PackageCache:
    def load_package(self, _path: Path):
        return SimpleNamespace(frequencies=np.asarray([40.0, 20.0, 40.0]))


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
        path.write_text(
            json.dumps(
                {
                    "frequencies_hz": [20.0, 40.0],
                    "observation_points_m": payload["observationPointsM"],
                }
            ),
            encoding="utf-8",
        )
        return path, {}

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


def test_coupled_worker_key_is_separate_from_level_two_worker() -> None:
    assert deploy_worker._worker_key({"backend": "cuda"}) == "cuda"
    assert deploy_worker._worker_key({"backend": "cuda", "fidelity": "coupled"}) == "coupled:cuda"


def test_coupled_result_maps_complex_pressure_to_deploy_field() -> None:
    raw = system_frequency_result_to_dict(
        SystemFrequencyResult(
            freq_hz=100.0,
            excitation_port_ids=("port:a", "port:b"),
            quantities=(
                QuantityResult(
                    id="deploy:field-pressure",
                    quantity="exterior_pressure",
                    unit="Pa",
                    axes=("observation",),
                    values=np.asarray([0.2 + 0.0j, 0.0 + 0.02j], dtype=np.complex64),
                ),
            ),
            diagnostics={"bem_backend": "cuda", "timings": {"solve_s": 0.25}},
        )
    )

    result = deploy_worker._coupled_deploy_result(
        raw,
        {
            "deploy": {
                "rows": 1,
                "columns": 2,
                "sample_indices": [0, 1],
                "source_count": 2,
                "rigid_object_count": 0,
            }
        },
    )

    np.testing.assert_allclose(result["spl_db"], [80.0, 60.0], atol=1e-5)
    assert result["field_pressure"] == {"real": [0.20000000298023224, 0.0], "imag": [0.0, 0.019999999552965164]}
    assert result["timings"] == {"solve_s": 0.25}
    assert result["diagnostics"]["fidelity"] == "coupled"
