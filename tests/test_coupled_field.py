from __future__ import annotations

import json
import threading
from dataclasses import replace

import numpy as np

from blab.solvers.coupled_field import BemFieldEvaluationRequest, evaluate_bem_field


def _request(*, value: float = 1.0) -> BemFieldEvaluationRequest:
    return BemFieldEvaluationRequest(
        domain_key="run:test",
        points_m=np.asarray([[0.0, 0.0, 1.0], [0.25, 0.25, 1.0]]),
        boundary_points_m=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        boundary_triangles=np.asarray([[0, 1, 2]]),
        boundary_pressure=np.full(3, value + 2.0j, dtype=np.complex64),
        boundary_normal_derivative=np.full(1, 3.0 - 4.0j, dtype=np.complex64),
        frequency_hz=1000.0,
        sound_speed_m_per_s=343.0,
        symmetry="x",
    )


def test_evaluate_bem_field_uses_warmed_worker_protocol(monkeypatch) -> None:
    import blab.solvers.coupled_field as field_module

    submissions = []

    class Worker:
        def submit(self, request_path, *, operation):
            submissions.append((json.loads(request_path.read_text(encoding="utf-8")), operation))
            yield {
                "type": "field_result",
                "values": {
                    "dtype": "complex64",
                    "shape": [2],
                    "real": [1.0, 2.0],
                    "imag": [-1.0, -2.0],
                },
            }
            yield {"type": "completed"}

    monkeypatch.setattr(field_module, "_get_julia_worker", lambda **_kwargs: Worker())

    values = evaluate_bem_field(_request(), backend_id="beat_cpu")

    np.testing.assert_allclose(values, [1.0 - 1.0j, 2.0 - 2.0j])
    payload, operation = submissions[0]
    assert operation == "bem_field"
    assert payload["bem_backend"] == "cpu"
    assert payload["precision"] == "float32"
    assert payload["symmetry"] == "x"
    assert payload["domain_key"] == "run:test"
    assert payload["boundary_triangles"] == [[0, 1, 2]]
    assert payload["boundary_pressure"]["shape"] == [3]


def test_exterior_field_service_coalesces_queued_requests(qapp, monkeypatch) -> None:
    import blab.ui.exterior_field_service as service_module
    from blab.ui.exterior_field_service import ExteriorFieldEvaluationService, ExteriorFieldTask

    started = threading.Event()
    release = threading.Event()
    evaluated = []

    def evaluate(request, **_kwargs):
        evaluated.append(float(request.boundary_pressure[0].real))
        if len(evaluated) == 1:
            started.set()
            assert release.wait(timeout=5.0)
        return np.full(request.points_m.shape[0], evaluated[-1], dtype=np.complex64)

    monkeypatch.setattr(service_module, "evaluate_bem_field", evaluate)
    service = ExteriorFieldEvaluationService()
    completed = []
    discarded = []
    service.completed.connect(lambda key, values: completed.append((key, np.asarray(values))))
    service.discarded.connect(discarded.append)
    try:
        service.submit(ExteriorFieldTask(("a",), "beat_cpu", _request(value=1.0)))
        assert started.wait(timeout=2.0)
        service.submit(ExteriorFieldTask(("b",), "beat_cpu", _request(value=2.0)))
        service.submit(ExteriorFieldTask(("c",), "beat_cpu", _request(value=3.0)))
        assert discarded == [("b",)]
        release.set()
        for _index in range(100):
            qapp.processEvents()
            if len(completed) == 2:
                break
            threading.Event().wait(0.01)

        assert evaluated == [1.0, 3.0]
        assert [item[0] for item in completed] == [("a",), ("c",)]
        np.testing.assert_allclose(completed[-1][1], 3.0)
    finally:
        release.set()
        service.close()


def test_exterior_field_service_masks_boundary_samples_off_ui_thread(qapp, monkeypatch) -> None:
    import blab.ui.exterior_field_service as service_module
    from blab.ui.exterior_field_service import ExteriorFieldEvaluationService, ExteriorFieldTask

    request = replace(
        _request(),
        points_m=np.asarray([[0.25, 0.25, 0.0], [0.25, 0.25, 1.0]]),
        symmetry="off",
    )
    monkeypatch.setattr(
        service_module,
        "evaluate_bem_field",
        lambda request, **_kwargs: np.ones(request.points_m.shape[0], dtype=np.complex64),
    )
    service = ExteriorFieldEvaluationService()
    try:
        values = service._evaluate_task(
            ExteriorFieldTask(("mask",), "beat_cpu", request, mask_distance_m=1.0e-4)
        )
    finally:
        service.close()

    assert np.isnan(values[0])
    assert values[1] == 1.0
