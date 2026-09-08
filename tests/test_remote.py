import io
import json
import threading
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import numpy as np
import pytest

from blab import __version__
from blab.headless import HeadlessSolveSpec, load_headless_project, prepare_headless_solve
from blab.remote import RemoteBackend
from blab.remote_contract import build_job_bundle, stage_job_bundle
from blab.server import SolveService, create_http_server
from blab.system_contract import QuantityResult, SystemFrequencyResult


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_remote_requests_identify_application_to_hosted_proxy(method):
    client = RemoteBackend("https://server.example", token="test-key")

    def open_response(request, **kwargs):
        assert request.get_header("User-agent") == f"BoundaryLab/{__version__}"
        assert request.get_header("Authorization") == "Bearer test-key"
        response = io.BytesIO(b'{"ok": true}')
        response.status = 200
        return response

    client.opener = SimpleNamespace(open=open_response)
    assert client.call("/v1/jobs", method=method) == {"ok": True}


@pytest.mark.parametrize("body", [b"", b"<html>Gateway unavailable</html>", b"[]", b"null"])
def test_proxy_error_preserves_http_status_without_exposing_body(body):
    client = RemoteBackend("https://server.example")

    def open_response(*args, **kwargs):
        raise HTTPError(client.url, 502, "Bad Gateway", {}, io.BytesIO(body))

    client.opener = SimpleNamespace(open=open_response)
    with pytest.raises(RuntimeError, match="HTTP 502.*server or proxy"):
        client.check_capabilities()


@pytest.mark.parametrize("body", [b"", b"<html>Starting</html>", b"[]", b"null"])
def test_successful_http_response_requires_json_object(body):
    client = RemoteBackend("https://server.example")
    response = io.BytesIO(body)
    response.status = 200
    client.opener = SimpleNamespace(open=lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="HTTP 200.*JSON"):
        client.check_capabilities()


@pytest.fixture
def prepared():
    root = Path(__file__).resolve().parents[1]
    project = load_headless_project(root / "examples/Simple_Sealed/simple_sealed.blab.json")
    return prepare_headless_solve(
        project,
        HeadlessSolveSpec(frequencies_hz=(500.0,), include_project_observations=False),
        backend_id="beat_cpu",
        include_observation_planes=False,
    )


def rewrite_bundle(bundle, change, extra=None):
    with zipfile.ZipFile(io.BytesIO(bundle)) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    envelope = json.loads(entries["job.json"])
    change(envelope)
    entries["job.json"] = json.dumps(envelope).encode()
    if extra:
        entries.update(extra)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return stream.getvalue()


def test_bundle_relocates_and_deduplicates_assets(prepared, tmp_path):
    bundle = build_job_bundle(prepared.request)
    staged = stage_job_bundle(bundle, tmp_path / "job")
    for old, new in zip(prepared.request.compiled_system.meshes, staged.compiled_system.meshes, strict=True):
        assert Path(new.file).is_relative_to(tmp_path / "job" / "assets")
        assert Path(new.file).read_bytes() == Path(old.file).read_bytes()
    assert staged.frequencies_hz == prepared.request.frequencies_hz
    assert staged.excitation_port_ids == prepared.request.excitation_port_ids
    envelope = json.loads((tmp_path / "job/job.json").read_text())
    assert len(envelope["assets"]) == len({mesh.file for mesh in staged.compiled_system.meshes})
    assert all(not Path(mesh["file"]).is_absolute() for mesh in envelope["request"]["compiled_system"]["meshes"])


@pytest.mark.parametrize(
    "change",
    [
        lambda job: job.update(version=True),
        lambda job: job.update(version=99),
        lambda job: job.update(backend_id="beat_cuda"),
        lambda job: job.update(observation_planes=True),
        lambda job: job["request"].update(cancel_path="C:/private/cancel"),
        lambda job: job["request"]["solver_options"].update(script="malicious.jl"),
        lambda job: job["request"]["compiled_system"]["meshes"][0].update(file="../../private.msh"),
        lambda job: job["assets"][0].update(sha256="0" * 64),
    ],
)
def test_invalid_bundle_rejected_before_staging(prepared, tmp_path, change):
    bundle = rewrite_bundle(build_job_bundle(prepared.request), change)
    with pytest.raises(ValueError):
        stage_job_bundle(bundle, tmp_path / "job")
    assert not (tmp_path / "job").exists()


def test_unlisted_archive_path_is_never_extracted(prepared, tmp_path):
    bundle = rewrite_bundle(build_job_bundle(prepared.request), lambda _job: None, {"../outside": b"unsafe"})
    with pytest.raises(ValueError, match="Unexpected"):
        stage_job_bundle(bundle, tmp_path / "job")
    assert not (tmp_path / "outside").exists()


def fake_result(request):
    return SystemFrequencyResult(
        freq_hz=500.0,
        excitation_port_ids=request.excitation_port_ids,
        quantities=(
            QuantityResult(
                id="test:pressure",
                quantity="exterior_pressure",
                unit="Pa",
                values=np.full((len(request.excitation_port_ids), 2), 1 + 2j, dtype=np.complex64),
                axes=("excitation", "observation"),
            ),
        ),
        diagnostics={"engine_provenance": {"engine": {"source_sha256": "test-hash"}}},
    )


@pytest.fixture
def http_service(tmp_path):
    backend = SimpleNamespace(
        create_system_session=lambda request: SimpleNamespace(
            request=request,
            solve_stream=lambda: iter([fake_result(request)]),
            stop=lambda: None,
            worker_provenance={"engine": {"version": "test-engine"}},
        )
    )
    service = SolveService(tmp_path / "jobs", backend=backend)
    server = create_http_server(service, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = RemoteBackend(f"http://127.0.0.1:{server.server_port}")
    try:
        yield service, client
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=5)


def test_http_complex_results_and_replay_after_service_restart(prepared, http_service):
    service, client = http_service
    session = client.create_system_session(prepared.request)
    results = list(session.solve_stream())
    assert len(results) == 1
    np.testing.assert_array_equal(results[0].quantities[0].values, fake_result(prepared.request).quantities[0].values)
    assert results[0].excitation_port_ids == prepared.request.excitation_port_ids
    assert results[0].diagnostics == fake_result(prepared.request).diagnostics
    assert session.worker_provenance["engine"]["version"] == "test-engine"
    replay = SolveService(service.root, backend=service.backend)
    assert replay.events(session.job_id, 0)[-1]["type"] == "completed"


def test_capability_mismatch_prevents_submission(monkeypatch, prepared):
    client = RemoteBackend("http://127.0.0.1:8765")
    monkeypatch.setattr(client, "call", lambda _path: {"protocol": "blab-remote", "version": 99})
    with pytest.raises(ValueError, match="Incompatible"):
        client.create_system_session(prepared.request)


def test_worker_failure_is_terminal_and_releases_slot(prepared, http_service):
    service, client = http_service

    def fail():
        raise RuntimeError("worker crashed")

    service.backend.create_system_session = lambda request: SimpleNamespace(
        request=request, solve_stream=fail, stop=lambda: None
    )
    for _ in range(2):
        session = client.create_system_session(prepared.request)
        with pytest.raises(RuntimeError, match="worker crashed"):
            list(session.solve_stream())
        for thread in service.threads:
            thread.join(timeout=5)
        assert service.events(session.job_id, 0)[-1]["type"] == "failed"


def test_busy_and_cancel_preserve_partial_results(prepared, http_service):
    service, client = http_service
    release = threading.Event()
    emitted = threading.Event()

    def stream():
        yield fake_result(prepared.request)
        emitted.set()
        assert release.wait(5)

    service.backend.create_system_session = lambda request: SimpleNamespace(
        request=request, solve_stream=stream, stop=release.set
    )
    job_id = client.call("/v1/jobs", data=build_job_bundle(prepared.request), method="POST")["job_id"]
    assert emitted.wait(5)
    with pytest.raises(RuntimeError, match="409"):
        client.call("/v1/jobs", data=build_job_bundle(prepared.request), method="POST")
    client.call(f"/v1/jobs/{job_id}", method="DELETE")
    for thread in service.threads:
        thread.join(timeout=5)
    events = service.events(job_id, 0)
    assert any(event["type"] == "result" for event in events)
    assert events[-1]["type"] == "cancelled"


def test_plane_exclusion_does_not_disable_polar_sampling(prepared, monkeypatch):
    import blab.headless as module

    captured = {}
    monkeypatch.setattr(module, "prepare_system_ui_solve", lambda *_args, **kwargs: captured.update(kwargs) or prepared)
    root = Path(__file__).resolve().parents[1]
    project = load_headless_project(root / "examples/Simple_Sealed/simple_sealed.blab.json")
    project = replace(project, payload=project.payload | {"observation_planes": "deliberately invalid"})
    prepare_headless_solve(project, HeadlessSolveSpec(), backend_id="beat_cpu", include_observation_planes=False)
    assert captured["observation_planes"] == ()
    assert captured["polar_angle_step_deg"] == project.preferences.polar_angle_step_deg
    assert captured["spherical_sampling_enabled"] == project.preferences.spherical_sampling_enabled


def test_restart_preserves_complete_events_and_marks_interrupted_job(tmp_path):
    job_id = "a" * 32
    directory = tmp_path / job_id
    directory.mkdir()
    (directory / "events.ndjson").write_bytes(b'{"type":"accepted"}\n{"type":"res')
    service = SolveService(tmp_path)
    events = service.events(job_id, 0)
    assert [event["type"] for event in events] == ["accepted", "failed"]
    assert "restarted" in events[-1]["error"]


def test_gpu_selection_crosses_http_and_routes_to_selected_backend(prepared, http_service):
    service, client = http_service
    service.backends["beat_rocm"] = service.backend
    service.backend_policy = "rocm"
    session = client.create_system_session(prepared.request)
    assert len(list(session.solve_stream())) == 1
    assert service.events(session.job_id, 0)[0]["backend_id"] == "beat_rocm"


def test_unavailable_backend_rejected_before_staging(prepared, http_service):
    service, client = http_service
    service.backend_policy = "cuda"
    with pytest.raises(RuntimeError, match="cannot run this model"):
        client.call("/v1/jobs", method="POST", data=build_job_bundle(prepared.request))
    assert not list(service.root.iterdir())
