"""The provider facade marshals commands to Qt without lending window state."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import get_ident
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from blab.generators.host import ProviderHostError, SolveCommand


def until(qapp, predicate):
    deadline = monotonic() + 5
    while not predicate() and monotonic() < deadline:
        qapp.processEvents()
        sleep(0.001)
    assert predicate(), "Qt command did not settle"


def resolve(qapp, future):
    until(qapp, future.done)
    return future.result()


def bind(window):
    document = window.project.generator_documents[0]
    return window.provider_host.bind(document.id, document.provider_id)


def test_provider_calls_from_worker_thread_execute_on_gui(main_window, qapp):
    host = bind(main_window)
    calls = []
    original = main_window.provider_host.service._context

    def context(owner):
        calls.append(get_ident())
        return original(owner)

    main_window.provider_host.service._context = context
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(lambda: host.context().result(timeout=5))
        until(qapp, waiting.done)
        result = waiting.result()
    assert calls == [get_ident()]
    assert result.freq_count == main_window.frequency_range().count
    assert result.document_id == main_window.project.generator_documents[0].id


def test_accepted_generation_keeps_handle_but_new_project_revokes_it(main_window, qapp):
    host = bind(main_window)
    events = []
    resolve(qapp, host.subscribe(events.append))
    old_context = resolve(qapp, host.context())
    # Versioned acceptance swaps in a detached ProjectDocument, not a new session.
    main_window.project = deepcopy(main_window.project)
    request = SimpleNamespace(
        document_id=old_context.document_id, provider_id=old_context.provider_id, request_id="generation"
    )
    main_window.geometry_workflow.generation_accepted.emit(SimpleNamespace(request=request))
    assert events[-1].kind == "geometry_accepted"
    assert resolve(qapp, host.context()).document_id == old_context.document_id
    main_window.project_session.replace(deepcopy(main_window.project), path=None)
    with pytest.raises(ProviderHostError) as error:
        resolve(qapp, host.context())
    assert error.value.code == "invalid_context"


def test_pending_request_waits_and_uses_captured_host_context(main_window, qapp):
    host = bind(main_window)
    context = resolve(qapp, host.context())
    service = main_window.provider_host.service
    service._busy = lambda: True
    calls = []
    service._start = lambda job_id: calls.append((job_id, get_ident()))
    job = resolve(qapp, host.solve(SolveCommand(context.project_revision)))
    assert job.state == "queued"
    assert not calls
    service._busy = lambda: False
    main_window.provider_host.wake()
    until(qapp, lambda: bool(calls))
    assert calls == [(job.job_id, get_ident())]
    status = resolve(qapp, host.status(job.job_id))
    assert status.context == context
    service.finish(job.job_id, "failed", message="Synthetic failure")
    assert resolve(qapp, host.status(job.job_id)).message == "Synthetic failure"


def test_frequency_change_invalidates_queued_request(main_window, qapp):
    host = bind(main_window)
    context = resolve(qapp, host.context())
    service = main_window.provider_host.service
    service._busy = lambda: True
    service._start = lambda _: pytest.fail("Stale solve started")
    job = resolve(qapp, host.solve(SolveCommand(context.project_revision)))
    main_window.freq_count_spin.setValue(context.freq_count + 1)
    service._busy = lambda: False
    main_window.provider_host.wake()
    until(qapp, lambda: service.pending is None)
    assert resolve(qapp, host.status(job.job_id)).state == "stale"


def test_close_resolves_pending_futures(main_window, qapp):
    host = bind(main_window)
    future = host.context()
    main_window.provider_host.close()
    with pytest.raises(ProviderHostError) as error:
        future.result()
    assert error.value.code == "closed"
    with pytest.raises(ProviderHostError):
        host.context().result()
    qapp.processEvents()  # already queued delivery must not overwrite the exception


def test_worker_injects_optional_host_without_changing_generation_request(qapp, monkeypatch, tmp_path):
    from blab.generators.base import GeneratedGeometry, GenerationRequest
    from blab.ui.generator_worker import GeneratorWorker
    from test_memory_mesh import tetra_surface

    host, observed = object(), []
    request = GenerationRequest("custom", "design", "tetra", {}, tmp_path, "example")

    class Backend:
        def bind_host(self, value):
            observed.append(value)

        def create_session(self, value):
            assert value is request
            return self

        def generate(self, **kwargs):
            return GeneratedGeometry("custom", tmp_path, None, (), mesh_data=tetra_surface())

    monkeypatch.setattr("blab.ui.generator_worker.create_generator", lambda *_a, **_kw: Backend())
    worker = GeneratorWorker(request, host=host)
    generated, errors = [], []
    worker.generated.connect(generated.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert observed == [host]
    assert len(generated) == 1 and generated[0].request.request_id == request.request_id
    assert not errors


def test_loading_project_payload_revokes_old_binding(main_window, qapp):
    from blab.project.model import generator_document_to_payload

    host = bind(main_window)
    resolve(qapp, host.context())
    # Reopening a project with identical document IDs is still a new session.
    document = main_window.project.generator_documents[0]
    main_window.project_workflow._apply_project_payload(
        {
            "generator_documents": [generator_document_to_payload(document)],
            "active_generator_document_id": document.id,
        },
        generated_results={},
    )
    with pytest.raises(ProviderHostError) as error:
        resolve(qapp, host.context())
    assert error.value.code == "invalid_context"
