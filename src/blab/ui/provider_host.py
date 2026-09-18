"""Desktop adapter for provider commands; all application access stays on Qt."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future
from dataclasses import asdict
from threading import Lock
from uuid import uuid4

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from blab.generators.configuration import project_revision
from blab.generators.host import ProviderContext, ProviderHost, ProviderHostError, ProviderService


class DesktopProviderHost(QObject):
    _command = Signal(object)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._bindings = {}
        self._closed = False
        self._scheduled = False
        self._lock = Lock()
        self._futures = set()
        self.service = ProviderService(
            context=self._context,
            busy=self._busy,
            start=window.solve_workflow.start_provider_solve,
            cancel=window.solve_workflow.cancel_provider_solve,
            current_result=lambda: window.solve_session.solved_system,
            validate=self._validate,
        )
        self._command.connect(self._dispatch, Qt.ConnectionType.QueuedConnection)
        window.solve_workflow.provider_preparation_failed.connect(self._failed)
        window.solve_workflow.provider_solve_finished.connect(self._finished)
        window.solve_controller.state_changed.connect(self._progress)
        window.solve_controller.finished.connect(self.wake)
        window.geometry_controller.finished.connect(self.wake)
        window.preparations.busy_changed.connect(self.wake)
        window.geometry_workflow.generation_accepted.connect(self._geometry_accepted)
        window.project_state_changed.connect(self.wake)
        window.mesh_state_changed.connect(self.wake)

    def bind(self, document_id, provider_id):
        """Called on the GUI thread; returned facade is usable from any thread."""
        project = self.window.project_session.epoch
        for token, (bound_project, doc_id, provider, host) in self._bindings.items():
            if bound_project == project and doc_id == document_id and provider == provider_id:
                return host
        token = uuid4().hex
        host = ProviderHost(lambda method, *args: self._submit(token, method, args))
        self._bindings[token] = (project, document_id, provider_id, host)
        return host

    def _validate(self, owner):
        if owner not in self._bindings:
            raise ProviderHostError("invalid_context", "Provider binding has expired.")
        epoch, document_id, provider_id, _host = self._bindings[owner]
        project = self.window._project_document()
        if self.window.project_session.epoch != epoch or not any(
            document.id == document_id and document.provider_id == provider_id
            for document in project.generator_documents
        ):
            raise ProviderHostError("invalid_context", "Provider document was removed or the project was replaced.")

    def _context(self, owner):
        self._validate(owner)
        _epoch, document_id, provider_id, _host = self._bindings[owner]
        snapshot = self.window._generation_project_snapshot()
        preferences = json.dumps(asdict(self.window.preferences), sort_keys=True, allow_nan=False)
        revision = hashlib.sha256((project_revision(snapshot) + preferences).encode()).hexdigest()
        frequencies = self.window.frequency_range()
        return ProviderContext(
            document_id, provider_id, revision, frequencies.min_hz, frequencies.max_hz, frequencies.count
        )

    def _busy(self):
        window = self.window
        return window.geometry_controller.active or window.solve_controller.active or window.preparations.active

    def _submit(self, owner, method, args):
        future = Future()
        with self._lock:
            if self._closed:
                future.set_exception(ProviderHostError("closed", "Provider host is closed."))
                return future
            self._futures.add(future)
            self._command.emit((owner, method, args, future))
        return future

    @Slot(object)
    def _dispatch(self, command):
        owner, method, args, future = command
        with self._lock:
            self._futures.discard(future)
        if future.done() or not future.set_running_or_notify_cancel():
            return
        try:
            result = self.service.call(owner, method, *args)
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)
        self.wake()

    def wake(self, *_args):
        if self._closed or self._scheduled:
            return
        self._scheduled = True
        QTimer.singleShot(0, self._pump)

    def _pump(self):
        self._scheduled = False
        if self._closed:
            return
        project = self.window._project_document()
        documents = {(document.id, document.provider_id) for document in project.generator_documents}
        for owner, (epoch, document_id, provider_id, _host) in tuple(self._bindings.items()):
            if epoch != self.window.project_session.epoch or (document_id, provider_id) not in documents:
                del self._bindings[owner]
                self.service.revoke(owner)
        self.service.pump()

    @Slot(str, str)
    def _failed(self, job_id, message):
        self.service.finish(job_id, "failed", message=message)
        self.wake()

    @Slot(str, object, object)
    def _finished(self, job_id, completion, result):
        self.service.finish(
            job_id, completion.phase.value, result=result, message=self.window.solve_controller.last_error or ""
        )
        self.wake()

    @Slot(object)
    def _progress(self, state):
        job_id = self.service.active
        if job_id is not None:
            self.service.progress(job_id, message=state.message, solved_count=self.window.solve_session.solved_count)

    @Slot(object)
    def _geometry_accepted(self, completed):
        request = completed.request
        for owner, (project, document_id, provider_id, _host) in tuple(self._bindings.items()):
            if (
                project == self.window.project_session.epoch
                and document_id == request.document_id
                and provider_id == request.provider_id
            ):
                self.service.geometry_accepted(owner, request.request_id)

    def close(self):
        with self._lock:
            self._closed = True
            futures, self._futures = self._futures, set()
        for future in futures:
            if not future.done() and future.set_running_or_notify_cancel():
                future.set_exception(ProviderHostError("closed", "Provider host is closed."))
        self.service.close()
        self._bindings.clear()
