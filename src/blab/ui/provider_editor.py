"""Document-scoped GUI contract for geometry provider editors.

Source and generation methods are GUI-thread only. ``services`` is the existing
asynchronous provider host API, with subscriptions scoped to this editor.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QWidget

from blab.generators.host import ProviderHost, ProviderHostError
from blab.project.model import replace_generator_document
from blab.ui.ath_editor import AthScriptEditor


@dataclass(frozen=True)
class SourceSnapshot:
    source: dict
    revision: str
    schema_version: int


class ProviderEditor(Protocol):
    widget: QWidget

    def apply_source(self, source: dict, revision: str) -> None: ...
    def set_operation_state(self, state) -> None: ...
    def dispose(self) -> None: ...


class DocumentHost:
    def __init__(self, window, document):
        self._window = window
        self.document_id = document.id
        self.provider_id = document.provider_id
        self._epoch = window.project_session.epoch
        self._disposed = False
        self._subscriptions = set()
        self._lock = RLock()
        self._services = window.provider_host.bind(document.id, document.provider_id)
        self.services = ProviderHost(self._dispatch)

    def _document(self):
        if QThread.currentThread() != self._window.thread():
            raise ProviderHostError("wrong_thread", "Editor source and generate methods require the GUI thread.")
        if self._disposed or self._epoch != self._window.project_session.epoch:
            raise ProviderHostError("invalid_context", "Editor has been disposed or its project replaced.")
        for document in self._window.generator_documents:
            if document.id == self.document_id and document.provider_id == self.provider_id:
                return document
        raise ProviderHostError("invalid_context", "Provider document no longer exists.")

    def snapshot(self) -> SourceSnapshot:
        document = self._document()
        payload = json.dumps(
            [self._epoch, document.id, document.provider_schema_version, document.source],
            sort_keys=True,
            allow_nan=False,
        )
        return SourceSnapshot(
            deepcopy(document.source), hashlib.sha256(payload.encode()).hexdigest(), document.provider_schema_version
        )

    def _check_revision(self, revision):
        if self.snapshot().revision != revision:
            raise ProviderHostError(
                "stale_source", "Source changed; read a fresh snapshot before editing or generating."
            )

    def update_source(self, source: dict, *, expected_revision: str) -> SourceSnapshot:
        self._check_revision(expected_revision)
        if not isinstance(source, dict):
            raise ValueError("Source must be a JSON object.")
        # Roundtrip both validates JSON values and detaches provider-owned data.
        source = json.loads(json.dumps(source, allow_nan=False))
        self._window.generator_documents = replace_generator_document(
            self._window.generator_documents,
            self.document_id,
            source=source,
        )
        return self.snapshot()

    def generate(self, *, expected_revision: str) -> str:
        self._check_revision(expected_revision)
        window = self._window
        if window.geometry_controller.active or window.solve_controller.active or window.preparations.active:
            raise ProviderHostError("busy", "Boundary Lab is already preparing, generating or solving.")
        window.active_generator_document_id = self.document_id
        window.editor_tabs.setCurrentIndex(window.active_generator_document_index())
        window.geometry_workflow.generate_geometry()
        request_id = window.geometry_workflow.pending_request_id
        if request_id is None:
            raise ProviderHostError("generation_failed", "The host could not start generation.")
        return request_id

    def _dispatch(self, method, *args):
        with self._lock:
            if self._disposed:
                future = Future()
                future.set_exception(ProviderHostError("closed", "Editor has been disposed."))
                return future
            if method == "subscribe":
                callback = args[0]
                if not callable(callback):
                    raise ValueError("Subscriber must be callable.")

                def guarded(event):
                    if not self._disposed:
                        callback(event)

                future = self._services.subscribe(guarded)

                def subscribed(done):
                    if done.cancelled() or done.exception() is not None:
                        return
                    with self._lock:
                        if self._disposed:
                            self._services.unsubscribe(done.result())
                        else:
                            self._subscriptions.add(done.result())

                future.add_done_callback(subscribed)
                return future
            if method == "unsubscribe":
                self._subscriptions.discard(args[0])
            return getattr(self._services, method)(*args)

    def dispose(self):
        with self._lock:
            self._disposed = True
            for token in self._subscriptions:
                self._services.unsubscribe(token)
            self._subscriptions.clear()


class AthProviderEditor:
    def __init__(self, parent, host, *, highlight_syntax=True):
        self.host = host
        self.widget = AthScriptEditor(highlight_syntax=highlight_syntax)
        self.widget.setParent(parent)
        self.revision = ""
        self.widget.textChanged.connect(self._edited)

    def apply_source(self, source, revision):
        if source.get("format", "ath_cfg") != "ath_cfg":
            raise ValueError("Unsupported Ath source format.")
        self.revision = revision
        self._source = deepcopy(source)
        blocked = self.widget.blockSignals(True)
        self.widget.setPlainText(str(source.get("text", "")))
        self.widget.blockSignals(blocked)

    def _edited(self):
        source = dict(self._source, text=self.widget.toPlainText())
        snapshot = self.host.update_source(source, expected_revision=self.revision)
        self._source, self.revision = snapshot.source, snapshot.revision

    def set_operation_state(self, state):
        self.widget.setReadOnly(state.active)

    def dispose(self):
        self.widget.textChanged.disconnect(self._edited)
