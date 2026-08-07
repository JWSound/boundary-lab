"""Solver backend capability probing and cached server health.

This is a controller, not a mixin: it owns the health cache and the probe
thread, and depends on exactly one thing from its host — a way to read the
current preferences. It touches no widgets, so it is testable without a
window and survives a UI revamp untouched.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot

from blab.solvers.http_server import server_health_supports_symmetry
from blab.solvers.registry import backend_info
from blab.ui.server_health_worker import ServerHealthCheckWorker
from blab.ui.server_tokens import load_server_access_token
from blab.ui.settings import GuiPreferences

SERVER_BACKEND_ID = "server"
HEALTH_CHECK_TIMEOUT_S = 5.0


class BackendHealthController(QObject):
    """Answers "can the selected solver backend do X?" and probes the server."""

    #: Emitted with a reason when a probe changes the answer to a capability
    #: question, so the host can invalidate anything derived from it.
    capability_changed = Signal(str)

    def __init__(self, parent: QObject | None, *, preferences: Callable[[], GuiPreferences]) -> None:
        super().__init__(parent)
        self._read_preferences = preferences
        self.payload: dict | None = None
        self.url: str | None = None
        self._thread: QThread | None = None
        self._worker: ServerHealthCheckWorker | None = None

    # -- cached health ----------------------------------------------------

    def matches_preferences(self, preferences: GuiPreferences | None = None) -> bool:
        """True when the cached health belongs to the currently configured server."""
        prefs = preferences or self._read_preferences()
        if prefs.solve_backend != SERVER_BACKEND_ID or self.payload is None or self.url is None:
            return False
        return self.url.rstrip("/") == prefs.solve_server_url.rstrip("/")

    def cache(self, payload: dict | None, url: str | None) -> None:
        self.payload = payload
        self.url = None if url is None else url.rstrip("/")

    def clear(self) -> None:
        self.cache(None, None)

    # -- capability questions ---------------------------------------------

    def supports_symmetry(
        self,
        backend_id: str,
        *,
        preferences: GuiPreferences | None = None,
        server_health_payload: dict | None = None,
    ) -> bool:
        if backend_id != SERVER_BACKEND_ID:
            return backend_info(backend_id).capabilities.supports_symmetry
        if server_health_payload is not None:
            return server_health_supports_symmetry(server_health_payload)
        if self.matches_preferences(preferences):
            return server_health_supports_symmetry(self.payload)
        return False

    def selected_backend_supports_symmetry(self) -> bool:
        return self.supports_symmetry(self._read_preferences().solve_backend)

    def effective_symmetry(
        self,
        symmetry: str,
        preferences: GuiPreferences,
        *,
        server_health_payload: dict | None = None,
    ) -> str:
        """Downgrade a symmetry choice the selected backend cannot honour."""
        if symmetry == "off" or self.supports_symmetry(
            preferences.solve_backend,
            preferences=preferences,
            server_health_payload=server_health_payload,
        ):
            return symmetry
        return "off"

    # -- startup probe -----------------------------------------------------

    @Slot()
    def check_on_startup(self) -> None:
        preferences = self._read_preferences()
        if preferences.solve_backend != SERVER_BACKEND_ID or self._thread is not None:
            return
        worker = ServerHealthCheckWorker(
            preferences.solve_server_url,
            access_token=load_server_access_token(preferences.solve_server_url),
            timeout_s=HEALTH_CHECK_TIMEOUT_S,
        )
        thread = QThread(self)
        self._thread = thread
        self._worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._on_succeeded)
        worker.failed.connect(lambda _message: None)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    @Slot(str, object)
    def _on_succeeded(self, server_url: str, payload: dict) -> None:
        preferences = self._read_preferences()
        if preferences.solve_backend != SERVER_BACKEND_ID:
            return
        normalized_url = server_url.rstrip("/")
        if normalized_url != preferences.solve_server_url.rstrip("/"):
            return
        previously_supported = self.selected_backend_supports_symmetry()
        self.cache(payload, normalized_url)
        if self.selected_backend_supports_symmetry() != previously_supported:
            self.capability_changed.emit("server_health_checked")

    @Slot()
    def _on_finished(self) -> None:
        self._thread = None
        self._worker = None
