"""Latest-request-wins preparation with GUI-thread completion callbacks."""

from __future__ import annotations

from threading import Event

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot


class _Signals(QObject):
    completed = Signal(object, object, object)


class _Job(QRunnable):
    def __init__(self, token, work, cancelled):
        super().__init__()
        self.token = token
        self.work = work
        self.cancelled = cancelled
        self.signals = _Signals()

    def run(self):
        result, error = None, None
        try:
            if not self.cancelled.is_set():
                result = self.work()
        except Exception as exc:
            error = exc
        self.signals.completed.emit(self.token, result, error)


class PreparationController(QObject):
    """Serial preparation jobs; workers receive snapshots, never widgets.

    Cancellation suppresses delivery and skips queued jobs. Already-running
    computations may finish, but cannot publish stale results or errors.
    """

    busy_changed = Signal(bool)

    def __init__(self, parent, activities):
        super().__init__(parent)
        self._activities = activities
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._requests = {}
        self._jobs = {}
        self._next = 0

    @property
    def active(self):
        return bool(self._jobs)

    def submit(self, key, message, work, complete, failed):
        self.cancel(key)
        self._next += 1
        token = (key, self._next)
        cancelled = Event()
        job = _Job(token, work, cancelled)
        activity = self._activities.start(message)
        self._requests[key] = (token, cancelled, activity, complete, failed, job)
        self._jobs[token] = (job, activity)
        job.signals.completed.connect(self._complete, Qt.QueuedConnection)
        self._pool.start(job)
        self.busy_changed.emit(True)

    def cancel(self, key):
        request = self._requests.pop(key, None)
        if request is not None:
            request[1].set()
            request[2].update("Stopping preparation...")

    def cancel_all(self):
        for key in tuple(self._requests):
            self.cancel(key)

    def close(self):
        self.cancel_all()
        # Jobs touch no window state. Keep the pool alive until workers finish.
        self._pool.waitForDone()
        for _job, activity in self._jobs.values():
            activity.finish()
        self._jobs.clear()

    @Slot(object, object, object)
    def _complete(self, token, result, error):
        job = self._jobs.pop(token, None)
        if job is not None:
            job[1].finish()
        request = self._requests.get(token[0])
        if request is None or request[0] != token:
            self.busy_changed.emit(self.active)
            return
        del self._requests[token[0]]
        self.busy_changed.emit(self.active)
        try:
            if error is not None:
                request[4](error)
            else:
                request[3](result)
        except Exception as exc:
            if error is not None:
                raise
            request[4](exc)
