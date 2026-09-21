"""Qt-free provider commands, correlated solve jobs, and result ownership.

The host serializes ``ProviderService`` calls on its owning thread. Providers
receive only ``ProviderHost``; its dispatcher returns Futures and may marshal
calls from any thread. No widgets, solver objects, or mutable project are lent.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Callable
from uuid import uuid4

from blab.solve_results import SolvedSystem

API_VERSION = 1
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "superseded", "stale"})


class ProviderHostError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ProviderContext:
    document_id: str
    provider_id: str
    project_revision: str
    freq_min_hz: float
    freq_max_hz: float
    freq_count: int
    schema_version: int = API_VERSION


@dataclass(frozen=True)
class SolveCommand:
    expected_revision: str
    request_id: str = field(default_factory=lambda: uuid4().hex)
    replace_pending: bool = False
    schema_version: int = API_VERSION

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != API_VERSION:
            raise ValueError("Unsupported provider command schema version.")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("A nonempty request ID is required.")
        if not isinstance(self.expected_revision, str) or not self.expected_revision:
            raise ValueError("An expected project revision is required.")
        if type(self.replace_pending) is not bool:
            raise ValueError("replace_pending must be boolean.")


@dataclass(frozen=True)
class SolveJob:
    job_id: str
    request_id: str
    context: ProviderContext
    state: str = "queued"
    message: str = "Waiting for host"
    solved_count: int = 0
    expected_count: int = 0
    run_id: str | None = None
    schema_version: int = API_VERSION


@dataclass(frozen=True)
class ProviderEvent:
    kind: str
    job: SolveJob | None = None
    context: ProviderContext | None = None
    generation_request_id: str | None = None
    schema_version: int = API_VERSION


class ProviderHost:
    """Nonblocking API. Futures acknowledge commands, not solve completion.

    Event callbacks run on the host dispatch thread: keep them short and never
    block waiting for another host Future there. Unsubscribe when disposing UI.
    """

    def __init__(self, dispatch: Callable[..., Future]):
        self._dispatch = dispatch

    def context(self) -> Future[ProviderContext]:
        return self._dispatch("context")

    def solve(self, command: SolveCommand) -> Future[SolveJob]:
        return self._dispatch("solve", command)

    def status(self, job_id: str) -> Future[SolveJob]:
        return self._dispatch("status", job_id)

    def cancel(self, job_id: str) -> Future[SolveJob]:
        return self._dispatch("cancel", job_id)

    def result(self, job_id: str) -> Future[SolvedSystem]:
        """Return a detached canonical SolvedSystem, including partial results."""
        return self._dispatch("result", job_id)

    def current_result(self) -> Future[SolvedSystem]:
        """Read the host's latest finalized snapshot, including manual solves."""
        return self._dispatch("current_result")

    def subscribe(self, callback: Callable[[ProviderEvent], None]) -> Future[str]:
        """Return a subscription ID; includes geometry acceptance and job events."""
        return self._dispatch("subscribe", callback)

    def unsubscribe(self, subscription_id: str) -> Future[None]:
        return self._dispatch("unsubscribe", subscription_id)


class ProviderService:
    """One active solve and one pending request, with bounded result retention.

    Owner tokens are supplied by the host, never chosen by provider requests.
    ``context(owner)`` must reject removed documents/replaced projects. ``start``
    must eventually call ``finish`` even if preparation fails. Host callbacks
    may report synchronously; active job identity is installed before dispatch.
    """

    def __init__(
        self,
        *,
        context,
        busy,
        start,
        cancel,
        validate=None,
        current_result=lambda: None,
        result_limit=4,
        history_limit=32,
    ):
        if result_limit < 1 or history_limit < result_limit + 2:
            raise ValueError("History must accommodate retained results and active jobs.")
        self._context = context
        self._validate = validate or (lambda owner: self._context(owner))
        self._busy = busy
        self._start = start
        self._cancel = cancel
        self._current_result = current_result
        self._jobs: OrderedDict[str, tuple[str, SolveCommand, SolveJob]] = OrderedDict()
        self._results: OrderedDict[str, SolvedSystem] = OrderedDict()
        self._subscriptions: dict[str, tuple[str, Callable]] = {}
        self.result_limit, self.history_limit = result_limit, history_limit
        self.active: str | None = None
        self.pending: str | None = None
        self.closed = False

    def call(self, owner, method, *args):
        if self.closed:
            raise ProviderHostError("closed", "Provider host is closed.")
        self._validate(owner)
        if method == "context":
            return self._context(owner)
        if method == "current_result":
            result = self._current_result()
            if result is None:
                raise ProviderHostError("result_unavailable", "The host has no finalized result snapshot.")
            return deepcopy(result)
        if method == "subscribe":
            if not callable(args[0]):
                raise ValueError("Subscriber must be callable.")
            if sum(item[0] == owner for item in self._subscriptions.values()) >= 64:
                raise ProviderHostError("subscription_limit", "Unsubscribe unused provider callbacks first.")
            token = uuid4().hex
            self._subscriptions[token] = (owner, args[0])
            return token
        if method == "unsubscribe":
            subscription = self._subscriptions.get(args[0])
            if subscription is not None and subscription[0] == owner:
                del self._subscriptions[args[0]]
            return None
        if method == "solve":
            return self._submit(owner, self._context(owner), args[0])
        if method not in {"status", "cancel", "result"}:
            raise ProviderHostError("unknown_command", f"Unknown command: {method}")
        job = self._owned(owner, args[0])
        if method == "status":
            return job
        if method == "cancel":
            if job.state in TERMINAL_STATES:
                return job
            if job.job_id == self.pending:
                self.pending = None
                return self._update(job.job_id, state="cancelled", message="Cancelled before start")
            self._update(job.job_id, state="cancelling", message="Cancellation requested")
            self._cancel(job.job_id)
            return self._jobs[job.job_id][2]
        if job.state not in TERMINAL_STATES:
            raise ProviderHostError("not_ready", "Results are available after the job settles.")
        result = self._results.get(job.job_id)
        if result is None:
            raise ProviderHostError("result_unavailable", "No retained result for this job (empty or evicted).")
        # The frozen dataclasses still contain mutable dicts and NumPy arrays.
        # Never expose the host's retained snapshot to provider mutation.
        return deepcopy(result)

    def _submit(self, owner, context, command):
        if not isinstance(command, SolveCommand):
            raise ValueError("solve requires a SolveCommand.")
        for stored_owner, previous, job in self._jobs.values():
            if stored_owner == owner and previous.request_id == command.request_id:
                if previous != command:
                    raise ProviderHostError("request_conflict", "Request ID was already used with different inputs.")
                return job
        if command.expected_revision != context.project_revision:
            raise ProviderHostError("stale_revision", "Project inputs changed; obtain a new context.")
        if self.pending is not None:
            previous_owner, _, _ = self._jobs[self.pending]
            if previous_owner != owner or not command.replace_pending:
                raise ProviderHostError("busy", "A provider solve is already pending.")
            previous_id, self.pending = self.pending, None
            self._update(previous_id, state="superseded", message="Replaced by newer provider request")
        job = SolveJob(uuid4().hex, command.request_id, context, expected_count=context.freq_count)
        self._jobs[job.job_id] = (owner, command, job)
        self.pending = job.job_id
        self._trim()
        self._emit(owner, ProviderEvent("solve_status", job=job))
        return job

    def pump(self):
        if self.closed or self.active is not None or self.pending is None or self._busy():
            return
        job_id, self.pending = self.pending, None
        owner, _, job = self._jobs[job_id]
        try:
            context = self._context(owner)
            if context != job.context:
                raise ProviderHostError("stale_revision", "Project inputs changed while queued.")
        except Exception as exc:
            self._update(job_id, state="stale", message=str(exc))
            return
        self.active = job_id
        self._update(job_id, state="preparing", message="Preparing solve")
        try:
            self._start(job_id)
        except Exception as exc:
            self.finish(job_id, "failed", message=str(exc))

    def progress(self, job_id, *, message, solved_count=0, expected_count=None):
        if job_id != self.active:
            return
        job = self._jobs[job_id][2]
        self._update(
            job_id,
            state="cancelling" if job.state == "cancelling" else "running",
            message=message,
            solved_count=solved_count,
            expected_count=job.expected_count if expected_count is None else expected_count,
        )

    def finish(self, job_id, state, *, result=None, message=""):
        if job_id != self.active:
            return  # Late callbacks cannot complete another job.
        if state not in {"completed", "failed", "cancelled", "stale"}:
            raise ValueError("Invalid terminal solve state.")
        owner, _, job = self._jobs[job_id]
        try:
            self._validate(owner)
        except Exception:
            state, result, message = "stale", None, "Provider document or project was replaced."
        if job.state == "cancelling" and state != "stale":
            state = "cancelled"
        if state == "completed" and (result is None or not result.complete):
            state, message = "failed", "Solver ended without a complete frequency result set."
        if result is not None:
            self._results[job_id] = result
            while len(self._results) > self.result_limit:
                self._results.popitem(last=False)
        self.active = None
        self._update(
            job_id,
            state=state,
            message=message or state,
            solved_count=0 if result is None else result.solved_count,
            run_id=None if result is None else result.run_id,
        )
        self._trim()

    def geometry_accepted(self, owner, request_id):
        self._emit(
            owner, ProviderEvent("geometry_accepted", context=self._context(owner), generation_request_id=request_id)
        )

    def revoke(self, owner):
        """Release a removed document/project and stop only its active request."""
        self._subscriptions = {key: value for key, value in self._subscriptions.items() if value[0] != owner}
        for job_id in tuple(self._results):
            if self._jobs[job_id][0] == owner:
                del self._results[job_id]
        if self.pending is not None and self._jobs[self.pending][0] == owner:
            pending, self.pending = self.pending, None
            self._update(pending, state="stale", message="Provider document or project was replaced.")
        if self.active is not None and self._jobs[self.active][0] == owner:
            self._cancel(self.active)

    def close(self):
        self.closed = True
        self._subscriptions.clear()
        self._results.clear()
        self._jobs.clear()
        active, self.active, self.pending = self.active, None, None
        if active is not None:
            self._cancel(active)

    def _owned(self, owner, job_id):
        record = self._jobs.get(job_id)
        if record is None or record[0] != owner:
            raise ProviderHostError("unknown_job", "Job does not belong to this provider document or has expired.")
        return record[2]

    def _update(self, job_id, **changes):
        owner, command, previous = self._jobs[job_id]
        job = replace(previous, **changes)
        self._jobs[job_id] = (owner, command, job)
        self._emit(owner, ProviderEvent("solve_status", job=job))
        return job

    def _emit(self, owner, event):
        for subscriber_owner, callback in tuple(self._subscriptions.values()):
            if subscriber_owner == owner:
                try:
                    callback(event)
                except Exception:
                    logging.getLogger(__name__).exception("Geometry provider event callback failed")

    def _trim(self):
        for job_id in tuple(self._jobs):
            if len(self._jobs) <= self.history_limit:
                break
            if job_id not in {self.active, self.pending} and job_id not in self._results:
                del self._jobs[job_id]
