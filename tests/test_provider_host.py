"""Provider commands own jobs, never application settings or mutable results."""

from dataclasses import replace

import numpy as np
import pytest

from blab.generators.host import ProviderContext, ProviderHostError, ProviderService, SolveCommand
from blab.solve_results import SolvedSystemBuilder, SolveProvenance
from blab.system_contract import QuantityResult, SystemFrequencyResult


def solved(*, complete=True):
    builder = SolvedSystemBuilder(
        frequencies_hz=(100.0, 200.0),
        excitation_ids=("drive",),
        provenance=SolveProvenance("beat_cpu", "exterior_bem"),
    )
    for frequency in (100.0, 200.0) if complete else (100.0,):
        builder.add(
            SystemFrequencyResult(
                frequency,
                (
                    QuantityResult(
                        "pressure", "pressure", "Pa", np.array([[1 + 2j]]), axes=("excitation", "observation")
                    ),
                ),
                ("drive",),
            )
        )
    return builder.finalize(status="completed" if complete else "cancelled")


@pytest.fixture
def host():
    contexts = {owner: ProviderContext(owner, "custom", "revision", 100.0, 200.0, 2) for owner in ("a", "b")}
    starts, cancels, events = [], [], []
    service = ProviderService(
        context=contexts.__getitem__, busy=lambda: False, start=starts.append, cancel=cancels.append
    )
    service.call("a", "subscribe", events.append)
    service.contexts, service.starts, service.cancels, service.events = contexts, starts, cancels, events
    return service


def test_job_lifecycle_correlates_revision_and_complex_results(host):
    command = SolveCommand("revision")
    job = host.call("a", "solve", command)
    assert job.context.freq_count == 2
    assert host.call("a", "solve", command) == job  # retries do not start another solve
    host.pump()
    host.progress(job.job_id, message="One frequency", solved_count=1)
    result = solved()
    host.finish(job.job_id, "completed", result=result)
    status = host.call("a", "status", job.job_id)
    assert status.state == "completed" and status.run_id == result.run_id
    assert [event.job.state for event in host.events] == ["queued", "preparing", "running", "completed"]
    copy = host.call("a", "result", job.job_id)
    assert np.iscomplexobj(copy.quantity("pressure").values)
    np.testing.assert_array_equal(copy.quantity("pressure").values, result.quantity("pressure").values)
    copy.quantities.clear()
    copy.provenance.solver_options["changed"] = True
    assert host.call("a", "result", job.job_id).quantities
    assert "changed" not in result.provenance.solver_options


def test_stale_or_conflicting_requests_never_start(host):
    with pytest.raises(ProviderHostError, match="changed"):
        host.call("a", "solve", SolveCommand("old"))
    command = SolveCommand("revision", request_id="retry")
    job = host.call("a", "solve", command)
    with pytest.raises(ProviderHostError) as error:
        host.call("a", "solve", replace(command, replace_pending=True))
    assert error.value.code == "request_conflict"
    host.contexts["a"] = replace(host.contexts["a"], freq_count=3)
    host.pump()
    assert host.call("a", "status", job.job_id).state == "stale"
    assert host.starts == []


def test_latest_pending_does_not_cancel_active_or_another_owner(host):
    active = host.call("a", "solve", SolveCommand("revision"))
    host.pump()
    pending = host.call("a", "solve", SolveCommand("revision"))
    with pytest.raises(ProviderHostError) as error:
        host.call("b", "solve", SolveCommand("revision", replace_pending=True))
    assert error.value.code == "busy"
    newer = host.call("a", "solve", SolveCommand("revision", replace_pending=True))
    assert host.call("a", "status", pending.job_id).state == "superseded"
    assert host.active == active.job_id and host.pending == newer.job_id
    assert host.cancels == []
    host.finish(active.job_id, "completed", result=solved())
    host.pump()
    host.finish(active.job_id, "failed")  # late event from previous job
    assert host.active == newer.job_id


def test_cancel_is_scoped_and_partial_results_are_explicit(host):
    job = host.call("a", "solve", SolveCommand("revision"))
    with pytest.raises(ProviderHostError) as error:
        host.call("b", "cancel", job.job_id)
    assert error.value.code == "unknown_job"
    with pytest.raises(ProviderHostError) as error:
        host.call("a", "result", job.job_id)
    assert error.value.code == "not_ready"
    host.pump()
    host.call("a", "cancel", job.job_id)
    assert host.cancels == [job.job_id]
    host.finish(job.job_id, "cancelled", result=solved(complete=False))
    assert host.call("a", "status", job.job_id).state == "cancelled"
    assert not host.call("a", "result", job.job_id).complete
    queued = host.call("a", "solve", SolveCommand("revision"))
    host.call("a", "cancel", queued.job_id)
    host.pump()
    assert host.starts == [job.job_id]


def test_missing_results_fail_completion_and_retention_is_bounded(host):
    jobs = []
    for _ in range(6):
        job = host.call("a", "solve", SolveCommand("revision"))
        jobs.append(job)
        host.pump()
        host.finish(job.job_id, "completed", result=solved())
    with pytest.raises(ProviderHostError) as error:
        host.call("a", "result", jobs[0].job_id)
    assert error.value.code == "result_unavailable"
    assert len(host._results) == 4
    job = host.call("a", "solve", SolveCommand("revision"))
    host.pump()
    host.finish(job.job_id, "completed")
    assert host.call("a", "status", job.job_id).state == "failed"


def test_callbacks_cannot_break_completion_and_can_unsubscribe(host):
    def broken(_event):
        raise ValueError("Provider bug")

    token = host.call("a", "subscribe", broken)
    host.geometry_accepted("a", "generation-1")
    assert host.events[-1].generation_request_id == "generation-1"
    assert host.events[-1].context.project_revision == "revision"
    host.call("a", "unsubscribe", token)
    host._start = lambda _: (_ for _ in ()).throw(ValueError("Bad physics"))
    job = host.call("a", "solve", SolveCommand("revision"))
    host.pump()
    assert host.call("a", "status", job.job_id).message == "Bad physics"


def test_revoked_context_cannot_publish_late_results(host):
    job = host.call("a", "solve", SolveCommand("revision"))
    host.pump()
    del host.contexts["a"]
    host.revoke("a")
    assert host.cancels == [job.job_id]
    host.finish(job.job_id, "completed", result=solved())
    assert host._jobs[job.job_id][2].state == "stale"
    assert not host._results
    host.close()
    with pytest.raises(ProviderHostError) as error:
        host.call("b", "context")
    assert error.value.code == "closed"


def test_commands_cannot_override_host_frequencies():
    with pytest.raises(TypeError):
        SolveCommand("revision", frequencies_hz=(500.0,))
    with pytest.raises(ValueError):
        SolveCommand("revision", schema_version=True)


def test_manual_host_result_is_available_without_a_provider_job(host):
    snapshot = solved()
    host._current_result = lambda: snapshot
    copy = host.call("a", "current_result")
    assert copy.run_id == snapshot.run_id
    copy.quantities.clear()
    assert snapshot.quantities


def test_busy_host_and_history_bounds(host):
    host._busy = lambda: True
    first = host.call("a", "solve", SolveCommand("revision"))
    host.pump()
    assert host.starts == []
    host.call("a", "cancel", first.job_id)
    for _ in range(40):
        job = host.call("a", "solve", SolveCommand("revision"))
        host.call("a", "cancel", job.job_id)
    assert len(host._jobs) == 32
    with pytest.raises(ProviderHostError) as error:
        host.call("a", "status", first.job_id)
    assert error.value.code == "unknown_job"
