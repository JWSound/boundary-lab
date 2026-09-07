"""Runtime identity survives canonicalization and snapshot boundaries."""

from blab.solve_results import SolvedSystemBuilder, SolveProvenance
from blab.system_contract import SystemFrequencyResult


def test_builder_preserves_distinct_engine_runs_and_isolates_input_mutation():
    builder = SolvedSystemBuilder(
        frequencies_hz=(100.0, 200.0),
        excitation_ids=("port",),
        provenance=SolveProvenance(backend_id="beat_cpu", solve_kind="exterior_bem"),
        domains=(),
    )
    provenance = {
        "schema_version": 1,
        "engine": {"repository_revision": None, "source_sha256": "first"},
        "execution": {"backend": "cpu", "precision": "float32"},
    }
    result = SystemFrequencyResult(
        freq_hz=100.0, excitation_port_ids=("port",), quantities=(), diagnostics={"engine_provenance": provenance}
    )
    builder.add(result)
    builder.add(result)
    first = builder.snapshot(status="partial")
    assert len(first.provenance.engine_runs) == 1
    provenance["engine"]["source_sha256"] = "second"
    builder.add(
        SystemFrequencyResult(
            freq_hz=200.0, excitation_port_ids=("port",), quantities=(), diagnostics={"engine_provenance": provenance}
        )
    )
    completed = builder.finalize(status="complete")
    assert first.provenance.engine_runs[0]["engine"]["source_sha256"] == "first"
    assert len(completed.provenance.engine_runs) == 2
    assert completed.provenance.engine_runs[1]["engine"]["source_sha256"] == "second"


def test_legacy_provenance_is_explicitly_empty():
    assert SolveProvenance(backend_id="beat_cpu", solve_kind="exterior_bem").engine_runs == ()
