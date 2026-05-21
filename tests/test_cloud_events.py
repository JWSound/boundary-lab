from pathlib import Path

from blab.cloud.events import EventStoreSink, LocalEventStore


def test_local_event_store_appends_and_lists_after_sequence(tmp_path: Path) -> None:
    store = LocalEventStore(tmp_path)

    first = store.append("job_test", {"type": "status", "job_id": "job_test", "message": "one"})
    second = store.append("job_test", {"type": "completed", "job_id": "job_test"})

    assert first == 1
    assert second == 2
    assert [record.seq for record in store.list_after("job_test", 0)] == [1, 2]
    assert [record.event["type"] for record in store.list_after("job_test", 1)] == ["completed"]


def test_event_store_sink_writes_events(tmp_path: Path) -> None:
    store = LocalEventStore(tmp_path)
    sink = EventStoreSink(store, "job_test")

    sink.emit({"type": "failed", "job_id": "job_test", "message": "boom"})

    records = store.list_after("job_test", 0)
    assert len(records) == 1
    assert records[0].event["message"] == "boom"
