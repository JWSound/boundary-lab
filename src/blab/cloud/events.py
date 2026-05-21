"""Durable cloud solve event stores."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class EventRecord:
    seq: int
    event: dict


class EventStore(Protocol):
    def append(self, job_id: str, event: dict) -> int:
        """Store an event and return its sequence number."""

    def list_after(self, job_id: str, seq: int) -> list[EventRecord]:
        """Return events for a job with sequence numbers greater than seq."""


class LocalEventStore:
    """JSONL-backed event store for local development and tests."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id / "events.jsonl"

    def append(self, job_id: str, event: dict) -> int:
        records = self.list_after(job_id, -1)
        seq = 1 if not records else records[-1].seq + 1
        path = self._path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"seq": seq, "event": event}, separators=(",", ":")))
            handle.write("\n")
        return seq

    def list_after(self, job_id: str, seq: int) -> list[EventRecord]:
        path = self._path(job_id)
        if not path.is_file():
            return []
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                item_seq = int(item["seq"])
                if item_seq > seq:
                    records.append(EventRecord(seq=item_seq, event=item["event"]))
        return records


class DynamoDbEventStore:
    """DynamoDB event store keyed by (job_id, seq)."""

    def __init__(self, table_name: str, *, resource=None):
        self.table_name = table_name
        self._resource = resource
        self._table = None

    @property
    def table(self):
        if self._table is None:
            if self._resource is None:
                try:
                    import boto3
                except ImportError as exc:  # pragma: no cover - optional AWS extra
                    raise RuntimeError('Install AWS dependencies with: python -m pip install -e ".[aws]"') from exc
                self._resource = boto3.resource("dynamodb")
            self._table = self._resource.Table(self.table_name)
        return self._table

    def append(self, job_id: str, event: dict) -> int:
        seq = time.time_ns()
        self.table.put_item(Item={"job_id": job_id, "seq": seq, "event": event})
        return seq

    def list_after(self, job_id: str, seq: int) -> list[EventRecord]:
        try:
            from boto3.dynamodb.conditions import Key
        except ImportError as exc:  # pragma: no cover - optional AWS extra
            raise RuntimeError('Install AWS dependencies with: python -m pip install -e ".[aws]"') from exc

        response = self.table.query(
            KeyConditionExpression=Key("job_id").eq(job_id) & Key("seq").gt(seq),
            ScanIndexForward=True,
        )
        return [
            EventRecord(seq=int(item["seq"]), event=item["event"])
            for item in response.get("Items", [])
        ]


class EventStoreSink:
    def __init__(self, store: EventStore, job_id: str):
        self.store = store
        self.job_id = job_id

    def emit(self, event: dict) -> None:
        self.store.append(self.job_id, event)
