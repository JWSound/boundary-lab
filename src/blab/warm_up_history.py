"""Qt-free record of past solver warm-ups, used to predict the next one."""

from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

MAX_ENTRIES_PER_KEY = 20
PREDICTION_WINDOW = 5


@dataclass(frozen=True)
class WarmUpKey:
    """Setup that determines how long a warm-up takes; mesh size is deliberately ignored."""

    backend: str
    engine_version: str
    julia_threads: str
    cold_worker: bool

    def text(self) -> str:
        return "|".join((self.backend, self.engine_version, self.julia_threads, "cold" if self.cold_worker else "warm"))


def default_history_path() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    base = Path(root) / "Boundary Lab" if root else Path.home() / ".boundary-lab"
    return base / "warm_up_history.json"


class WarmUpHistory:
    def __init__(self, path: Path | None = None):
        self.path = default_history_path() if path is None else path

    def durations_s(self, key: WarmUpKey) -> list[float]:
        return self._load().get(key.text(), [])

    def predict(self, key: WarmUpKey) -> float | None:
        durations_s = self.durations_s(key)[-PREDICTION_WINDOW:]
        return statistics.median(durations_s) if durations_s else None

    def record(self, key: WarmUpKey, duration_s: float) -> None:
        entries = self._load()
        entries[key.text()] = [*entries.get(key.text(), []), float(duration_s)][-MAX_ENTRIES_PER_KEY:]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
        except OSError:
            logging.getLogger(__name__).warning("Could not save warm-up history to %s", self.path, exc_info=True)

    def _load(self) -> dict[str, list[float]]:
        # A missing, unreadable or corrupt history only costs the prediction.
        try:
            entries = json.loads(self.path.read_text(encoding="utf-8"))["entries"]
            return {
                str(key): [float(value) for value in values]
                for key, values in entries.items()
                if isinstance(values, list)
            }
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return {}
