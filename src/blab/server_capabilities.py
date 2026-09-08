"""Background runtime discovery using isolated BEAT worker announcements."""

import threading

from blab.solvers.beat_engine_runtime import BeatEngineWorkerProcess


class BackendDiscovery:
    def __init__(self, backends, *, timeout=120):
        self.lock = threading.Lock()
        self.records = {
            key: {"available": False, "state": "checking", "reason": "Runtime check in progress."} for key in backends
        }
        self.workers = []
        self.threads = []
        for key, backend in backends.items():
            worker = BeatEngineWorkerProcess(
                julia_executable=backend.julia_executable,
                solver_script=backend.solver_script,
                julia_project=backend.julia_project,
                julia_threads=backend.julia_threads,
            )
            self.workers.append(worker)
            thread = threading.Thread(target=self.probe, args=(key, worker, timeout), daemon=True)
            self.threads.append(thread)
            thread.start()

    def probe(self, key, worker, timeout):
        expired = threading.Event()

        def stop_on_timeout():
            expired.set()
            worker.terminate()

        timer = threading.Timer(timeout, stop_on_timeout)
        timer.daemon = True
        timer.start()
        try:
            worker.ensure_started()
            info = worker.worker_info
            record = info["backends"].get(key.removeprefix("beat_"), {})
            available = record.get("available") is True
            result = {
                "available": available,
                "state": "ready",
                "reason": record.get("reason", "" if available else "Runtime unavailable."),
                "engine": info.get("engine"),
                "runtime": info.get("runtime"),
                "solve_kinds": info.get("solve_kinds", []),
            }
        except Exception as exc:
            result = {"available": False, "state": "ready", "reason": str(exc)}
        finally:
            timer.cancel()
            worker.terminate()
        if expired.is_set():
            result = {
                "available": False,
                "state": "ready",
                "reason": f"Runtime discovery timed out after {timeout:g} seconds.",
            }
        with self.lock:
            self.records[key] = result

    def snapshot(self):
        with self.lock:
            return {key: dict(value) for key, value in self.records.items()}

    def close(self):
        for worker in self.workers:
            worker.terminate()
        for thread in self.threads:
            thread.join(timeout=3)
