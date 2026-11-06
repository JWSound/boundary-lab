from blab.warm_up_history import WarmUpHistory, WarmUpKey

CPU_COLD = WarmUpKey(backend="cpu", engine_version="0.4.0", julia_threads="8", cold_worker=True)
CPU_WARM = WarmUpKey(backend="cpu", engine_version="0.4.0", julia_threads="8", cold_worker=False)


def test_prediction_is_median_of_last_five_warm_ups(tmp_path) -> None:
    history = WarmUpHistory(tmp_path / "history.json")
    assert history.predict(CPU_COLD) is None
    for duration_s in (500.0, 10.0, 90.0, 80.0, 70.0, 60.0):
        history.record(CPU_COLD, duration_s)

    assert WarmUpHistory(tmp_path / "history.json").predict(CPU_COLD) == 70.0


def test_history_keeps_keys_separate(tmp_path) -> None:
    history = WarmUpHistory(tmp_path / "history.json")
    history.record(CPU_COLD, 90.0)
    history.record(CPU_WARM, 5.0)
    history.record(WarmUpKey("cpu", "0.5.0", "8", True), 120.0)

    assert history.predict(CPU_COLD) == 90.0
    assert history.predict(CPU_WARM) == 5.0
    assert history.predict(WarmUpKey("cpu", "0.4.0", "4", True)) is None


def test_history_keeps_last_twenty_entries_per_key(tmp_path) -> None:
    history = WarmUpHistory(tmp_path / "history.json")
    for duration_s in range(25):
        history.record(CPU_COLD, float(duration_s))

    assert history.durations_s(CPU_COLD) == [float(value) for value in range(5, 25)]


def test_corrupt_history_is_ignored_and_replaced(tmp_path) -> None:
    path = tmp_path / "history.json"
    path.write_text("{not json", encoding="utf-8")
    history = WarmUpHistory(path)
    assert history.predict(CPU_COLD) is None

    history.record(CPU_COLD, 42.0)

    assert WarmUpHistory(path).predict(CPU_COLD) == 42.0


def test_unexpected_history_shape_is_ignored(tmp_path) -> None:
    path = tmp_path / "history.json"
    path.write_text('{"entries": {"cpu|0.4.0|8|cold": "oops"}}', encoding="utf-8")
    assert WarmUpHistory(path).predict(CPU_COLD) is None
