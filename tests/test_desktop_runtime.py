import json
import os
from pathlib import Path

from blab.desktop_runtime import configure_runtime, documents_directory, seed_documents


def test_documents_folder_is_absolute():
    assert documents_directory().is_absolute()


def test_seed_documents_preserves_edits_and_adds_missing_files(tmp_path):
    source = tmp_path / "source"
    (source / "examples").mkdir(parents=True)
    (source / "documentation").mkdir()
    (source / "examples/test.cfg").write_text("original")
    (source / "documentation/guide.txt").write_text("guide")
    destination = tmp_path / "Documents/Boundary Lab"
    seed_documents(source, destination)
    (destination / "examples/test.cfg").write_text("user edit")
    (source / "examples/added.cfg").write_text("new example")
    seed_documents(source, destination)
    assert (destination / "examples/test.cfg").read_text() == "user edit"
    assert (destination / "examples/added.cfg").read_text() == "new example"
    assert (destination / "documentation/guide.txt").read_text() == "guide"
    assert (destination / "runs").is_dir()


def test_runtime_isolation_and_redirected_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JULIA_PROJECT", "wrong-project")
    monkeypatch.setenv("JULIA_DEPOT_PATH", "wrong-depot")
    monkeypatch.setenv("BLAB_JULIA_EXE", "wrong-julia")
    monkeypatch.setenv("PYTHONPATH", "wrong-python")
    monkeypatch.setenv("QT_PLUGIN_PATH", "wrong-qt")
    root = tmp_path / "Program Files/Boundary Lab"
    root.mkdir(parents=True)
    (root / "runtime-manifest.json").write_text(json.dumps({"runtime_id": "test", "backends": ["cpu"]}))
    docs = tmp_path / "OneDrive/Documents"
    local = tmp_path / "Local AppData"
    configure_runtime(root, documents=docs, local_data=local)
    assert Path.cwd() == docs / "Boundary Lab"
    assert Path(os.environ["BLAB_JULIA_EXE"]).is_relative_to(root)
    assert os.environ["BLAB_PACKAGED_BACKENDS"] == "cpu"
    assert "wrong" not in os.environ["JULIA_DEPOT_PATH"]
    assert "JULIA_PROJECT" not in os.environ
    assert "PYTHONPATH" not in os.environ
    assert "QT_PLUGIN_PATH" not in os.environ
    assert (local / "Boundary Lab/runtime/test/julia-depot").is_dir()
    assert not (root / "runs").exists()


def test_cpu_package_never_probes_cuda_and_rejects_unbundled_backend(monkeypatch):
    import pytest

    from blab.headless import resolve_headless_backend
    from blab.solvers.registry import available_backend_infos
    from blab.ui.settings import GuiPreferences

    monkeypatch.setenv("BLAB_PACKAGED_BACKENDS", "cpu")
    monkeypatch.setattr("blab.headless._command_available", lambda _: pytest.fail("CPU package probed CUDA"))
    assert resolve_headless_backend() == "beat_cpu"
    assert {info.backend_id for info in available_backend_infos()} == {"beat_cpu", "beat_remote"}
    assert GuiPreferences(solve_backend="beat_cuda").solve_backend == "beat_cpu"
    assert GuiPreferences(solve_backend="beat_remote").solve_backend == "beat_remote"
    with pytest.raises(ValueError, match="not included"):
        resolve_headless_backend("beat_cuda")
