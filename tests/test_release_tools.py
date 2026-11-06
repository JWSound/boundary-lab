import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def release_tools(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/release_tools.py"
    spec = importlib.util.spec_from_file_location("release_tools", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    pin = (
        "beat-engine @ https://github.com/JWSound/BEAT_Engine/releases/download/"
        "v0.6.0/beat_engine-0.6.0-py3-none-any.whl#sha256=" + "a" * 64
    )
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.5.0"\ndependencies = [' + json.dumps(pin) + "]\n")
    (tmp_path / "docs/releases").mkdir(parents=True)
    (tmp_path / "docs/releases/0.5.0.md").write_text("# Release\n")
    (tmp_path / "packaging").mkdir()
    (tmp_path / "packaging/runtime-lock.json").write_text(json.dumps({"beat_requirement": pin}))
    (tmp_path / "src/blab").mkdir(parents=True)
    (tmp_path / "src/blab/__init__.py").write_text('__version__ = "0.5.0"\n')
    return module


def test_release_checks_runtime_lock_without_desktop(release_tools):
    release_tools.check("v0.5.0")
    (release_tools.ROOT / "packaging/runtime-lock.json").write_text(
        json.dumps({"beat_requirement": "beat-engine @ git+https://example.com/old"})
    )
    with pytest.raises(ValueError, match="bundled runtime BEAT pins"):
        release_tools.check("v0.5.0")


def test_release_rejects_stale_application_version(release_tools):
    (release_tools.ROOT / "src/blab/__init__.py").write_text('__version__ = "0.5.0.dev0"\n')
    with pytest.raises(ValueError, match="Application runtime version"):
        release_tools.check("v0.5.0")
