from pathlib import Path

import pytest

from blab.ath_setup import discover_wine, validate_paths


def tools_at(root):
    ath = root / "Ath tool.exe"
    wine = root / "Wine executable"
    ath.write_text("ath")
    wine.write_text("wine")
    wine.chmod(0o700)
    return ath, wine


def test_paths_require_explicit_tools_and_preserve_spaces(tmp_path):
    ath, wine = tools_at(tmp_path)
    assert validate_paths(str(ath), str(wine)) == (ath.resolve(), wine.resolve())
    with pytest.raises(ValueError, match="setup required"):
        validate_paths("", str(wine))
    with pytest.raises(ValueError, match="Wine"):
        validate_paths(str(ath), str(tmp_path))
    ath.unlink()
    with pytest.raises(ValueError, match="Ath .exe"):
        validate_paths(str(ath), str(wine))


def test_discovery_uses_executable_on_path(tmp_path, monkeypatch):
    _, wine = tools_at(tmp_path)
    monkeypatch.setattr("blab.ath_setup.shutil.which", lambda _: str(wine))
    assert discover_wine() == str(wine)


def test_explicit_wine_does_not_depend_on_path(monkeypatch):
    from blab.ath import _ath_process_command

    monkeypatch.setattr("blab.ath.sys.platform", "darwin")
    monkeypatch.setattr("blab.ath.shutil.which", lambda _: None)
    assert _ath_process_command(Path("Ath tool.exe"), Path("test file.cfg"), "/chosen/wine") == [
        "/chosen/wine",
        "Ath tool.exe",
        "test file.cfg",
        "-b",
    ]
    monkeypatch.setattr("blab.ath.sys.platform", "win32")
    assert _ath_process_command(Path("ath.exe"), Path("test.cfg"), "/chosen/wine") == ["ath.exe", "test.cfg", "-b"]
