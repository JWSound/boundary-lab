import tomllib
from pathlib import Path

from blab import __version__


def test_package_version_matches_project_metadata() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert __version__ == pyproject["project"]["version"]
