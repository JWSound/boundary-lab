import tomllib
from pathlib import Path


def test_hornlab_integrations_are_declared_as_public_dependencies() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert pyproject["tool"]["hatch"]["metadata"]["allow-direct-references"] is True

    mesher = next(dep for dep in dependencies if dep.startswith("hornlab-waveguide-mesher @ "))
    metal = next(dep for dep in dependencies if dep.startswith("hornlab-metal-bem @ "))

    assert "git+https://github.com/m3gnus/hornlab-waveguide-mesher.git@" in mesher
    assert "git+https://github.com/m3gnus/hornlab-metal-bem.git@" in metal
    # Windows installs must not pull either package.
    assert "platform_system != 'Windows'" in mesher
    assert "platform_system == 'Darwin'" in metal
    assert "platform_machine == 'arm64'" in metal
    # Never re-introduce machine-local editable paths.
    assert "/Users/" not in "\n".join(dependencies)
