import importlib.metadata

from blab.startup_checks import (
    StartupCheckResult,
    check_python_dependency_versions,
    format_startup_check_results,
)


def test_check_python_dependency_versions_reports_versions(monkeypatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda name: f"{name}-1.2.3")

    results = check_python_dependency_versions(("numpy", "scipy"))

    assert results == (
        StartupCheckResult(name="numpy", status="ok", detail="numpy-1.2.3"),
        StartupCheckResult(name="scipy", status="ok", detail="scipy-1.2.3"),
    )


def test_check_python_dependency_versions_reports_missing(monkeypatch) -> None:
    def fake_version(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    results = check_python_dependency_versions(("not-a-package",))

    assert results == (StartupCheckResult(name="not-a-package", status="missing", detail="not installed"),)


def test_format_startup_check_results() -> None:
    lines = format_startup_check_results(
        (
            StartupCheckResult(name="numpy", status="ok", detail="2.4.4"),
            StartupCheckResult(name="pyvista", status="missing", detail="not installed"),
        )
    )

    assert lines == ("numpy: 2.4.4", "pyvista: not installed")
