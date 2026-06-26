"""Lightweight startup environment checks."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Literal

DEPENDENCY_NAMES = (
    "bempp-cl",
    "matplotlib",
    "meshio",
    "numpy",
    "pyopencl",
    "scipy",
    "PySide6",
    "pyvista",
    "pyvistaqt",
)

StartupCheckStatus = Literal["ok", "missing", "error"]


@dataclass(frozen=True)
class StartupCheckResult:
    name: str
    status: StartupCheckStatus
    detail: str

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"


def check_python_dependency_versions(
    dependency_names: tuple[str, ...] = DEPENDENCY_NAMES,
) -> tuple[StartupCheckResult, ...]:
    """Return installed package versions without importing the packages."""
    results: list[StartupCheckResult] = []
    for name in dependency_names:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            results.append(StartupCheckResult(name=name, status="missing", detail="not installed"))
        except Exception as exc:
            results.append(
                StartupCheckResult(
                    name=name,
                    status="error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            results.append(StartupCheckResult(name=name, status="ok", detail=version))
    return tuple(results)


def format_startup_check_results(results: tuple[StartupCheckResult, ...]) -> tuple[str, ...]:
    return tuple(f"{result.name}: {result.detail}" for result in results)
