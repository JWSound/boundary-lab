"""Availability gate for the GUI test stack.

``pytest.importorskip("PySide6")`` used to guard every GUI test module. It is
the wrong tool here: PySide6 is a hard dependency of the ``[gui]`` extra this
project installs, so a failed import means the environment is broken, not that
the tests are inapplicable. Measured 2026-08-07, ``importorskip`` gated **154
tests, roughly a third of the suite** — all of which would have silently turned
into skips on a bad wheel or a missing system library, leaving a green run.

So a missing GUI stack is a **loud collection error** by default. Setting
``BLAB_SKIP_GUI_TESTS=1`` downgrades it to an explicit skip, for a deliberately
headless environment that cannot install Qt. The difference from ``importorskip``
is that somebody has to *choose* it.

This module imports no Qt itself, so it stays importable in exactly the broken
environment it exists to report on.
"""

from __future__ import annotations

import importlib
import os

import pytest

#: Set to "1" to downgrade a missing GUI stack from an error to a skip.
SKIP_GUI_ENV = "BLAB_SKIP_GUI_TESTS"


INSTALL_HINT = 'pip install -e ".[gui]"'


def import_error(module_name: str) -> BaseException | None:
    """The exception that stopped ``module_name`` importing, or None."""
    try:
        importlib.import_module(module_name)
    except BaseException as exc:  # noqa: BLE001 - report anything that stops the import
        return exc
    return None


PYSIDE6_ERROR = import_error("PySide6")


def skip_requested() -> bool:
    return os.environ.get(SKIP_GUI_ENV) == "1"


def missing_stack_error(module_name: str, error: BaseException) -> RuntimeError:
    return RuntimeError(
        f"{module_name} could not be imported, so the GUI tests cannot run: {error!r}\n"
        f"This is an environment failure, not a reason to skip. Install it with "
        f"{INSTALL_HINT}, or set {SKIP_GUI_ENV}=1 to skip the GUI suite deliberately."
    )


#: Importing any of these pulls Qt in, directly or transitively. ``blab.ui`` is
#: on the list because most of it reaches Qt a couple of imports down — naming
#: only ``PySide6`` missed ten modules that still failed to collect.
#: Named here rather than inline so the suite-health test can refer to it
#: without spelling the literal, which would make that file match its own
#: Qt-dependency heuristic and get ignored in degraded mode.
PYSIDE6_MODULE = "PySide6"

QT_DEPENDENT_IMPORTS = (
    PYSIDE6_MODULE,
    "blab.ui",
    "pyvistaqt",
    # scripts/build_help_pdf.py renders through Qt's GUI/PDF stack, so its test
    # needs Qt without ever naming it.
    "build_help_pdf",
)


def gui_test_modules(tests_dir) -> list[str]:
    """Test module filenames that need Qt, found by what they import.

    Derived rather than hardcoded so a new GUI test module is covered the day
    it is written, with no list to remember to update. Deliberately generous:
    over-ignoring in an already-degraded environment costs coverage that the
    environment could not have produced anyway, while under-ignoring turns the
    opt-out into a wall of collection errors.
    """
    return sorted(
        path.name
        for path in tests_dir.glob("test_*.py")
        if any(needle in path.read_text(encoding="utf-8") for needle in QT_DEPENDENT_IMPORTS)
    )


def require_pyvistaqt() -> None:
    """Fail loudly (or skip, if explicitly opted out) when pyvistaqt is missing.

    Called from inside the fixture that needs it, so the 3D plotting stack can
    be missing without taking the rest of the GUI suite with it.
    """
    error = import_error("pyvistaqt")
    if error is None:
        return
    if skip_requested():
        pytest.skip(f"pyvistaqt unavailable and {SKIP_GUI_ENV}=1: {error}")
    raise missing_stack_error("pyvistaqt", error)
