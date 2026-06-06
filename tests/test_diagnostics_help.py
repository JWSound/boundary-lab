import pytest

pytest.importorskip("PySide6")

from blab.ui.diagnostics import collect_diagnostics, format_diagnostics
from blab.ui.settings import GuiPreferences


def test_diagnostics_report_includes_preferences_and_dependencies() -> None:
    diagnostics = collect_diagnostics(GuiPreferences(worker_count=7))
    text = format_diagnostics(diagnostics)

    assert "Boundary Lab:" in text
    assert "Python:" in text
    assert "Dependencies:" in text
    assert "worker_count: 7" in text

