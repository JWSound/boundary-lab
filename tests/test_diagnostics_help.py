from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from blab.ui.diagnostics import collect_diagnostics, format_diagnostics
from blab.ui.help import _discover_docs, _doc_title
from blab.ui.settings import GuiPreferences


def test_diagnostics_report_includes_preferences_and_dependencies() -> None:
    diagnostics = collect_diagnostics(GuiPreferences(worker_count=7))
    text = format_diagnostics(diagnostics)

    assert "Boundary Lab:" in text
    assert "Python:" in text
    assert "Dependencies:" in text
    assert "worker_count: 7" in text


def test_help_doc_titles_come_from_first_heading(tmp_path: Path) -> None:
    doc = tmp_path / "multi-mesh.md"
    doc.write_text("# Multi Mesh Workflows\n\nPlaceholder.", encoding="utf-8")

    assert _doc_title(doc) == "Multi Mesh Workflows"
    assert _discover_docs(tmp_path) == {"Multi Mesh Workflows": doc}
