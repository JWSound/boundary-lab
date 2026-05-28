from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from blab.ui.diagnostics import collect_diagnostics, format_diagnostics
from blab.ui.help import _discover_docs, _doc_title, _document_base_url, _markdown_for_qt
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


def test_help_doc_title_uses_user_guide_filename(tmp_path: Path) -> None:
    doc = tmp_path / "User Guide.md"
    doc.write_text("# Main Window\n\nGuide contents.", encoding="utf-8")

    assert _doc_title(doc) == "User Guide"
    assert _discover_docs(tmp_path) == {"User Guide": doc}


def test_help_document_base_url_points_to_document_directory(tmp_path: Path) -> None:
    doc = tmp_path / "guide.md"
    doc.write_text("# Guide\n\n![Image](../assets/example.png)", encoding="utf-8")

    assert _document_base_url(doc).toLocalFile() == f"{tmp_path.as_posix()}/"


def test_help_markdown_prepares_math_for_qt_renderer() -> None:
    markdown = "Inline $k = omega / c$.\n\n$$\nq = dp/dn\n$$\n"

    prepared = _markdown_for_qt(markdown)

    assert "`k = omega / c`" in prepared
    assert "```math\nq = dp/dn\n```" in prepared


def test_help_markdown_converts_html_images_to_markdown_images() -> None:
    markdown = 'Before\n\n<img src="../assets/scripteditor.png" alt="Script Editor" width="300">\n\nAfter'

    prepared = _markdown_for_qt(markdown)

    assert '<img src="../assets/scripteditor.png"' not in prepared
    assert "![Script Editor](../assets/scripteditor.png)" in prepared
    assert "Before" in prepared
    assert "After" in prepared
