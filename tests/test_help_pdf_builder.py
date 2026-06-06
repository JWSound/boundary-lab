from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from scripts.build_help_pdf import GENERATED_NOTE, build_combined_markdown, discover_markdown_docs


def test_help_pdf_docs_use_stable_guide_order(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    advanced_dir = docs_dir / "advanced"
    advanced_dir.mkdir(parents=True)
    (advanced_dir / "cli-workflow.md").write_text("# CLI\n", encoding="utf-8")
    (docs_dir / "Model Assumptions.md").write_text("# Model\n", encoding="utf-8")
    (docs_dir / "User Guide.md").write_text("# User\n", encoding="utf-8")
    (docs_dir / "Inputs and Outputs.md").write_text("# IO\n", encoding="utf-8")

    docs = discover_markdown_docs(docs_dir)

    assert [path.relative_to(docs_dir).as_posix() for path in docs] == [
        "User Guide.md",
        "Inputs and Outputs.md",
        "Model Assumptions.md",
        "advanced/cli-workflow.md",
    ]


def test_help_pdf_combined_markdown_normalizes_images_and_math(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    guide = docs_dir / "User Guide.md"
    guide.write_text(
        'Before\n\n<img src="../assets/scripteditor.png" alt="Script Editor" width="300">\n\n'
        "Inline $k = omega / c$.\n\n$$\nq = dp/dn\n$$\n",
        encoding="utf-8",
    )

    markdown = build_combined_markdown([guide], docs_dir)

    assert GENERATED_NOTE in markdown
    assert (
        '<table width="100%" border="0" cellspacing="0" cellpadding="0"><tr><td align="center">'
        '<img src="assets/scripteditor.png" alt="Script Editor" width="200" />'
        '</td></tr></table>'
    ) in markdown
    assert "`k = omega / c`" in markdown
    assert "```math\nq = dp/dn\n```" in markdown
