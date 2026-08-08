from pathlib import Path

from scripts.build_help_pdf import (
    GENERATED_NOTE,
    _normalize_math_expression,
    _plain_math_expression,
    build_combined_markdown,
    discover_markdown_docs,
)


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
        "</td></tr></table>"
    ) in markdown
    assert "`k = omega / c`" in markdown
    assert "```math\nq = dp/dn\n```" in markdown


def test_help_pdf_math_normalization_and_matrix_fallback_are_readable() -> None:
    assert _normalize_math_expression(r"\mathbf n\mathbin{\cdot}\mathbf d") == (
        r"\mathbf{n}\cdot\mathbf{d}"
    )

    fallback = _plain_math_expression(
        "\\begin{bmatrix}\nA_F & 0 "
        + r"\\"
        + "\n0 & A_B\n\\end{bmatrix}"
        + "\\begin{bmatrix}\np_F "
        + r"\\"
        + "\np_B\n\\end{bmatrix}"
        + " = \\begin{bmatrix}\nf_v "
        + r"\\"
        + "\n0\n\\end{bmatrix}."
    )

    assert [" ".join(line.split()) for line in fallback.splitlines()] == [
        "[ A_F 0 [ p_F [ f_v",
        "0 A_B ] p_B ] = 0 ].",
    ]
    assert "\\begin" not in fallback
