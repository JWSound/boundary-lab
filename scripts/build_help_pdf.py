"""Build the bundled Boundary Lab help guide PDF from Markdown docs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtCore import QMarginsF, QSizeF, QUrl
from PySide6.QtGui import QGuiApplication, QPageLayout, QPageSize, QPdfWriter, QTextDocument

APP_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = APP_ROOT / "docs"
OUTPUT_PDF = DOCS_DIR / "Boundary Lab Guide.pdf"

PREFERRED_DOC_ORDER = (
    "User Guide.md",
    "Inputs and Outputs.md",
    "Model Assumptions.md",
)

_HTML_IMG_RE = re.compile(r"<img\s+([^>]*?)\s*/?>", re.IGNORECASE)
_HTML_ATTR_RE = re.compile(r"""([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*["']([^"']*)["']""")
_MATH_BLOCK_RE = re.compile(r"^\$\$\s*\n(.*?)\n\$\$\s*$", re.MULTILINE | re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^$\n]+)\$(?!\$)")
_MATRIX_RE = re.compile(r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}", re.DOTALL)

MAX_IMAGE_WIDTH_PT = 200
MAX_IMAGE_HEIGHT_PT = 200
EQUATION_RENDER_DPI = 220
EQUATION_FONT_SIZE_PT = 18
MAX_EQUATION_WIDTH_PT = 360
GENERATED_NOTE = "This guide was generated automatically from markdown files inside `/docs/`."

_PDF_STYLESHEET = """
body {
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 12pt;
    line-height: 1.42;
}
h1 {
    font-size: 30pt;
    margin-top: 0;
    margin-bottom: 18pt;
}
h2 {
    font-size: 22pt;
    margin-top: 26pt;
    margin-bottom: 12pt;
}
h3 {
    font-size: 17pt;
    margin-top: 20pt;
    margin-bottom: 8pt;
}
p {
    margin-top: 0;
    margin-bottom: 12pt;
}
ul, ol {
    margin-top: 3pt;
    margin-bottom: 12pt;
}
li {
    margin-bottom: 4pt;
}
code {
    font-family: "Cascadia Mono", "Consolas", monospace;
    background-color: #f1f3f5;
}
pre {
    font-family: "Cascadia Mono", "Consolas", monospace;
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    padding: 9pt;
}
img {
    max-width: 80%;
    max-height: __MAX_IMAGE_HEIGHT_PT__pt;
}
""".replace("__MAX_IMAGE_HEIGHT_PT__", str(MAX_IMAGE_HEIGHT_PT))


def discover_markdown_docs(docs_dir: Path = DOCS_DIR) -> list[Path]:
    """Return bundled Markdown docs in a stable guide order."""

    if not docs_dir.exists():
        return []

    docs = [path for path in docs_dir.rglob("*.md") if path.is_file()]
    by_relative = {path.relative_to(docs_dir).as_posix().casefold(): path for path in docs}
    ordered: list[Path] = []

    for name in PREFERRED_DOC_ORDER:
        path = by_relative.pop(name.casefold(), None)
        if path is not None:
            ordered.append(path)

    ordered.extend(path for _key, path in sorted(by_relative.items()))
    return ordered


def build_combined_markdown(
    paths: list[Path],
    docs_dir: Path = DOCS_DIR,
    math_dir: Path | None = None,
) -> str:
    sections = [f"# Boundary Lab Guide\n\n{GENERATED_NOTE}\n"]
    math_renderer = MathRenderer(math_dir) if math_dir is not None else None
    for path in paths:
        text = path.read_text(encoding="utf-8")
        text = _prepare_markdown(text, path.parent, docs_dir, math_renderer)
        sections.append(text.strip())
    return "\n\n".join(section for section in sections if section)


def write_pdf(markdown: str, output_pdf: Path = OUTPUT_PDF, docs_dir: Path = DOCS_DIR) -> None:
    app = QGuiApplication.instance()
    if app is None:
        app = QGuiApplication(sys.argv[:1])

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    writer = QPdfWriter(str(output_pdf))
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.Letter))
    writer.setPageMargins(QMarginsF(0.65, 0.65, 0.65, 0.65), QPageLayout.Unit.Inch)
    writer.setResolution(300)

    page_size = writer.pageLayout().paintRect(QPageLayout.Unit.Point).size()
    document = QTextDocument()
    document.setBaseUrl(QUrl.fromLocalFile(str(docs_dir.resolve().parent) + "/"))
    document.setDefaultStyleSheet(_PDF_STYLESHEET)
    document.setPageSize(QSizeF(page_size))
    document.setMarkdown(markdown, QTextDocument.MarkdownDialectGitHub)
    document.print_(writer)


def build_help_pdf(docs_dir: Path = DOCS_DIR, output_pdf: Path = OUTPUT_PDF) -> list[Path]:
    docs = discover_markdown_docs(docs_dir)
    if not docs:
        raise FileNotFoundError(f"No Markdown docs found in {docs_dir}")
    with TemporaryDirectory(prefix="boundary-lab-guide-math-") as math_tmp:
        markdown = build_combined_markdown(docs, docs_dir, Path(math_tmp))
        write_pdf(markdown, output_pdf, docs_dir)
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR, help="Directory containing Markdown docs.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PDF, help="PDF file to write.")
    args = parser.parse_args(argv)

    docs = build_help_pdf(args.docs_dir, args.output)
    print(f"Wrote {args.output} from {len(docs)} Markdown files.")
    return 0


class MathRenderer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def render(self, expression: str) -> RenderedEquation:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from PySide6.QtGui import QImage

        self.count += 1
        output_path = self.output_dir / f"equation-{self.count:03d}.png"
        normalized = _normalize_math_expression(expression)

        figure = Figure(figsize=(8.0, 1.2), dpi=EQUATION_RENDER_DPI)
        figure.patch.set_alpha(0)
        FigureCanvasAgg(figure)
        figure.text(0.5, 0.5, f"${normalized}$", ha="center", va="center", fontsize=EQUATION_FONT_SIZE_PT)
        try:
            figure.savefig(output_path, transparent=True, bbox_inches="tight", pad_inches=0.08)
        except ValueError:
            # Matplotlib mathtext intentionally supports only a LaTeX subset
            # (notably excluding matrix environments). Keep the guide build
            # reproducible and the equation readable when a source document
            # uses a construct outside that subset.
            fallback = _plain_math_expression(expression)
            line_count = max(1, fallback.count("\n") + 1)
            figure.clear()
            figure.set_size_inches(8.0, max(1.2, 0.27 * line_count + 0.4))
            figure.text(
                0.5,
                0.5,
                fallback,
                ha="center",
                va="center",
                family="monospace",
                fontsize=11,
            )
            figure.savefig(output_path, transparent=True, bbox_inches="tight", pad_inches=0.08)
        image_width_px = QImage(str(output_path)).width()
        width_pt = min(MAX_EQUATION_WIDTH_PT, max(1, round(image_width_px * 72 / EQUATION_RENDER_DPI)))
        return RenderedEquation(output_path, width_pt)


@dataclass(frozen=True)
class RenderedEquation:
    path: Path
    width_pt: int


def _prepare_markdown(
    markdown: str,
    base_dir: Path,
    docs_dir: Path,
    math_renderer: MathRenderer | None = None,
) -> str:
    markdown = _HTML_IMG_RE.sub(lambda match: _html_image_as_markdown(match, base_dir, docs_dir), markdown)
    markdown = _rewrite_markdown_image_paths(markdown, base_dir, docs_dir)
    if math_renderer is not None:
        markdown = _MATH_BLOCK_RE.sub(lambda match: _math_block_as_image(match, math_renderer), markdown)
    else:
        markdown = _MATH_BLOCK_RE.sub(_math_block_as_code, markdown)
    markdown = _INLINE_MATH_RE.sub(r"`\1`", markdown)
    return markdown


def _html_image_as_markdown(match: re.Match[str], base_dir: Path, docs_dir: Path) -> str:
    attrs = {key.casefold(): value for key, value in _HTML_ATTR_RE.findall(match.group(1))}
    src = attrs.get("src")
    if not src:
        return ""
    alt = attrs.get("alt", "")
    width = attrs.get("width")
    return _centered_image_tag(_guide_relative_path(src, base_dir, docs_dir), alt, width)


def _rewrite_markdown_image_paths(markdown: str, base_dir: Path, docs_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        alt = match.group(1)
        target = match.group(2).strip()
        return _centered_image_tag(_guide_relative_path(target, base_dir, docs_dir), alt)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace, markdown)


def _guide_relative_path(target: str, base_dir: Path, docs_dir: Path) -> str:
    if "://" in target or target.startswith("#"):
        return target
    clean_target = target.strip("<>")
    absolute = (base_dir / clean_target).resolve()
    try:
        return absolute.relative_to(docs_dir.resolve()).as_posix()
    except ValueError:
        try:
            return absolute.relative_to(docs_dir.resolve().parent).as_posix()
        except ValueError:
            try:
                return absolute.relative_to(APP_ROOT.resolve()).as_posix()
            except ValueError:
                return absolute.as_posix()


def _math_block_as_code(match: re.Match[str]) -> str:
    body = match.group(1).strip("\n")
    return f"```math\n{body}\n```"


def _math_block_as_image(match: re.Match[str], renderer: MathRenderer) -> str:
    equation = renderer.render(match.group(1))
    return _centered_image_tag(
        equation.path.as_posix(),
        "Equation",
        str(equation.width_pt),
        max_width_pt=MAX_EQUATION_WIDTH_PT,
    )


def _normalize_math_expression(expression: str) -> str:
    normalized = " ".join(line.strip() for line in expression.strip().splitlines() if line.strip())
    normalized = normalized.replace(r"\lVert", r"\Vert").replace(r"\rVert", r"\Vert")
    normalized = normalized.replace(r"\mathbin{\cdot}", r"\cdot")
    return re.sub(r"\\mathbf\s+([A-Za-z])", r"\\mathbf{\1}", normalized)


def _plain_math_expression(expression: str) -> str:
    """Return a readable text fallback for unsupported display math."""

    matrix_matches = list(_MATRIX_RE.finditer(expression))
    if matrix_matches:
        return _plain_matrix_equation(expression, matrix_matches)
    return _plain_math_text(expression)


def _plain_matrix_equation(expression: str, matches: list[re.Match[str]]) -> str:
    matrices = [_plain_matrix(match.group(1)) for match in matches]
    separators = []
    for left, right in zip(matches, matches[1:]):
        gap = expression[left.end() : right.start()]
        separators.append(" = " if "=" in gap else "   ")

    height = max(len(matrix) for matrix in matrices)
    widths = [max(len(line) for line in matrix) for matrix in matrices]
    rendered = []
    for row in range(height):
        parts = []
        for index, matrix in enumerate(matrices):
            parts.append((matrix[row] if row < len(matrix) else "").ljust(widths[index]))
            if index < len(separators):
                separator = separators[index]
                parts.append(separator if separator.strip() != "=" or row == height // 2 else " " * len(separator))
        rendered.append("".join(parts).rstrip())

    suffix = expression[matches[-1].end() :].strip()
    if suffix:
        rendered[-1] += suffix
    return "\n".join(rendered)


def _plain_matrix(body: str) -> list[str]:
    rows = [row.strip() for row in body.strip().split(r"\\") if row.strip()]
    cells = [[_plain_math_text(cell.strip()) for cell in row.split("&")] for row in rows]
    column_count = max(len(row) for row in cells)
    widths = [
        max(len(row[column]) if column < len(row) else 0 for row in cells)
        for column in range(column_count)
    ]
    content = [
        "  ".join(
            (row[column] if column < len(row) else "").rjust(widths[column])
            for column in range(column_count)
        )
        for row in cells
    ]
    if len(content) == 1:
        return [f"[ {content[0]} ]"]
    return [
        ("[ " if index == 0 else "  ") + line + (" ]" if index == len(content) - 1 else "")
        for index, line in enumerate(content)
    ]


def _plain_math_text(expression: str) -> str:
    plain = expression.strip()
    plain = plain.replace(r"\\", "\n").replace("&", "   ")
    plain = plain.replace(r"\,", " ").replace(r"\left", "").replace(r"\right", "")
    for command, replacement in (
        (r"\rho", "rho"),
        (r"\omega", "omega"),
        (r"\Gamma", "Gamma"),
        (r"\sum", "sum"),
        (r"\int", "integral"),
        (r"\cdot", "*"),
    ):
        plain = plain.replace(command, replacement)
    plain = re.sub(r"\\(?:mathrm|mathbf|mathsf)\{([^{}]*)\}", r"\1", plain)
    return plain


def _centered_image_tag(
    src: str,
    alt: str,
    width: str | None = None,
    max_width_pt: int = MAX_IMAGE_WIDTH_PT,
) -> str:
    escaped_src = src.replace('"', "%22")
    escaped_alt = alt.replace('"', "&quot;")
    width_value = _bounded_image_width(width, max_width_pt)
    width_attr = f' width="{width_value}"' if width_value else ""
    return (
        f'\n\n<table width="100%" border="0" cellspacing="0" cellpadding="0">'
        f'<tr><td align="center">'
        f'<img src="{escaped_src}" alt="{escaped_alt}"{width_attr} />'
        f"</td></tr></table>\n\n"
    )


def _bounded_image_width(width: str | None, max_width_pt: int) -> int | None:
    if width is None:
        return None
    try:
        return min(max(1, int(float(width))), max_width_pt)
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
