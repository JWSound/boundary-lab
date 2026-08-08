"""Syntax colouring for Ath ``.cfg`` sources.

Scanned rather than regex-matched, so ``Delimiter = ";"`` stays a string.
Nothing spans lines, so the scan keeps no state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from PySide6.QtGui import QColor, QPalette, QSyntaxHighlighter, QTextCharFormat


class AthToken(StrEnum):
    KEY = "key"
    LABEL = "label"
    NUMBER = "number"
    STRING = "string"
    COMMENT = "comment"
    FUNCTION = "function"
    OPERATOR = "operator"


@dataclass(frozen=True)
class AthSpan:
    start: int
    length: int
    token: AthToken


@dataclass(frozen=True)
class AthSyntaxColors:
    key: str
    label: str
    number: str
    string: str
    comment: str
    function: str
    operator: str


ATH_LIGHT_COLORS = AthSyntaxColors(
    key="#0033B3",
    label="#00807F",
    number="#AA5500",
    string="#A31515",
    comment="#3F7F5F",
    function="#7B3FA0",
    operator="#5C5C5C",
)

ATH_DARK_COLORS = AthSyntaxColors(
    key="#6FAEE8",
    label="#4EC9B0",
    number="#DCC48A",
    string="#D98D6E",
    comment="#7EA86A",
    function="#C99AD1",
    operator="#B4B4B4",
)

MATH_FUNCTIONS = frozenset(
    {
        "abs",
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "cos",
        "cosh",
        "exp",
        "floor",
        "log",
        "log10",
        "max",
        "min",
        "pow",
        "round",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
)

# Dots are part of the key: ``Mesh.ThroatResolution``.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_NUMBER_RE = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_OPERATORS = frozenset("={}(),:+-*/^")


def tokenize_ath_line(line: str) -> tuple[AthSpan, ...]:
    """Split one line into non-overlapping coloured spans.

    Bare identifiers such as ``p1`` get no span and stay plain.
    """
    spans: list[AthSpan] = []
    end_of_line = len(line)
    index = 0
    leading_identifier_seen = False
    previous_operator = ""

    while index < end_of_line:
        char = line[index]
        if char.isspace():
            index += 1
            continue

        if char == ";":
            spans.append(AthSpan(index, end_of_line - index, AthToken.COMMENT))
            break

        if char == '"':
            closing = line.find('"', index + 1)
            end = end_of_line if closing < 0 else closing + 1
            spans.append(AthSpan(index, end - index, AthToken.STRING))
            index = end
            previous_operator = ""
            continue

        identifier = _IDENTIFIER_RE.match(line, index)
        if identifier is not None:
            token = _classify_identifier(
                identifier.group(),
                line,
                identifier.end(),
                after_colon=previous_operator == ":",
                leading=not leading_identifier_seen,
            )
            if token is not None:
                spans.append(AthSpan(index, identifier.end() - index, token))
            leading_identifier_seen = True
            previous_operator = ""
            index = identifier.end()
            continue

        number = _NUMBER_RE.match(line, index)
        if number is not None:
            spans.append(AthSpan(index, number.end() - index, AthToken.NUMBER))
            previous_operator = ""
            index = number.end()
            continue

        if char in _OPERATORS:
            spans.append(AthSpan(index, 1, AthToken.OPERATOR))
            previous_operator = char
        index += 1

    return tuple(spans)


def _classify_identifier(
    name: str,
    line: str,
    end: int,
    *,
    after_colon: bool,
    leading: bool,
) -> AthToken | None:
    if after_colon:
        # The ``throat`` in ``GridExport:throat``.
        return AthToken.LABEL
    if name.lower() in MATH_FUNCTIONS and line[end:].lstrip().startswith("("):
        return AthToken.FUNCTION
    if leading:
        # An assignment key, or a contour primitive like ``point``.
        return AthToken.KEY
    return None


def ath_syntax_colors(palette: QPalette) -> AthSyntaxColors:
    """Colour set for this palette's text background."""
    return ATH_LIGHT_COLORS if palette.color(QPalette.Base).lightness() >= 128 else ATH_DARK_COLORS


def ath_syntax_formats(palette: QPalette) -> dict[AthToken, QTextCharFormat]:
    colors = ath_syntax_colors(palette)
    formats: dict[AthToken, QTextCharFormat] = {}
    for token in AthToken:
        char_format = QTextCharFormat()
        char_format.setForeground(QColor(getattr(colors, token.value)))
        formats[token] = char_format
    formats[AthToken.COMMENT].setFontItalic(True)
    return formats


class AthSyntaxHighlighter(QSyntaxHighlighter):
    """Colours Ath source in a document; a no-op while disabled.

    Disabling re-runs it: ``setDocument(None)`` would leave the old formats
    on screen.
    """

    def __init__(self, document, palette: QPalette, *, enabled: bool = True):
        super().__init__(document)
        self._enabled = bool(enabled)
        self._formats = ath_syntax_formats(palette)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self.rehighlight()

    def refresh_colors(self, palette: QPalette) -> None:
        self._formats = ath_syntax_formats(palette)
        if self._enabled:
            self.rehighlight()

    def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt override
        if not self._enabled:
            return
        for span in tokenize_ath_line(text):
            self.setFormat(span.start, span.length, self._formats[span.token])


__all__ = [
    "ATH_DARK_COLORS",
    "ATH_LIGHT_COLORS",
    "MATH_FUNCTIONS",
    "AthSpan",
    "AthSyntaxColors",
    "AthSyntaxHighlighter",
    "AthToken",
    "ath_syntax_colors",
    "ath_syntax_formats",
    "tokenize_ath_line",
]
