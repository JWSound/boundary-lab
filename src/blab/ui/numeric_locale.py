"""Locale-aware numeric text helpers for Qt input controls."""

from __future__ import annotations

import math
import re

from PySide6.QtCore import QLocale
from PySide6.QtGui import QValidator

_NUMBER_PATTERN = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?")
_PARTIAL_NUMBER_PATTERN = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d*))?(?:[eE][+-]?\d*)?")


def _normalized_decimal_text(text: str, locale: QLocale | None = None) -> str:
    """Normalize a locale decimal mark, comma, or period without accepting grouping."""

    stripped = text.strip()
    decimal_marks = {".", ","}
    if locale is not None:
        decimal_marks.add(str(locale.decimalPoint()))
    used_marks = {mark for mark in decimal_marks if mark and mark in stripped}
    if len(used_marks) > 1:
        raise ValueError("mixed decimal separators")
    if used_marks:
        mark = used_marks.pop()
        if stripped.count(mark) > 1:
            raise ValueError("multiple decimal separators")
        stripped = stripped.replace(mark, ".")
    return stripped


def parse_decimal_number(text: str, locale: QLocale | None = None) -> float:
    """Parse a number using the locale decimal mark while also accepting a period.

    Thousands separators are deliberately unsupported: engineering inputs such as
    ``0.015`` must never become ``15`` under a comma-decimal locale.
    """

    normalized = _normalized_decimal_text(text, locale)
    if _NUMBER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{text!r} is not a decimal number")
    value = float(normalized)
    if not math.isfinite(value):
        raise ValueError(f"{text!r} is not a finite decimal number")
    return value


def format_decimal_number(value: float, locale: QLocale, *, precision: int = 12) -> str:
    """Format a number without grouping, using the locale's decimal mark."""

    text = f"{float(value):.{precision}g}"
    decimal_point = str(locale.decimalPoint())
    return text if decimal_point == "." else text.replace(".", decimal_point, 1)


class FlexibleDoubleValidator(QValidator):
    """Validate scientific decimal input with either comma or period decimals."""

    def validate(self, input_text: str, position: int):  # noqa: N802 - Qt override
        try:
            normalized = _normalized_decimal_text(input_text, self.locale())
        except ValueError:
            return QValidator.State.Invalid, input_text, position
        if _NUMBER_PATTERN.fullmatch(normalized) is not None:
            try:
                parse_decimal_number(input_text, self.locale())
            except ValueError:
                return QValidator.State.Invalid, input_text, position
            return QValidator.State.Acceptable, input_text, position
        if _PARTIAL_NUMBER_PATTERN.fullmatch(normalized) is not None:
            return QValidator.State.Intermediate, input_text, position
        return QValidator.State.Invalid, input_text, position


__all__ = ["FlexibleDoubleValidator", "format_decimal_number", "parse_decimal_number"]
