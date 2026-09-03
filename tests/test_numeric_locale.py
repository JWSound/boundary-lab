from __future__ import annotations

import pytest
from PySide6.QtCore import QLocale

from blab.ui.numeric_locale import format_decimal_number, parse_decimal_number


@pytest.mark.parametrize(
    ("locale_name", "text", "expected"),
    (
        ("de_DE", "6,2", 6.2),
        ("de_DE", "6.2", 6.2),
        ("en_US", "6.2", 6.2),
        ("en_US", "6,2", 6.2),
        ("de_DE", "1,2e-3", 0.0012),
    ),
)
def test_decimal_parser_accepts_locale_and_period_decimals(locale_name: str, text: str, expected: float) -> None:
    assert parse_decimal_number(text, QLocale(locale_name)) == pytest.approx(expected)


def test_decimal_parser_never_treats_period_as_a_group_separator() -> None:
    assert parse_decimal_number("0.015", QLocale("de_DE")) == pytest.approx(0.015)
    with pytest.raises(ValueError):
        parse_decimal_number("1.234,5", QLocale("de_DE"))


def test_decimal_formatter_uses_the_requested_locale_without_grouping() -> None:
    assert format_decimal_number(1234.5, QLocale("de_DE")) == "1234,5"
    assert format_decimal_number(1234.5, QLocale("en_US")) == "1234.5"
