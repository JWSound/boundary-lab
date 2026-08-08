"""Ath tokenizing, the line-number gutter, and the colouring toggle."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QTabBar

from blab.generators.ath import ath_source_text
from blab.ui.ath_editor import GUTTER_MIN_DIGITS, AthScriptEditor
from blab.ui.ath_syntax import (
    ATH_DARK_COLORS,
    ATH_LIGHT_COLORS,
    AthToken,
    ath_syntax_colors,
    tokenize_ath_line,
)
from blab.ui.main_window_widgets import TabCloseButton, dimmed_button_text_color
from blab.ui.settings import EDITOR_SYNTAX_HIGHLIGHTING_KEY, load_syntax_highlighting_enabled

EXAMPLE_SOURCE = "\n".join(
    (
        "Throat.Profile = 1 ; 1 = OS-SE waveguide",
        "Throat.Diameter = 25.4",
        "",
        "GridExport:throat = {",
        ' Delimiter = ";"',
        "}",
    )
)


def _tokens(line: str) -> list[tuple[AthToken, str]]:
    return [(span.token, line[span.start : span.start + span.length]) for span in tokenize_ath_line(line)]


def test_assignment_splits_into_key_operator_value_and_comment() -> None:
    assert _tokens("Throat.Diameter = 25.4 ; [mm]") == [
        (AthToken.KEY, "Throat.Diameter"),
        (AthToken.OPERATOR, "="),
        (AthToken.NUMBER, "25.4"),
        (AthToken.COMMENT, "; [mm]"),
    ]


def test_a_semicolon_inside_a_string_is_not_a_comment() -> None:
    """Separate comment and string rules cannot both handle this line."""
    assert _tokens('Delimiter = ";"') == [
        (AthToken.KEY, "Delimiter"),
        (AthToken.OPERATOR, "="),
        (AthToken.STRING, '";"'),
    ]


def test_an_unterminated_string_stops_at_the_line_end() -> None:
    assert _tokens('FileExtension = "csv') == [
        (AthToken.KEY, "FileExtension"),
        (AthToken.OPERATOR, "="),
        (AthToken.STRING, '"csv'),
    ]


def test_block_label_after_a_colon_is_not_a_second_key() -> None:
    assert _tokens("GridExport:throat = {") == [
        (AthToken.KEY, "GridExport"),
        (AthToken.OPERATOR, ":"),
        (AthToken.LABEL, "throat"),
        (AthToken.OPERATOR, "="),
        (AthToken.OPERATOR, "{"),
    ]


def test_expression_functions_are_marked_only_when_called() -> None:
    spanned = {text: token for token, text in _tokens("Coverage.Angle = 36 + 17*cos(p)^2")}
    assert spanned["cos"] == AthToken.FUNCTION
    # ``p`` is a bare variable and stays plain.
    assert "p" not in spanned

    assert _tokens("cos = 1") == [(AthToken.KEY, "cos"), (AthToken.OPERATOR, "="), (AthToken.NUMBER, "1")]


def test_contour_primitives_name_their_line_and_leave_point_names_plain() -> None:
    assert _tokens("        point p1 6.8 0 1.5") == [
        (AthToken.KEY, "point"),
        (AthToken.NUMBER, "6.8"),
        (AthToken.NUMBER, "0"),
        (AthToken.NUMBER, "1.5"),
    ]


def test_leading_dot_and_exponent_numbers_are_one_token_each() -> None:
    assert _tokens("GCurve.Dist = .5") == [
        (AthToken.KEY, "GCurve.Dist"),
        (AthToken.OPERATOR, "="),
        (AthToken.NUMBER, ".5"),
    ]
    assert (AthToken.NUMBER, "1e-3") in _tokens("Tolerance = 1e-3")


def test_a_whole_line_comment_swallows_everything_after_the_semicolon() -> None:
    assert _tokens(";Throat.Angle = 20") == [(AthToken.COMMENT, ";Throat.Angle = 20")]


def test_colour_set_follows_the_text_background_not_the_window() -> None:
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor("#FFFFFF"))
    assert ath_syntax_colors(palette) is ATH_LIGHT_COLORS

    palette.setColor(QPalette.Base, QColor("#262626"))
    assert ath_syntax_colors(palette) is ATH_DARK_COLORS


def _format_colors(editor: AthScriptEditor) -> list[str]:
    """Foreground colour of every applied span, in document order.

    Read inside the loop: a FormatRange's format dies with its layout.
    """
    colors = []
    block = editor.document().firstBlock()
    while block.isValid():
        colors.extend(entry.format.foreground().color().name() for entry in block.layout().formats())
        block = block.next()
    return colors


def test_editor_colours_source_by_default_and_clears_it_when_switched_off(qapp) -> None:
    editor = AthScriptEditor()
    editor.setPlainText(EXAMPLE_SOURCE)

    assert editor.syntax_highlighting_enabled
    assert _format_colors(editor)

    editor.set_syntax_highlighting_enabled(False)

    # Detaching would leave these on screen; an empty pass clears them.
    assert not editor.syntax_highlighting_enabled
    assert _format_colors(editor) == []

    editor.set_syntax_highlighting_enabled(True)
    assert _format_colors(editor)


def test_editor_starts_uncoloured_when_asked(qapp) -> None:
    editor = AthScriptEditor(highlight_syntax=False)
    editor.setPlainText(EXAMPLE_SOURCE)

    assert not editor.syntax_highlighting_enabled
    assert _format_colors(editor) == []


def test_gutter_reserves_the_text_margin_and_widens_with_the_line_count(qapp) -> None:
    editor = AthScriptEditor()
    editor.setPlainText("Length = 80\n")

    narrow = editor.line_number_width()
    assert narrow > 0
    assert editor.viewportMargins().left() == narrow
    assert editor.line_number_gutter.sizeHint().width() == narrow

    editor.setPlainText("\n".join(f"Line{index} = {index}" for index in range(120)))

    assert editor.line_number_width() > narrow
    assert editor.viewportMargins().left() == editor.line_number_width()


def test_gutter_width_does_not_jitter_across_the_first_ten_lines(qapp) -> None:
    """Short designs would otherwise shift sideways on reaching line 10."""
    editor = AthScriptEditor()
    editor.setPlainText("Length = 80")
    single_digit = editor.line_number_width()

    editor.setPlainText("\n".join(["Length = 80"] * 10))

    assert editor.line_number_width() == single_digit
    assert GUTTER_MIN_DIGITS == 2


def test_theme_change_recolours_an_open_editor(qapp) -> None:
    from blab.ui.theme import apply_application_theme

    editor = AthScriptEditor()
    editor.setPlainText("Throat.Diameter = 25.4")
    try:
        apply_application_theme("light")
        light = _format_colors(editor)[0]

        apply_application_theme("dark")
        dark = _format_colors(editor)[0]

        assert light == ATH_LIGHT_COLORS.key.lower()
        assert dark == ATH_DARK_COLORS.key.lower()
    finally:
        apply_application_theme("system")


def test_toggle_action_is_on_the_design_dock_and_reaches_every_ath_tab(main_window) -> None:
    action = main_window.syntax_highlighting_action
    editors = [main_window.editor_tabs.widget(index) for index in range(len(main_window.generator_documents))]

    assert editors, "a fresh project should open with one design tab"
    assert action.isCheckable()
    assert action.isChecked()
    assert action in [button.defaultAction() for button in main_window.editor_dock.titleBarWidget().tool_buttons]
    assert all(editor.syntax_highlighting_enabled for editor in editors)

    action.setChecked(False)

    assert not main_window.syntax_highlighting_enabled
    assert not any(editor.syntax_highlighting_enabled for editor in editors)


def test_the_toggle_survives_a_rebuild_of_the_design_tabs(main_window) -> None:
    main_window.set_syntax_highlighting_enabled(False)

    main_window.rebuild_generator_document_tabs()

    editor = main_window.editor_tabs.widget(0)
    assert not editor.syntax_highlighting_enabled
    assert editor.toPlainText() == ath_source_text(main_window.generator_documents[0])


def _tab_close_buttons(main_window) -> list:
    tab_bar = main_window.editor_tabs.tabBar()
    return [tab_bar.tabButton(index, QTabBar.ButtonPosition.RightSide) for index in range(tab_bar.count())]


def test_design_tabs_carry_the_themed_close_button_and_the_add_tab_carries_none(main_window) -> None:
    """Qt's own close button is a fixed red-orange bitmap."""
    buttons = _tab_close_buttons(main_window)

    assert len(buttons) == len(main_window.generator_documents) + 1
    assert all(isinstance(button, TabCloseButton) for button in buttons[:-1])
    assert buttons[-1] is None


def test_close_button_glyph_is_dimmer_than_tab_text_in_both_themes(qapp) -> None:
    from blab.ui.theme import apply_application_theme

    try:
        for theme in ("light", "dark"):
            apply_application_theme(theme)
            palette = qapp.palette()
            glyph = dimmed_button_text_color(palette)
            text = palette.color(QPalette.ButtonText)
            face = palette.color(QPalette.Button)

            # Not exactly neutral in the light theme; its face is a warm grey.
            channels = (glyph.red(), glyph.green(), glyph.blue())
            assert max(channels) - min(channels) <= 8, f"{theme}: {glyph.name()} is not grey"

            low, high = sorted((text.lightness(), face.lightness()))
            assert low < glyph.lightness() < high, f"{theme}: {glyph.name()} is not dimmed"
    finally:
        apply_application_theme("system")


def test_a_close_button_removes_its_own_design_after_the_tabs_renumber(main_window) -> None:
    """The index is resolved on click, after any renumbering."""
    main_window.add_generator_document()
    main_window.add_generator_document()
    names = [document.name for document in main_window.generator_documents]
    assert len(names) == 3

    _tab_close_buttons(main_window)[1].click()

    assert [document.name for document in main_window.generator_documents] == [names[0], names[2]]

    _tab_close_buttons(main_window)[1].click()
    assert [document.name for document in main_window.generator_documents] == [names[0]]


def test_recolouring_never_marks_the_project_unsaved(main_window) -> None:
    """A re-highlight must not count as an edit.

    Otherwise the toggle or a theme switch raises the unsaved-changes prompt.
    """
    from blab.ui.theme import apply_application_theme

    assert not main_window._has_unsaved_project_changes()

    main_window.set_syntax_highlighting_enabled(False)
    main_window.set_syntax_highlighting_enabled(True)
    assert not main_window._has_unsaved_project_changes()

    try:
        apply_application_theme("dark")
        assert not main_window._has_unsaved_project_changes()
    finally:
        apply_application_theme("system")


def test_the_toggle_is_remembered_and_defaults_to_on(main_window) -> None:
    settings = main_window.settings
    try:
        main_window.set_syntax_highlighting_enabled(False)
        assert load_syntax_highlighting_enabled(settings) is False

        main_window.set_syntax_highlighting_enabled(True)
        assert load_syntax_highlighting_enabled(settings) is True
    finally:
        settings.remove(EDITOR_SYNTAX_HIGHLIGHTING_KEY)

    assert load_syntax_highlighting_enabled(settings) is True
