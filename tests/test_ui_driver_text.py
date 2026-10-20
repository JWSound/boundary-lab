from types import SimpleNamespace

import pytest
from PySide6.QtCore import QLocale

from blab.driver_text import parse_driver_text
from blab.physical_model import ComponentKind
from blab.ui.component_editor import _ComponentDraft, _ComponentEditorDialog
from blab.ui.numeric_locale import parse_decimal_number
from test_driver_text import EXAMPLE


@pytest.fixture
def editor(qapp):
    widget = _ComponentEditorDialog(
        _ComponentDraft(
            "driver",
            "Existing component",
            ComponentKind.ELECTRODYNAMIC_TRANSDUCER,
            (),
            "main",
            {
                "lumped_sealed_rear_chamber": {
                    "enabled": True,
                    "volume_m3": 0.01,
                    "projected_area_m2": 0.05,
                }
            },
        ),
        boundaries=(),
        resources_by_id={},
        region_names={},
        channel_names=("main",),
        unavailable_boundary_ids=set(),
        symmetry_mode="off",
        mesh_cache={},
    )
    yield widget
    widget.close()
    widget.deleteLater()


@pytest.mark.parametrize("locale_name", ["en_US", "de_DE"])
def test_import_export_preserves_configuration(editor, tmp_path, locale_name):
    source = tmp_path / "source.txt"
    target = tmp_path / "saved"
    source.write_text(EXAMPLE, encoding="utf-8")
    editor.file_dialogs = SimpleNamespace(open_file=lambda *a: source, save_file=lambda *a: target)
    for edit in editor.parameter_edits.values():
        edit.setLocale(QLocale(locale_name))
    before = (
        editor.name_edit.text(),
        editor.channel_combo.currentData(),
        editor.axis_mode_combo.currentData(),
        editor.selected_boundary_ids(),
        editor.rear_chamber_volume_spin.value(),
    )
    editor._semi_inductance_parameters = {"enabled": True, "le_h": 0.5}
    editor.import_driver_button.click()
    assert editor._semi_inductance_parameters is None
    for key, expected in (("le_h", 1.3), ("mmd_kg", 62.4), ("cms_m_per_n", 136.0)):
        edit = editor.parameter_edits[key]
        assert parse_decimal_number(edit.text(), edit.locale()) == pytest.approx(expected)
    assert editor.parameter_edits["le_h"].isEnabled()
    assert before == (
        editor.name_edit.text(),
        editor.channel_combo.currentData(),
        editor.axis_mode_combo.currentData(),
        editor.selected_boundary_ids(),
        editor.rear_chamber_volume_spin.value(),
    )
    editor.export_driver_button.click()
    driver = parse_driver_text(target.with_suffix(".txt").read_text())
    assert driver.name == "Example Driver"
    assert driver.sd_m2 == pytest.approx(0.0522)
    assert driver.parameters["le_h"] == pytest.approx(0.0013)


def test_bad_import_leaves_fields_unchanged(editor, tmp_path, monkeypatch):
    source = tmp_path / "bad.txt"
    source.write_text(EXAMPLE.replace("Mmd=62.4", "Mmd=nan"))
    editor.file_dialogs = SimpleNamespace(open_file=lambda *a: source)
    errors = []
    monkeypatch.setattr("blab.ui.component_editor.QMessageBox.warning", lambda *args: errors.append(args[-1]))
    editor.parameter_edits["re_ohm"].setText("12")
    editor.import_driver_button.click()
    assert errors and "finite" in errors[0]
    assert editor.parameter_edits["re_ohm"].text() == "12"
    assert not editor._driver_reference


def test_cancelled_import_leaves_fields_unchanged(editor):
    editor.file_dialogs = SimpleNamespace(open_file=lambda *a: None)
    editor.parameter_edits["re_ohm"].setText("12")
    editor.import_driver_button.click()
    assert editor.parameter_edits["re_ohm"].text() == "12"


def test_enabled_semi_inductance_import_and_export(editor, tmp_path):
    source = tmp_path / "semi.txt"
    target = tmp_path / "export.txt"
    text = EXAMPLE.replace("Re'=0.00", "Re'=6.2").replace("Leb=0.00", "Leb=0.1")
    text = text.replace("Le=0.00", "Le=2.5").replace("Ke=0.00", "Ke=0.04")
    text = text.replace("Rss=0.00", "Rss=1000").replace("SemiInductance=0", "SemiInductance=1")
    source.write_text(text)
    editor.file_dialogs = SimpleNamespace(open_file=lambda *a: source, save_file=lambda *a: target)
    editor.import_driver_button.click()
    assert not editor.parameter_edits["le_h"].isEnabled()
    assert editor._semi_inductance_parameters["le_h"] == pytest.approx(0.0025)
    editor.export_driver_button.click()
    restored = parse_driver_text(target.read_text())
    assert restored.parameters["le_h"] == pytest.approx(0.0013)
    assert restored.parameters["semi_inductance"] == editor._semi_inductance_parameters
