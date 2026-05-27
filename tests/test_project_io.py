import json

import pytest

from blab.ui.project_io import (
    PROJECT_SCHEMA_VERSION,
    build_project_payload,
    migrate_project_payload,
    normalize_project_path,
    read_project_file,
    write_project_file,
)


def test_project_path_gets_default_suffix() -> None:
    assert normalize_project_path("speaker_project").name == "speaker_project.blab.json"
    assert normalize_project_path("speaker_project.json").name == "speaker_project.json"


def test_project_file_round_trip(tmp_path) -> None:
    payload = build_project_payload(
        ath_config_text="Ath config text",
        ath_mesh={
            "name": "ath",
            "source_file": "C:/meshes/ath.msh",
            "cleaned_file": None,
            "translation_mm": [1, 2, 3],
            "enabled": True,
        },
        imported_meshes=[
            {
                "name": "enclosure",
                "source_file": "C:/meshes/enclosure.msh",
                "cleaned_file": None,
                "translation_mm": [0, 10, 0],
                "enabled": True,
            }
        ],
        source_config_by_name={
            "ath:SD1D1001": {
                "driven": True,
                "channel": "tweeter",
                "velocity_offset_db": -3.0,
            }
        },
        channel_config_by_name={
            "tweeter": {
                "level_db": -1.0,
                "polarity": 1,
                "delay_ms": 0.25,
                "hpf": {},
                "lpf": {},
            }
        },
    )

    project_path = write_project_file(tmp_path / "test_project", payload)
    loaded = read_project_file(project_path)

    assert project_path.name == "test_project.blab.json"
    assert loaded == payload
    assert json.loads(project_path.read_text(encoding="utf-8"))["schema_version"] == PROJECT_SCHEMA_VERSION


def test_project_file_rejects_unknown_schema(tmp_path) -> None:
    project_path = tmp_path / "future_project.json"
    project_path.write_text('{"schema_version": 999}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported project schema"):
        read_project_file(project_path)


def test_project_file_migrates_legacy_unversioned_payload(tmp_path) -> None:
    project_path = tmp_path / "legacy_project.json"
    project_path.write_text('{"ath_config_text": "legacy"}', encoding="utf-8")

    loaded = read_project_file(project_path)

    assert loaded == {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "ath_config_text": "legacy",
        "ath_scripts": [],
        "active_ath_script_id": None,
        "ath_mesh": {},
        "imported_meshes": [],
        "source_config_by_name": {},
        "channel_config_by_name": {},
    }


def test_project_migration_rejects_non_integer_schema() -> None:
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        migrate_project_payload({"schema_version": "future"})
