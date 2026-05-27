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
        stitch_imported_meshes=True,
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
        "stitch_imported_meshes": False,
        "source_config_by_name": {},
        "channel_config_by_name": {},
    }


def test_project_file_resolves_relative_paths(tmp_path) -> None:
    project_dir = tmp_path / "sample"
    project_dir.mkdir()
    project_path = project_dir / "sample.blab.json"
    project_path.write_text(
        json.dumps(
            {
                "schema_version": PROJECT_SCHEMA_VERSION,
                "ath_mesh": {
                    "source_file": "ath/source.msh",
                    "cleaned_file": "ath/cleaned.msh",
                },
                "imported_meshes": [
                    {
                        "name": "cabinet",
                        "source_file": "meshes/cabinet.msh",
                        "cleaned_file": None,
                    }
                ],
                "ath_scripts": [
                    {
                        "id": "abc",
                        "name": "ath",
                        "output_dir": "runs/case",
                        "stl_path": "runs/case/case.stl",
                        "msh_path": "runs/case/case.msh",
                        "cleaned_msh_path": "",
                        "config_path": "runs/case/config.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = read_project_file(project_path)

    assert loaded["ath_mesh"]["source_file"] == str((project_dir / "ath/source.msh").resolve())
    assert loaded["ath_mesh"]["cleaned_file"] == str((project_dir / "ath/cleaned.msh").resolve())
    assert loaded["imported_meshes"][0]["source_file"] == str((project_dir / "meshes/cabinet.msh").resolve())
    assert loaded["imported_meshes"][0]["cleaned_file"] is None
    assert loaded["ath_scripts"][0]["output_dir"] == str((project_dir / "runs/case").resolve())
    assert loaded["ath_scripts"][0]["stl_path"] == str((project_dir / "runs/case/case.stl").resolve())
    assert loaded["ath_scripts"][0]["msh_path"] == str((project_dir / "runs/case/case.msh").resolve())
    assert loaded["ath_scripts"][0]["cleaned_msh_path"] == ""
    assert loaded["ath_scripts"][0]["config_path"] == str((project_dir / "runs/case/config.txt").resolve())


def test_project_migration_rejects_non_integer_schema() -> None:
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        migrate_project_payload({"schema_version": "future"})
