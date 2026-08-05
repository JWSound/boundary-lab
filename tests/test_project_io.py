import json
from pathlib import Path

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
        generator_documents=[
            {
                "id": "design1",
                "name": "waveguide",
                "provider_id": "ath",
                "provider_schema_version": 1,
                "source": {"format": "ath_cfg", "text": "Ath config text"},
                "artifact": None,
            }
        ],
        active_generator_document_id="design1",
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
        symmetry="xy",
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
        component_channel_by_id={"component:woofer": "tweeter"},
    )

    project_path = write_project_file(tmp_path / "test_project", payload)
    loaded = read_project_file(project_path)

    assert project_path.name == "test_project.blab.json"
    assert loaded == payload
    assert loaded["symmetry"] == "xy"
    assert loaded["component_channel_by_id"] == {"component:woofer": "tweeter"}
    assert json.loads(project_path.read_text(encoding="utf-8"))["schema_version"] == PROJECT_SCHEMA_VERSION


def test_pending_legacy_source_configuration_can_coexist_with_partial_physical_system() -> None:
    payload = build_project_payload(
        generator_documents=[],
        active_generator_document_id=None,
        imported_meshes=[],
        source_config_by_name={"speaker:driver": {"driven": True}},
        physical_system={"model_version": 1},
    )

    assert payload["source_config_by_name"] == {"speaker:driver": {"driven": True}}


def test_project_file_rejects_unknown_schema(tmp_path) -> None:
    project_path = tmp_path / "future_project.json"
    project_path.write_text('{"schema_version": 999}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported project schema"):
        read_project_file(project_path)


def test_project_file_rejects_unversioned_payload(tmp_path) -> None:
    project_path = tmp_path / "unversioned_project.json"
    project_path.write_text('{"ath_config_text": "legacy"}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing schema_version"):
        read_project_file(project_path)


def test_project_file_resolves_relative_paths(tmp_path) -> None:
    project_dir = tmp_path / "sample"
    project_dir.mkdir()
    project_path = project_dir / "sample.blab.json"
    project_path.write_text(
        json.dumps(
            {
                "schema_version": PROJECT_SCHEMA_VERSION,
                "imported_meshes": [
                    {
                        "name": "cabinet",
                        "source_file": "meshes/cabinet.msh",
                        "cleaned_file": None,
                    }
                ],
                "generator_documents": [
                    {
                        "id": "abc",
                        "name": "waveguide",
                        "provider_id": "ath",
                        "provider_schema_version": 1,
                        "source": {"format": "ath_cfg", "text": ""},
                        "artifact": {
                            "output_dir": "runs/case",
                            "mesh_path": "runs/case/case.msh",
                            "cleaned_mesh_path": "",
                            "source_path": "runs/case/config.txt",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = read_project_file(project_path)

    assert loaded["imported_meshes"][0]["source_file"] == str((project_dir / "meshes/cabinet.msh").resolve())
    assert loaded["imported_meshes"][0]["cleaned_file"] is None
    artifact = loaded["generator_documents"][0]["artifact"]
    assert artifact["output_dir"] == str((project_dir / "runs/case").resolve())
    assert artifact["mesh_path"] == str((project_dir / "runs/case/case.msh").resolve())
    assert artifact["cleaned_mesh_path"] == ""
    assert artifact["source_path"] == str((project_dir / "runs/case/config.txt").resolve())


def test_project_migration_rejects_non_integer_schema() -> None:
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        migrate_project_payload({"schema_version": "future"})


def test_schema_v1_migrates_without_application_preference_override() -> None:
    migrated = migrate_project_payload({"schema_version": 1, "ath_config_text": "legacy"})

    assert migrated["schema_version"] == PROJECT_SCHEMA_VERSION
    assert migrated["generator_documents"][0]["provider_id"] == "ath"
    assert migrated["generator_documents"][0]["source"] == {"format": "ath_cfg", "text": "legacy"}
    assert "project_preferences" not in migrated


def test_schema_v2_migrates_every_ath_script_to_generator_document() -> None:
    migrated = migrate_project_payload(
        {
            "schema_version": 2,
            "active_ath_script_id": "second",
            "ath_scripts": [
                {"id": "first", "name": "hf", "config_text": "first config"},
                {"id": "second", "name": "lf", "config_text": "second config"},
            ],
        }
    )

    assert migrated["active_generator_document_id"] == "second"
    assert [document["provider_id"] for document in migrated["generator_documents"]] == ["ath", "ath"]
    assert [document["source"]["text"] for document in migrated["generator_documents"]] == [
        "first config",
        "second config",
    ]


def test_schema_v3_migrates_without_inventing_a_physical_system() -> None:
    migrated = migrate_project_payload(
        {
            "schema_version": 3,
            "generator_documents": [],
            "imported_meshes": [],
        }
    )

    assert migrated["schema_version"] == PROJECT_SCHEMA_VERSION
    assert "physical_system" not in migrated


def test_legacy_stitch_setting_migrates_to_exterior_region_name() -> None:
    migrated = migrate_project_payload(
        {
            "schema_version": 5,
            "stitch_imported_meshes": True,
        }
    )

    assert migrated["stitch_exterior_meshes"] is True
    assert "stitch_imported_meshes" not in migrated


def test_project_preferences_are_optional_and_drop_solver_settings() -> None:
    payload = build_project_payload(
        generator_documents=[],
        active_generator_document_id=None,
        imported_meshes=[],
        source_config_by_name={},
        project_preferences={
            "freq_min_hz": 80,
            "solve_backend": "bempp",
            "gmres_tolerance": 1e-8,
            "use_burton_miller": False,
            "fem_bulk_loss_factor": 0.02,
        },
    )

    saved_preferences = payload["project_preferences"]
    assert saved_preferences["freq_min_hz"] == 80
    assert "solve_backend" not in saved_preferences
    assert "gmres_tolerance" not in saved_preferences
    assert "use_burton_miller" not in saved_preferences
    assert saved_preferences["fem_bulk_loss_factor"] == 0.02


def test_shipped_examples_do_not_request_preference_overrides() -> None:
    example_paths = sorted(Path("examples").glob("**/*.blab.json"))

    assert example_paths
    for example_path in example_paths:
        payload = json.loads(example_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == PROJECT_SCHEMA_VERSION
        assert "project_preferences" not in payload
