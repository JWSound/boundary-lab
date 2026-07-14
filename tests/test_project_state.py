from blab.ui.project_state import (
    ProjectPreferencesState,
    default_scripts,
    new_project_document,
    replace_script,
    script_from_payload,
    script_to_payload,
    scripts_from_payload,
    unique_script_name,
)


def test_ath_script_payload_round_trip_preserves_identity_and_mesh_settings() -> None:
    script = replace_script(
        default_scripts("config text"),
        default_scripts("config text")[0].id,
    )
    script = script[0]
    payload = script_to_payload(script)
    restored = script_from_payload(payload)

    assert restored == script


def test_scripts_from_payload_falls_back_to_default_script() -> None:
    scripts = scripts_from_payload([], fallback_config_text="legacy")

    assert len(scripts) == 1
    assert scripts[0].config_text == "legacy"


def test_unique_script_name_avoids_existing_names() -> None:
    scripts = default_scripts("")

    assert unique_script_name("ath", scripts) == "ath_2"


def test_project_preferences_round_trip_only_contains_project_fields() -> None:
    preferences = ProjectPreferencesState(
        freq_min_hz=100,
        freq_max_hz=12000,
        normalized_channel_correction=False,
        spherical_sampling_enabled=True,
    )

    payload = preferences.to_payload()

    assert ProjectPreferencesState.from_payload(payload) == preferences
    assert "solve_backend" not in payload
    assert "gmres_tolerance" not in payload
    assert "use_burton_miller" not in payload


def test_new_project_document_owns_project_local_collections() -> None:
    project = new_project_document(project_preferences=ProjectPreferencesState())

    assert len(project.ath_scripts) == 1
    assert project.active_ath_script_id == project.ath_scripts[0].id
    assert project.source_config_by_name == {}
    assert project.channel_config_by_name == {}
