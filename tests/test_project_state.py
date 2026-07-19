from blab.ui.project_state import (
    ProjectPreferencesState,
    default_generator_documents,
    generator_document_from_payload,
    generator_document_to_payload,
    generator_documents_from_payload,
    new_project_document,
    replace_generator_document,
    unique_generator_name,
)


def test_generator_document_payload_round_trip_preserves_identity_and_mesh_settings() -> None:
    documents = default_generator_documents("config text")
    document = replace_generator_document(documents, documents[0].id)[0]
    payload = generator_document_to_payload(document)
    restored = generator_document_from_payload(payload)

    assert restored == document
    assert restored.provider_id == "ath"
    assert restored.source == {"format": "ath_cfg", "text": "config text"}


def test_generator_documents_from_payload_falls_back_to_default_design() -> None:
    documents = generator_documents_from_payload([], fallback_config_text="legacy")

    assert len(documents) == 1
    assert documents[0].source["text"] == "legacy"


def test_unique_generator_name_avoids_existing_names() -> None:
    documents = default_generator_documents("")

    assert unique_generator_name("waveguide", documents) == "waveguide_2"


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

    assert len(project.generator_documents) == 1
    assert project.active_generator_document_id == project.generator_documents[0].id
    assert project.source_config_by_name == {}
    assert project.channel_config_by_name == {}
