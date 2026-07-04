from blab.ui.project_state import (
    default_scripts,
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
    assert scripts[0].mesh_scale_factor == 0.001


def test_ath_script_payload_without_scale_uses_millimetre_mesh_scale() -> None:
    restored = script_from_payload({"id": "script1", "name": "ath", "config_text": ""})

    assert restored is not None
    assert restored.mesh_scale_factor == 0.001


def test_ath_script_payload_normalizes_legacy_metre_scale_to_millimetres() -> None:
    restored = script_from_payload(
        {
            "id": "script1",
            "name": "ath",
            "config_text": "",
            "mesh_scale_factor": 1.0,
        }
    )

    assert restored is not None
    assert restored.mesh_scale_factor == 0.001


def test_ath_script_payload_clamps_custom_scale_to_millimetres() -> None:
    restored = script_from_payload(
        {
            "id": "script1",
            "name": "ath",
            "config_text": "",
            "mesh_scale_factor": 1000.0,
        }
    )

    assert restored is not None
    assert restored.mesh_scale_factor == 0.001


def test_ath_script_payload_serializes_stale_scale_as_millimetres() -> None:
    script = script_from_payload(
        {
            "id": "script1",
            "name": "ath",
            "config_text": "",
            "mesh_scale_factor": 0.001,
        }
    )
    assert script is not None
    stale_script = replace_script((script,), script.id, mesh_scale_factor=1000.0)[0]

    payload = script_to_payload(stale_script)

    assert payload["mesh_scale_factor"] == 0.001


def test_unique_script_name_avoids_existing_names() -> None:
    scripts = default_scripts("")

    assert unique_script_name("ath", scripts) == "ath_2"
