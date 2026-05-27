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


def test_unique_script_name_avoids_existing_names() -> None:
    scripts = default_scripts("")

    assert unique_script_name("ath", scripts) == "ath_2"
