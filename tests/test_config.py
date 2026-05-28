from pathlib import Path

import pytest

from blab.config import load_external_config, parse_translation_m


def test_loads_multimesh_toml_and_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        """
[[meshes]]
name = "waveguide"
file = "meshes/wg.msh"
scale_factor = 0.001
translation_m = [0.0, 0.1, -0.2]

[[radiators]]
name = "HF"
mesh = "waveguide"
tag = 4
level_db = -2.5
polarity = -1
delay_ms = 0.125

[radiators.hpf]
filter = "butterworth"
order = 2
frequency_hz = 800.0

[radiators.lpf]
filter = "linkwitz_riley"
order = 4
frequency_hz = 5000.0
""".strip(),
        encoding="utf-8",
    )

    meshes, radiators = load_external_config(config_path)

    assert len(meshes) == 1
    assert meshes[0].file == str(tmp_path / "meshes" / "wg.msh")
    assert meshes[0].translation_m == (0.0, 0.1, -0.2)

    assert len(radiators) == 1
    radiator = radiators[0]
    assert radiator.name == "HF"
    assert radiator.mesh == "waveguide"
    assert radiator.tag == 4
    assert radiator.polarity == -1
    assert radiator.hpf.type == "highpass"
    assert radiator.hpf.frequency_hz == 800.0
    assert radiator.lpf.type == "lowpass"
    assert radiator.lpf.frequency_hz == 5000.0


def test_translation_requires_three_values() -> None:
    with pytest.raises(ValueError, match="translation_m"):
        parse_translation_m("mesh", [0.0, 1.0])


def test_duplicate_radiator_names_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        """
[[radiators]]
name = "dup"
tag = 1

[[radiators]]
name = "dup"
tag = 2
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate radiator name"):
        load_external_config(config_path)


def test_external_config_rejects_json_files(tmp_path: Path) -> None:
    config_path = tmp_path / "case.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Use .toml"):
        load_external_config(config_path)
