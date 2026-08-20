from pathlib import Path

import meshio
import numpy as np

from blab.config import ChannelConfig, CrossoverConfig, RadiatorConfig
from blab.generators.base import GeneratedGeometry
from blab.ui.source_channel_config import (
    apply_saved_imported_source_config,
    apply_saved_source_config_to_result,
    channel_config_payload,
    channel_configs,
    channel_configs_from_payload,
    channels_for_solver_radiators,
    load_source_config_by_name,
    save_channel_config,
    save_source_config,
    source_config_payload,
)


class _Settings:
    def __init__(self):
        self.values = {}
        self.sync_count = 0

    def value(self, key: str, default=None):
        return self.values.get(key, default)

    def setValue(self, key: str, value) -> None:  # noqa: N802 - Qt-style test double
        self.values[key] = value

    def sync(self) -> None:
        self.sync_count += 1


def test_channel_config_round_trips_crossover_settings() -> None:
    settings = _Settings()
    save_channel_config(
        settings,
        (
            ChannelConfig(
                name="HF",
                level_db=-3.0,
                polarity=-1,
                delay_ms=0.25,
                hpf=CrossoverConfig(type="highpass", filter="butterworth", order=2, frequency_hz=800.0),
            ),
        ),
    )

    (channel,) = channel_configs(settings)

    assert channel.name == "HF"
    assert channel.level_db == -3.0
    assert channel.polarity == -1
    assert channel.delay_ms == 0.25
    assert channel.hpf.type == "highpass"
    assert channel.hpf.order == 2
    assert channel.hpf.frequency_hz == 800.0
    assert channel.lpf.type == "none"


def test_save_source_config_preserves_driven_surface_assignments() -> None:
    settings = _Settings()
    surface_tags = {
        "cabinet:woofer": ("cabinet", 7),
        "cabinet:port": ("cabinet", 8),
    }
    radiators = (
        RadiatorConfig(
            name="cabinet:woofer",
            mesh="cabinet",
            tag=7,
            channel="LF",
            velocity_offset_db=-1.5,
        ),
    )

    save_source_config(settings, surface_tags, radiators)
    saved = load_source_config_by_name(settings)

    assert saved["cabinet:woofer"] == {
        "driven": True,
        "channel": "LF",
        "velocity_offset_db": -1.5,
    }
    assert saved["cabinet:port"] == {
        "driven": False,
        "channel": "main",
        "velocity_offset_db": 0.0,
    }


def test_apply_saved_imported_source_config_ignores_generated_meshes() -> None:
    radiators = apply_saved_imported_source_config(
        surface_tags={
            "ath:driver": ("ath", 1),
            "cabinet:woofer": ("cabinet", 7),
        },
        generated_mesh_names={"ath"},
        existing_radiators=(),
        config_by_name={
            "ath:driver": {"driven": True, "channel": "main"},
            "cabinet:woofer": {"driven": True, "channel": "LF", "velocity_offset_db": 2.0},
        },
    )

    assert radiators == (
        RadiatorConfig(
            name="cabinet:woofer",
            mesh="cabinet",
            tag=7,
            channel="LF",
            velocity_offset_db=2.0,
        ),
    )


def test_saved_surface_settings_preserve_generated_ath_drive_group(tmp_path: Path) -> None:
    mesh_path = tmp_path / "ath.msh"
    meshio.write(
        mesh_path,
        meshio.Mesh(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
            cell_data={"gmsh:physical": [np.array([2], dtype=np.int32)]},
            field_data={"SD1D1001": np.array([2, 2], dtype=np.int32)},
        ),
        file_format="gmsh22",
        binary=False,
    )
    result = GeneratedGeometry(
        provider_id="ath",
        output_dir=tmp_path,
        mesh_path=mesh_path,
        radiators=(
            RadiatorConfig(
                name="SD1D1001",
                tag=2,
                drive_group="ath:0",
                drive_group_name="horn_driver",
                velocity_offset_db=-12.042,
            ),
        ),
    )

    updated = apply_saved_source_config_to_result(
        result,
        "2way",
        {
            "2way:SD1D1001": {
                "driven": True,
                "channel": "High",
                "velocity_offset_db": -12.0,
            }
        },
    )

    assert updated is not None
    assert updated.radiators[0].name == "2way:SD1D1001"
    assert updated.radiators[0].channel == "High"
    assert updated.radiators[0].velocity_offset_db == -12.0
    assert updated.radiators[0].drive_group == "ath:0"
    assert updated.radiators[0].drive_group_name == "horn_driver"


def test_imported_source_assignment_follows_surface_name_when_tag_changes() -> None:
    radiators = apply_saved_imported_source_config(
        surface_tags={"cabinet:woofer": ("cabinet", 17)},
        generated_mesh_names=set(),
        existing_radiators=(
            RadiatorConfig(
                name="cabinet:woofer",
                mesh="cabinet",
                tag=7,
                channel="LF",
                velocity_offset_db=-1.5,
            ),
        ),
        config_by_name={},
    )

    assert radiators == (
        RadiatorConfig(
            name="cabinet:woofer",
            mesh="cabinet",
            tag=17,
            channel="LF",
            velocity_offset_db=-1.5,
        ),
    )


def test_channels_for_solver_radiators_adds_missing_channel_names() -> None:
    channels = channels_for_solver_radiators(
        (ChannelConfig(name="LF"),),
        (
            RadiatorConfig(name="woofer", tag=1, channel="LF"),
            RadiatorConfig(name="tweeter", tag=2, channel="HF"),
        ),
    )

    assert channels == (ChannelConfig(name="LF"), ChannelConfig(name="HF"))


def test_pure_payload_helpers_do_not_require_qsettings() -> None:
    channels = (ChannelConfig(name="main", level_db=-2.0),)
    source_payload = source_config_payload(
        {"woofer": ("cabinet", 7)},
        (RadiatorConfig(name="woofer", mesh="cabinet", tag=7),),
    )
    channel_payload = channel_config_payload(channels)

    assert source_payload["woofer"]["driven"] is True
    assert channel_configs_from_payload(channel_payload) == channels
