from blab.config import ChannelConfig, CrossoverConfig, RadiatorConfig
from blab.ui.source_channel_config import (
    apply_saved_imported_source_config,
    channel_configs,
    channels_for_solver_radiators,
    load_source_config_by_name,
    save_channel_config,
    save_source_config,
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


def test_channels_for_solver_radiators_adds_missing_channel_names() -> None:
    channels = channels_for_solver_radiators(
        (ChannelConfig(name="LF"),),
        (
            RadiatorConfig(name="woofer", tag=1, channel="LF"),
            RadiatorConfig(name="tweeter", tag=2, channel="HF"),
        ),
    )

    assert channels == (ChannelConfig(name="LF"), ChannelConfig(name="HF"))
