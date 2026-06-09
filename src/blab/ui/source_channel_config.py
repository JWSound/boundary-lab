"""Source and channel configuration persistence helpers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSettings

from blab.ath import AthRunResult, read_surface_physical_names
from blab.config import ChannelConfig, CrossoverConfig, RadiatorConfig


SOURCE_CONFIG_SETTINGS_KEY = "source/config_by_name"
CHANNEL_CONFIG_SETTINGS_KEY = "channel/config_by_name"


def clear_source_channel_configs(settings: QSettings) -> None:
    settings.setValue(SOURCE_CONFIG_SETTINGS_KEY, "{}")
    settings.setValue(CHANNEL_CONFIG_SETTINGS_KEY, "{}")
    settings.sync()


def load_source_config_by_name(settings: QSettings) -> dict[str, dict]:
    return _load_config_by_name(settings, SOURCE_CONFIG_SETTINGS_KEY)


def save_source_config(
    settings: QSettings,
    surface_tags: dict[str, tuple[str, int]],
    radiators: tuple[RadiatorConfig, ...],
) -> None:
    radiators_by_name = {radiator.name: radiator for radiator in radiators}
    config_by_name = load_source_config_by_name(settings)
    for surface_name in surface_tags:
        radiator = radiators_by_name.get(surface_name)
        config_by_name[surface_name] = {
            "driven": radiator is not None,
            "channel": "main" if radiator is None else radiator.channel,
            "velocity_offset_db": 0.0 if radiator is None else float(radiator.velocity_offset_db),
        }
    save_source_config_by_name(settings, config_by_name)


def save_source_config_by_name(settings: QSettings, config_by_name: dict) -> None:
    _save_config_by_name(settings, SOURCE_CONFIG_SETTINGS_KEY, config_by_name)


def load_channel_config_by_name(settings: QSettings) -> dict[str, dict]:
    return _load_config_by_name(settings, CHANNEL_CONFIG_SETTINGS_KEY)


def save_channel_config(settings: QSettings, channels: tuple[ChannelConfig, ...]) -> None:
    payload = {
        channel.name: {
            "level_db": float(channel.level_db),
            "polarity": int(channel.polarity),
            "delay_ms": float(channel.delay_ms),
            "hpf": crossover_settings(channel.hpf),
            "lpf": crossover_settings(channel.lpf),
        }
        for channel in channels
    }
    save_channel_config_by_name(settings, payload)


def save_channel_config_by_name(settings: QSettings, config_by_name: dict) -> None:
    _save_config_by_name(settings, CHANNEL_CONFIG_SETTINGS_KEY, config_by_name)


def channel_configs(settings: QSettings) -> tuple[ChannelConfig, ...]:
    raw = load_channel_config_by_name(settings)
    if not raw:
        return (ChannelConfig(name="main"),)
    channels = []
    for name, payload in sorted(raw.items()):
        payload = payload if isinstance(payload, dict) else {}
        channels.append(
            ChannelConfig(
                name=str(name),
                level_db=float(payload.get("level_db", 0.0)),
                polarity=int(payload.get("polarity", 1)),
                delay_ms=float(payload.get("delay_ms", 0.0)),
                hpf=saved_crossover(payload.get("hpf"), crossover_type="highpass"),
                lpf=saved_crossover(payload.get("lpf"), crossover_type="lowpass"),
            )
        )
    return tuple(channels) or (ChannelConfig(name="main"),)


def channels_for_solver_radiators(
    configured_channels: tuple[ChannelConfig, ...],
    radiators: tuple[RadiatorConfig, ...],
) -> tuple[ChannelConfig, ...]:
    channels = list(configured_channels)
    configured_names = {channel.name for channel in channels}
    for radiator in radiators:
        if radiator.channel in configured_names:
            continue
        channels.append(ChannelConfig(name=radiator.channel))
        configured_names.add(radiator.channel)
    return tuple(channels)


def crossover_settings(crossover: CrossoverConfig | None) -> dict:
    if crossover is None or crossover.type.lower() == "none":
        return {}
    return {
        "type": crossover.type,
        "filter": crossover.filter,
        "order": int(crossover.order),
        "frequency_hz": None if crossover.frequency_hz is None else float(crossover.frequency_hz),
    }


def saved_crossover(raw: object, *, crossover_type: str) -> CrossoverConfig:
    if not isinstance(raw, dict) or raw.get("frequency_hz") is None:
        return CrossoverConfig()
    return CrossoverConfig(
        type=crossover_type,
        filter=str(raw.get("filter", "butterworth")).lower(),
        order=int(raw.get("order", 1)),
        frequency_hz=float(raw["frequency_hz"]),
    )


def apply_saved_source_config_to_result(
    result: AthRunResult | None,
    mesh_name: str,
    config_by_name: dict[str, dict],
) -> AthRunResult | None:
    if result is None:
        return None
    try:
        surface_tags = {
            f"{mesh_name}:{surface_name}": (mesh_name, tag)
            for surface_name, tag in read_surface_physical_names(Path(result.solver_msh_path)).items()
        }
    except Exception:
        return result

    existing_by_tag = {radiator.tag: radiator for radiator in result.radiators}
    radiators = []
    for surface_name, (mesh_name, tag) in sorted(surface_tags.items(), key=lambda item: (item[1][0], item[1][1], item[0])):
        saved = config_by_name.get(surface_name)
        if isinstance(saved, dict):
            if not bool(saved.get("driven", False)):
                continue
            radiators.append(
                RadiatorConfig(
                    name=surface_name,
                    mesh=mesh_name,
                    tag=tag,
                    channel=str(saved.get("channel", "main")),
                    velocity_offset_db=float(saved.get("velocity_offset_db", 0.0)),
                )
            )
            continue

        existing = existing_by_tag.get(tag)
        if existing is not None:
            radiators.append(
                replace(
                    existing,
                    name=surface_name,
                    mesh=mesh_name,
                    tag=tag,
                    channel=existing.channel or "main",
                    velocity_offset_db=float(existing.velocity_offset_db),
                )
            )

    return replace(result, radiators=tuple(radiators))


def apply_saved_imported_source_config(
    *,
    surface_tags: dict[str, tuple[str, int]],
    generated_mesh_names: set[str],
    existing_radiators: tuple[RadiatorConfig, ...],
    config_by_name: dict[str, dict],
) -> tuple[RadiatorConfig, ...]:
    existing_by_key = {(radiator.mesh, radiator.tag): radiator for radiator in existing_radiators}
    radiators = []
    for surface_name, (mesh_name, tag) in sorted(surface_tags.items(), key=lambda item: (item[1][0], item[1][1], item[0])):
        if mesh_name in generated_mesh_names:
            continue
        saved = config_by_name.get(surface_name)
        existing = existing_by_key.get((mesh_name, tag))
        if isinstance(saved, dict):
            if not bool(saved.get("driven", False)):
                continue
            radiators.append(
                RadiatorConfig(
                    name=surface_name,
                    mesh=mesh_name,
                    tag=tag,
                    channel=str(saved.get("channel", "main")),
                    velocity_offset_db=float(saved.get("velocity_offset_db", 0.0)),
                )
            )
        elif existing is not None:
            radiators.append(existing)
    return tuple(radiators)


def _load_config_by_name(settings: QSettings, key: str) -> dict[str, dict]:
    raw_config = settings.value(key, "{}")
    try:
        loaded = json.loads(str(raw_config))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _save_config_by_name(settings: QSettings, key: str, config_by_name: dict) -> None:
    payload = config_by_name if isinstance(config_by_name, dict) else {}
    settings.setValue(key, json.dumps(payload, sort_keys=True))
    settings.sync()
