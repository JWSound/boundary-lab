"""Qt-free, transactional configuration patches for geometry providers.

Only explicit fields change. Collections use upsert/remove operations; nested
objects merge, while supplied arrays replace. Frequency settings are deliberately
absent. Mesh files remain owned by the generation/artifact path.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields, replace
from hashlib import sha256
from typing import Any

from blab.config import ChannelConfig, CrossoverConfig, normalize_symmetry
from blab.physical_model import (
    AcousticInterface,
    AcousticRegion,
    Boundary,
    ExcitationPort,
    PhysicalComponent,
    PhysicalGroupRef,
    physical_system_from_dict,
    physical_system_to_dict,
)
from blab.project.migration import AUTO_SEEDED_EXTERIOR_KEY
from blab.project.model import ProjectDocument, ProjectPreferencesState

ENTITY_SECTIONS = {
    "regions": ("regions", AcousticRegion),
    "surface_assignments": ("boundaries", Boundary),
    "interfaces": ("interfaces", AcousticInterface),
    "component_assignments": ("components", PhysicalComponent),
    "excitation_ports": ("excitation_ports", ExcitationPort),
}
PATCH_SECTIONS = frozenset(ENTITY_SECTIONS) | {
    "component_parameters", "channel_config", "component_channels", "symmetry_config", "stitching_config",
}


def configuration_snapshot(project: ProjectDocument) -> dict[str, Any]:
    """Editable configuration exposed to providers, with host entity IDs."""
    return {
        "physical_system": (
            None if project.physical_system is None else physical_system_to_dict(project.physical_system)
        ),
        "channel_config": deepcopy(project.channel_config_by_name),
        "component_channels": dict(project.component_channel_by_id),
        "symmetry_config": {"mode": project.symmetry},
        "stitching_config": {
            "enabled": project.stitch_imported_meshes,
            "tolerance_mm": (project.project_preferences or ProjectPreferencesState()).stitch_tolerance_mm,
        },
    }


def project_revision(project: ProjectDocument) -> str:
    """Content revision includes artifacts and input source, not active-tab state."""
    payload = asdict(project)
    payload.pop("active_generator_document_id", None)
    return sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _object(value: Any, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{label} keys must be nonempty strings.")
    return dict(value)


def _keys(value: dict, allowed: set | frozenset, label: str) -> None:
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")


def _operations(value: Any, label: str, key: str = "id") -> tuple[list[dict], list[str]]:
    value = _object(value, label)
    _keys(value, {"upsert", "remove"}, label)
    upsert, remove = value.get("upsert", []), value.get("remove", [])
    if not isinstance(upsert, list) or not isinstance(remove, list):
        raise ValueError(f"{label}.upsert and .remove must be arrays.")
    items = [_object(item, label) for item in upsert]
    ids = [item.get(key) for item in items]
    if any(not isinstance(item, str) or not item.strip() for item in [*ids, *remove]):
        raise ValueError(f"{label} requires nonempty {key} values.")
    if len(set(ids + remove)) != len(ids + remove):
        raise ValueError(f"Duplicate or conflicting operations in {label}.")
    return items, remove


def _merge(current: dict, patch: dict) -> dict:
    result = deepcopy(current)
    for key, value in patch.items():
        if value is None:
            raise ValueError(f"Null is not a removal operation ({key}); use explicit remove.")
        if isinstance(value, dict):
            result[key] = _merge(result.get(key, {}) if isinstance(result.get(key), dict) else {}, value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_patch(raw: Mapping[str, Any]) -> dict[str, Any]:
    patch = deepcopy(_object(raw, "configuration_patch"))
    _keys(patch, PATCH_SECTIONS, "configuration_patch")
    try:
        json.dumps(patch, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("Configuration patches must contain finite JSON values.") from exc
    for section, (_, model) in ENTITY_SECTIONS.items():
        if section in patch:
            items, _ = _operations(patch[section], section)
            for item in items:
                _keys(item, {field.name for field in fields(model)}, section)
                for key in ("name", "kind", "region_id", "component_id", "bounded_boundary_id", "unbounded_boundary_id"):
                    if key in item and (not isinstance(item[key], str) or not item[key].strip()):
                        raise ValueError(f"{section}.{key} must be a nonempty string.")
                for key in ("parameters", "loss_model"):
                    if key in item:
                        _object(item[key], key)
                for key in ("mesh_ids", "boundary_ids"):
                    if key in item and (
                        not isinstance(item[key], list)
                        or any(not isinstance(value, str) or not value for value in item[key])
                    ):
                        raise ValueError(f"{section}.{key} must be an array of IDs.")
                if "group" in item:
                    _keys(_object(item["group"], "group"), {field.name for field in fields(PhysicalGroupRef)}, "group")
                if "volume_groups" in item:
                    if not isinstance(item["volume_groups"], list):
                        raise ValueError("volume_groups must be an array.")
                    for group in item["volume_groups"]:
                        _keys(_object(group, "group"), {field.name for field in fields(PhysicalGroupRef)}, "group")
                _merge({}, item)  # Reject ambiguous nulls even before application.
    for section in ("channel_config", "component_channels"):
        if section in patch:
            items, _ = _operations(patch[section], section, "name" if section == "channel_config" else "id")
            for item in items:
                allowed = {field.name for field in fields(ChannelConfig)} if section == "channel_config" else {"id", "channel"}
                _keys(item, allowed, section)
                _merge({}, item)
    if "component_parameters" in patch:
        for value in _object(patch["component_parameters"], "component_parameters").values():
            _merge({}, _object(value, "component parameters"))
    for section, allowed in (
        ("symmetry_config", {"mode"}), ("stitching_config", {"enabled", "tolerance_mm"}),
    ):
        if section in patch:
            value = _object(patch[section], section)
            _keys(value, allowed, section)
            _merge({}, value)
    return patch


def _owned_entities(system, mesh_ids: set[str]) -> dict[str, set[str]]:
    boundaries = {item.id for item in system.boundaries if item.group.mesh_id in mesh_ids}
    components = {
        item.id for item in system.components if item.boundary_ids and set(item.boundary_ids) <= boundaries
    }
    return {
        "regions": {item.id for item in system.regions if item.mesh_ids and set(item.mesh_ids) <= mesh_ids},
        "boundaries": boundaries,
        "components": components,
        "excitation_ports": {item.id for item in system.excitation_ports if item.component_id in components},
        "interfaces": {
            item.id for item in system.interfaces
            if {item.bounded_boundary_id, item.unbounded_boundary_id} <= boundaries
        },
    }


def _check_references(system) -> None:
    meshes = {item.id for item in system.meshes}
    regions = {item.id: item for item in system.regions}
    boundaries = {item.id for item in system.boundaries}
    components = {item.id for item in system.components}
    for region in system.regions:
        if not region.mesh_ids or not set(region.mesh_ids) <= meshes:
            raise ValueError(f"Region {region.id!r} references missing meshes.")
        if any(group.mesh_id not in region.mesh_ids for group in region.volume_groups):
            raise ValueError(f"Region {region.id!r} has an invalid volume group reference.")
    for boundary in system.boundaries:
        region = regions.get(boundary.region_id)
        if region is None or boundary.group.mesh_id not in region.mesh_ids:
            raise ValueError(f"Boundary {boundary.id!r} references a missing region or mesh.")
        if boundary.group.dimension != 2 or (boundary.group.name is None and boundary.group.tag is None):
            raise ValueError(f"Boundary {boundary.id!r} requires a surface group reference.")
    for component in system.components:
        if not component.boundary_ids or not set(component.boundary_ids) <= boundaries:
            raise ValueError(f"Component {component.id!r} references missing boundaries.")
    for port in system.excitation_ports:
        if port.component_id not in components:
            raise ValueError(f"Excitation {port.id!r} references a missing component.")
    for interface in system.interfaces:
        if not {interface.bounded_boundary_id, interface.unbounded_boundary_id} <= boundaries:
            raise ValueError(f"Interface {interface.id!r} references missing boundaries.")


def apply_configuration_patch(
    project: ProjectDocument,
    patch: Mapping[str, Any],
    *,
    document_id: str,
    mesh_ids: set[str],
) -> ProjectDocument:
    """Return an independent candidate; failure never modifies the input project.

    Existing physical entities must belong entirely to the generating document's
    meshes. New IDs use ``document_id/local_id``. Shared regions may be referenced
    but cannot be rewritten by a provider owning only some of their meshes.
    """
    patch = validate_patch(patch)
    candidate = deepcopy(project)
    system = candidate.physical_system
    physical_update = any(
        patch.get(section, {}).get("upsert") or patch.get(section, {}).get("remove") for section in ENTITY_SECTIONS
    ) or any(patch.get("component_parameters", {}).values())
    owned = _owned_entities(system, mesh_ids) if system else {}
    if physical_update:
        if system is None:
            raise ValueError("Physical assignments require an existing or seeded physical system.")
        raw = physical_system_to_dict(system)
        for section, (collection, _) in ENTITY_SECTIONS.items():
            if section not in patch:
                continue
            upsert, remove = _operations(patch[section], section)
            entities = {item["id"]: item for item in raw[collection]}
            for item_id in remove:
                if item_id not in owned[collection]:
                    raise ValueError(f"Cannot remove unowned or missing {collection} entity {item_id!r}.")
                del entities[item_id]
            for item in upsert:
                item_id = item["id"]
                if item_id in entities:
                    if item_id not in owned[collection]:
                        raise ValueError(f"Cannot modify unowned {collection} entity {item_id!r}.")
                elif not item_id.startswith(f"{document_id}/") or item_id == f"{document_id}/":
                    raise ValueError(f"New entity IDs must start with {document_id}/.")
                entities[item_id] = _merge(entities.get(item_id, {}), item)
            raw[collection] = list(entities.values())
        components = {item["id"]: item for item in raw["components"]}
        for component_id, parameters in patch.get("component_parameters", {}).items():
            if component_id not in components:
                raise ValueError(f"Unknown component {component_id!r}.")
            components[component_id]["parameters"] = _merge(components[component_id].get("parameters", {}), parameters)
        try:
            candidate.physical_system = physical_system_from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid physical configuration: {exc}") from exc
        _check_references(candidate.physical_system)
        new_owned = _owned_entities(candidate.physical_system, mesh_ids)
        for section, (collection, _) in ENTITY_SECTIONS.items():
            updated = {item["id"] for item in patch.get(section, {}).get("upsert", [])}
            if not updated <= new_owned[collection]:
                raise ValueError(f"{section} must remain within the generating document's meshes.")
        if not set(patch.get("component_parameters", {})) <= new_owned["components"]:
            raise ValueError("Cannot modify parameters of an unrelated component.")
        candidate.physical_system = replace(
            candidate.physical_system,
            metadata={**candidate.physical_system.metadata, AUTO_SEEDED_EXTERIOR_KEY: False},
        )
        owned = new_owned

    channels = candidate.channel_config_by_name
    if patch.get("channel_config", {}).get("upsert") or patch.get("channel_config", {}).get("remove"):
        if not channels:
            channels["main"] = {key: value for key, value in asdict(ChannelConfig("main")).items() if key != "name"}
        upsert, remove = _operations(patch["channel_config"], "channel_config", "name")
        for name in remove:
            if name not in channels:
                raise ValueError(f"Unknown channel {name!r}.")
            del channels[name]
        for item in upsert:
            name = item["name"]
            channels[name] = _merge(channels.get(name, {}), {key: value for key, value in item.items() if key != "name"})
        if not channels:
            raise ValueError("At least one channel must remain.")
        for name, channel in channels.items():
            _validate_channel(name, channel)

    if "component_channels" in patch:
        upsert, remove = _operations(patch["component_channels"], "component_channels")
        for item_id in [*remove, *(item["id"] for item in upsert)]:
            if item_id not in owned.get("components", set()):
                raise ValueError(f"Cannot route unowned or missing component {item_id!r}.")
        for item_id in remove:
            candidate.component_channel_by_id.pop(item_id, None)
        for item in upsert:
            if not isinstance(item.get("channel"), str) or not item["channel"]:
                raise ValueError("Component routing requires a channel name.")
            candidate.component_channel_by_id[item["id"]] = item["channel"]
        if (upsert or remove) and candidate.physical_system is not None:
            candidate.physical_system = replace(
                candidate.physical_system,
                metadata={**candidate.physical_system.metadata, AUTO_SEEDED_EXTERIOR_KEY: False},
            )
    if candidate.physical_system:
        valid_components = {item.id for item in candidate.physical_system.components}
        removed_components = ({item.id for item in system.components} if system else set()) - valid_components
        for key in removed_components:
            candidate.component_channel_by_id.pop(key, None)
        channel_names = set(channels) or {"main"}
        for port in candidate.physical_system.excitation_ports:
            channel = candidate.component_channel_by_id.get(port.component_id, "main")
            if channel not in channel_names:
                raise ValueError(f"Component {port.component_id!r} references missing channel {channel!r}.")
    if "mode" in patch.get("symmetry_config", {}):
        candidate.symmetry = normalize_symmetry(patch["symmetry_config"]["mode"])
    stitching = patch.get("stitching_config", {})
    if "enabled" in stitching:
        if not isinstance(stitching["enabled"], bool):
            raise ValueError("Stitching enabled must be a boolean.")
        candidate.stitch_imported_meshes = stitching["enabled"]
    if "tolerance_mm" in stitching:
        tolerance = stitching["tolerance_mm"]
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ValueError("Stitch tolerance must be a nonnegative finite number.")
        candidate.project_preferences = replace(
            candidate.project_preferences or ProjectPreferencesState(), stitch_tolerance_mm=float(tolerance),
        )
    return candidate


def _validate_channel(name: str, raw: dict) -> None:
    for key in ("voltage_v", "level_db", "delay_ms"):
        value = raw.get(key, getattr(ChannelConfig(name), key))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Channel {name!r}: {key} must be a finite number.")
        if key == "voltage_v" and value < 0:
            raise ValueError("Channel voltage cannot be negative.")
    if type(raw.get("polarity", 1)) is not int or raw.get("polarity", 1) not in (-1, 1):
        raise ValueError("Channel polarity must be -1 or 1.")
    for key in ("hpf", "lpf"):
        value = _object(raw.get(key, {}), key)
        _keys(value, {field.name for field in fields(CrossoverConfig)}, key)
        if value.get("type", "none") not in ("none", "highpass", "lowpass"):
            raise ValueError("Unknown crossover type.")
        if value.get("type", "none") != "none":
            if value.get("filter", "butterworth") not in ("butterworth", "linkwitz_riley"):
                raise ValueError("Unknown crossover filter.")
            frequency = value.get("frequency_hz")
            if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or frequency <= 0:
                raise ValueError("An enabled crossover requires a positive frequency_hz.")
            order = value.get("order", 1)
            if type(order) is not int or order < 1:
                raise ValueError("Crossover order must be a positive integer.")
