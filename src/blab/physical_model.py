"""Solver-independent physical product model and compiled-system data types.

The authoring model describes loudspeaker physics in terms of meshes, acoustic
regions, boundaries, interfaces, components, and excitation ports. The compiled
model resolves named mesh groups and interface topology into an immutable
contract suitable for numerical backends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from typing import Any

from blab.interface_conform import InterfaceTopologyMap

PHYSICAL_MODEL_VERSION = 1
COMPILED_SYSTEM_VERSION = 1

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class MeshPurpose(StrEnum):
    FEM_VOLUME = "fem_volume"
    BEM_SURFACE = "bem_surface"


class AcousticRegionKind(StrEnum):
    BOUNDED_AIR = "bounded_air"
    UNBOUNDED_AIR = "unbounded_air"


class BoundaryKind(StrEnum):
    RIGID = "rigid"
    MOVING = "moving"
    INTERFACE = "interface"
    IMPEDANCE = "impedance"
    UNUSED = "unused"


class ComponentKind(StrEnum):
    IDEAL_VELOCITY_SOURCE = "ideal_velocity_source"
    ELECTRODYNAMIC_TRANSDUCER = "electrodynamic_transducer"
    PASSIVE_RADIATOR = "passive_radiator"


class ExcitationPortKind(StrEnum):
    NORMAL_VELOCITY = "normal_velocity"
    VOLTAGE = "voltage"


class AssumptionStatus(StrEnum):
    INCLUDED = "included"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class PhysicalGroupRef:
    """Reference a Gmsh physical group by stable name and optional tag."""

    mesh_id: str
    dimension: int
    name: str | None = None
    tag: int | None = None


@dataclass(frozen=True)
class MeshResource:
    id: str
    name: str
    file: str
    purpose: MeshPurpose
    scale_to_m: float = 1.0
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class AcousticRegion:
    id: str
    name: str
    kind: AcousticRegionKind
    mesh_ids: tuple[str, ...]
    volume_groups: tuple[PhysicalGroupRef, ...] = ()
    sound_speed_m_per_s: float = 343.0
    density_kg_per_m3: float = 1.21
    loss_model: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Boundary:
    id: str
    name: str
    region_id: str
    group: PhysicalGroupRef
    kind: BoundaryKind
    parameters: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class AcousticInterface:
    id: str
    name: str
    bounded_boundary_id: str
    unbounded_boundary_id: str
    coordinate_tolerance_m: float = 1e-8


@dataclass(frozen=True)
class PhysicalComponent:
    id: str
    name: str
    kind: ComponentKind
    boundary_ids: tuple[str, ...]
    parameters: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ExcitationPort:
    """An independently driven physical input used to form a unit-response basis."""

    id: str
    name: str
    component_id: str
    kind: ExcitationPortKind


@dataclass(frozen=True)
class PhysicalSystem:
    id: str
    name: str
    meshes: tuple[MeshResource, ...]
    regions: tuple[AcousticRegion, ...]
    boundaries: tuple[Boundary, ...]
    interfaces: tuple[AcousticInterface, ...] = ()
    components: tuple[PhysicalComponent, ...] = ()
    excitation_ports: tuple[ExcitationPort, ...] = ()
    model_version: int = PHYSICAL_MODEL_VERSION
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedPhysicalGroup:
    mesh_id: str
    dimension: int
    tag: int
    name: str | None
    element_count: int


@dataclass(frozen=True)
class CompiledMesh:
    id: str
    name: str
    file: str
    purpose: MeshPurpose
    scale_to_m: float
    translation_m: tuple[float, float, float]
    point_count: int
    triangle_count: int
    tetrahedron_count: int


@dataclass(frozen=True)
class CompiledRegion:
    id: str
    name: str
    kind: AcousticRegionKind
    mesh_ids: tuple[str, ...]
    volume_groups: tuple[ResolvedPhysicalGroup, ...]
    sound_speed_m_per_s: float
    density_kg_per_m3: float
    loss_model: dict[str, JsonValue]


@dataclass(frozen=True)
class CompiledBoundary:
    id: str
    name: str
    region_id: str
    kind: BoundaryKind
    group: ResolvedPhysicalGroup
    parameters: dict[str, JsonValue]


@dataclass(frozen=True)
class CompiledInterface:
    id: str
    name: str
    bounded_boundary_id: str
    unbounded_boundary_id: str
    topology: InterfaceTopologyMap


@dataclass(frozen=True)
class PhysicsAssumption:
    status: AssumptionStatus
    statement: str


@dataclass(frozen=True)
class CompiledPhysicalSystem:
    id: str
    name: str
    meshes: tuple[CompiledMesh, ...]
    regions: tuple[CompiledRegion, ...]
    boundaries: tuple[CompiledBoundary, ...]
    interfaces: tuple[CompiledInterface, ...]
    components: tuple[PhysicalComponent, ...]
    excitation_ports: tuple[ExcitationPort, ...]
    assumptions: tuple[PhysicsAssumption, ...]
    source_model_version: int = PHYSICAL_MODEL_VERSION
    contract_version: int = COMPILED_SYSTEM_VERSION
    metadata: dict[str, JsonValue] = field(default_factory=dict)


def object_by_id(items: tuple[Any, ...], object_id: str, *, label: str) -> Any:
    """Return an object by id with a domain-oriented error."""

    for item in items:
        if item.id == object_id:
            return item
    raise ValueError(f"Unknown {label} id: {object_id}")


def physical_system_to_dict(system: PhysicalSystem) -> dict[str, Any]:
    """Serialize the editable physical model for project persistence."""

    return _to_json_value(asdict(system))


def physical_system_from_dict(raw: dict[str, Any]) -> PhysicalSystem:
    """Deserialize a versioned editable physical model from a project."""

    model_version = int(raw.get("model_version", 0))
    if model_version != PHYSICAL_MODEL_VERSION:
        raise ValueError(f"Unsupported physical model_version {model_version}.")
    return PhysicalSystem(
        id=str(raw["id"]),
        name=str(raw["name"]),
        meshes=tuple(
            MeshResource(
                id=str(item["id"]),
                name=str(item["name"]),
                file=str(item["file"]),
                purpose=MeshPurpose(str(item["purpose"])),
                scale_to_m=float(item.get("scale_to_m", 1.0)),
                translation_m=tuple(float(value) for value in item.get("translation_m", (0.0, 0.0, 0.0))),
            )
            for item in raw.get("meshes", ())
        ),
        regions=tuple(
            AcousticRegion(
                id=str(item["id"]),
                name=str(item["name"]),
                kind=AcousticRegionKind(str(item["kind"])),
                mesh_ids=tuple(str(value) for value in item.get("mesh_ids", ())),
                volume_groups=tuple(_physical_group_ref_from_dict(group) for group in item.get("volume_groups", ())),
                sound_speed_m_per_s=float(item.get("sound_speed_m_per_s", 343.0)),
                density_kg_per_m3=float(item.get("density_kg_per_m3", 1.21)),
                loss_model=dict(item.get("loss_model", {})),
            )
            for item in raw.get("regions", ())
        ),
        boundaries=tuple(
            Boundary(
                id=str(item["id"]),
                name=str(item["name"]),
                region_id=str(item["region_id"]),
                group=_physical_group_ref_from_dict(item["group"]),
                kind=BoundaryKind(str(item["kind"])),
                parameters=dict(item.get("parameters", {})),
            )
            for item in raw.get("boundaries", ())
        ),
        interfaces=tuple(
            AcousticInterface(
                id=str(item["id"]),
                name=str(item["name"]),
                bounded_boundary_id=str(item["bounded_boundary_id"]),
                unbounded_boundary_id=str(item["unbounded_boundary_id"]),
                coordinate_tolerance_m=float(item.get("coordinate_tolerance_m", 1e-8)),
            )
            for item in raw.get("interfaces", ())
        ),
        components=tuple(
            PhysicalComponent(
                id=str(item["id"]),
                name=str(item["name"]),
                kind=ComponentKind(str(item["kind"])),
                boundary_ids=tuple(str(value) for value in item.get("boundary_ids", ())),
                parameters=dict(item.get("parameters", {})),
            )
            for item in raw.get("components", ())
        ),
        excitation_ports=tuple(
            ExcitationPort(
                id=str(item["id"]),
                name=str(item["name"]),
                component_id=str(item["component_id"]),
                kind=ExcitationPortKind(str(item["kind"])),
            )
            for item in raw.get("excitation_ports", ())
        ),
        model_version=model_version,
        metadata=dict(raw.get("metadata", {})),
    )


def _physical_group_ref_from_dict(raw: dict[str, Any]) -> PhysicalGroupRef:
    return PhysicalGroupRef(
        mesh_id=str(raw["mesh_id"]),
        dimension=int(raw["dimension"]),
        name=None if raw.get("name") is None else str(raw["name"]),
        tag=None if raw.get("tag") is None else int(raw["tag"]),
    )


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json_value(item) for item in value]
    return value
