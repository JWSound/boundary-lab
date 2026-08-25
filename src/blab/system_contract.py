"""Versioned wire contract for compiled multiphysics system solves.

This contract intentionally lives beside the legacy BEM protocol. Numerical
backends can adopt it incrementally without changing existing Boundary Lab
projects or solver sessions.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Protocol

import numpy as np

from blab.interface_conform import InterfaceTopologyMap
from blab.physical_model import (
    COMPILED_SYSTEM_VERSION,
    AcousticRegionKind,
    AssumptionStatus,
    BoundaryKind,
    CompiledBoundary,
    CompiledInterface,
    CompiledMesh,
    CompiledPhysicalSystem,
    CompiledRegion,
    ComponentKind,
    ExcitationPort,
    ExcitationPortKind,
    MeshPurpose,
    PhysicalComponent,
    PhysicsAssumption,
    ResolvedPhysicalGroup,
)

SYSTEM_SOLVE_REQUEST_VERSION = 1
SYSTEM_RESULT_VERSION = 2
SUPPORTED_SYSTEM_RESULT_VERSIONS = {1, SYSTEM_RESULT_VERSION}


@dataclass(frozen=True)
class OutputRequest:
    id: str
    quantity: str
    target_ids: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemSolveRequest:
    compiled_system: CompiledPhysicalSystem
    frequencies_hz: tuple[float, ...]
    excitation_port_ids: tuple[str, ...]
    outputs: tuple[OutputRequest, ...] = ()
    solver_options: dict[str, Any] = field(default_factory=dict)
    status_callback: Callable[[str], None] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class QuantityResult:
    id: str
    quantity: str
    unit: str
    values: np.ndarray
    target_id: str | None = None
    axes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemFrequencyResult:
    freq_hz: float
    quantities: tuple[QuantityResult, ...]
    excitation_port_ids: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemSolveMetadata:
    system_id: str
    assumptions: tuple[PhysicsAssumption, ...]
    excitation_port_ids: tuple[str, ...] = ()
    available_quantity_ids: tuple[str, ...] = ()


class SystemSolverSession(Protocol):
    @property
    def metadata(self) -> SystemSolveMetadata: ...

    def solve_stream(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[SystemFrequencyResult]: ...

    def stop(self) -> None: ...


class SystemSolverBackend(Protocol):
    backend_id: str
    label: str

    def create_system_session(self, request: SystemSolveRequest) -> SystemSolverSession: ...


def validate_system_solve_request(request: SystemSolveRequest) -> None:
    if not request.frequencies_hz:
        raise ValueError("System solve request must contain at least one frequency.")
    frequencies = np.asarray(request.frequencies_hz, dtype=float)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("System solve frequencies must be finite and greater than zero.")
    if not request.excitation_port_ids:
        raise ValueError("System solve request must contain at least one excitation port id.")
    if len(request.excitation_port_ids) != len(set(request.excitation_port_ids)):
        raise ValueError("Each excitation port may appear only once in a system solve request.")
    available_port_ids = {port.id for port in request.compiled_system.excitation_ports}
    for port_id in request.excitation_port_ids:
        if port_id not in available_port_ids:
            raise ValueError(f"System solve request references unknown excitation port '{port_id}'.")
    output_ids = [output.id for output in request.outputs]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("System solve output request ids must be unique.")
    if any(not output.id.strip() or not output.quantity.strip() for output in request.outputs):
        raise ValueError("System solve outputs require non-empty id and quantity.")
    _validate_json_value(request.solver_options, label="solver_options")
    for output in request.outputs:
        _validate_json_value(output.options, label=f"output '{output.id}' options")


def validate_system_frequency_result(result: SystemFrequencyResult) -> None:
    if not np.isfinite(result.freq_hz) or result.freq_hz <= 0.0:
        raise ValueError("System result frequency must be finite and greater than zero.")
    quantity_ids = [quantity.id for quantity in result.quantities]
    if len(quantity_ids) != len(set(quantity_ids)):
        raise ValueError("System result quantity ids must be unique within a frequency.")
    if len(result.excitation_port_ids) != len(set(result.excitation_port_ids)):
        raise ValueError("System result excitation port ids must be unique.")
    for quantity in result.quantities:
        if not quantity.id.strip() or not quantity.quantity.strip() or not quantity.unit.strip():
            raise ValueError("System result quantities require non-empty id, quantity, and unit.")
        if quantity.axes and len(quantity.axes) != np.asarray(quantity.values).ndim:
            raise ValueError(
                f"System result quantity '{quantity.id}' has {len(quantity.axes)} axes "
                f"for a {np.asarray(quantity.values).ndim}-dimensional array."
            )
        if "excitation" in quantity.axes:
            axis_index = quantity.axes.index("excitation")
            if np.asarray(quantity.values).shape[axis_index] != len(result.excitation_port_ids):
                raise ValueError(
                    f"System result quantity '{quantity.id}' excitation axis has length "
                    f"{np.asarray(quantity.values).shape[axis_index]}, but the result declares "
                    f"{len(result.excitation_port_ids)} excitation ports."
                )
        _validate_json_value(quantity.metadata, label=f"quantity '{quantity.id}' metadata")
    _validate_json_value(result.diagnostics, label="result diagnostics")


def compiled_system_to_dict(system: CompiledPhysicalSystem) -> dict[str, Any]:
    return _to_wire_value(asdict(system))


def compiled_system_from_dict(raw: dict[str, Any]) -> CompiledPhysicalSystem:
    contract_version = int(raw.get("contract_version", 0))
    if contract_version != COMPILED_SYSTEM_VERSION:
        raise ValueError(f"Unsupported compiled system contract_version {contract_version}.")
    return CompiledPhysicalSystem(
        id=str(raw["id"]),
        name=str(raw["name"]),
        meshes=tuple(_compiled_mesh_from_dict(item) for item in raw.get("meshes", ())),
        regions=tuple(_compiled_region_from_dict(item) for item in raw.get("regions", ())),
        boundaries=tuple(_compiled_boundary_from_dict(item) for item in raw.get("boundaries", ())),
        interfaces=tuple(_compiled_interface_from_dict(item) for item in raw.get("interfaces", ())),
        components=tuple(_component_from_dict(item) for item in raw.get("components", ())),
        excitation_ports=tuple(_excitation_port_from_dict(item) for item in raw.get("excitation_ports", ())),
        assumptions=tuple(_assumption_from_dict(item) for item in raw.get("assumptions", ())),
        source_model_version=int(raw.get("source_model_version", 1)),
        contract_version=contract_version,
        metadata=dict(raw.get("metadata", {})),
    )


def system_solve_request_to_dict(request: SystemSolveRequest) -> dict[str, Any]:
    validate_system_solve_request(request)
    return {
        "schema_version": SYSTEM_SOLVE_REQUEST_VERSION,
        "compiled_system": compiled_system_to_dict(request.compiled_system),
        "frequencies_hz": [float(value) for value in request.frequencies_hz],
        "excitation_port_ids": list(request.excitation_port_ids),
        "outputs": [
            {
                "id": output.id,
                "quantity": output.quantity,
                "target_ids": list(output.target_ids),
                "options": _to_wire_value(output.options),
            }
            for output in request.outputs
        ],
        "solver_options": _to_wire_value(request.solver_options),
    }


def system_solve_request_from_dict(raw: dict[str, Any]) -> SystemSolveRequest:
    version = int(raw.get("schema_version", 0))
    if version != SYSTEM_SOLVE_REQUEST_VERSION:
        raise ValueError(f"Unsupported system solve request schema_version {version}.")
    request = SystemSolveRequest(
        compiled_system=compiled_system_from_dict(raw["compiled_system"]),
        frequencies_hz=tuple(float(value) for value in raw.get("frequencies_hz", ())),
        excitation_port_ids=tuple(str(value) for value in raw.get("excitation_port_ids", ())),
        outputs=tuple(
            OutputRequest(
                id=str(item["id"]),
                quantity=str(item["quantity"]),
                target_ids=tuple(str(value) for value in item.get("target_ids", ())),
                options=dict(item.get("options", {})),
            )
            for item in raw.get("outputs", ())
        ),
        solver_options=dict(raw.get("solver_options", {})),
    )
    validate_system_solve_request(request)
    return request


def system_frequency_result_to_dict(result: SystemFrequencyResult) -> dict[str, Any]:
    validate_system_frequency_result(result)
    return {
        "schema_version": SYSTEM_RESULT_VERSION,
        "freq_hz": float(result.freq_hz),
        "excitation_port_ids": list(result.excitation_port_ids),
        "quantities": [
            {
                "id": quantity.id,
                "quantity": quantity.quantity,
                "unit": quantity.unit,
                "target_id": quantity.target_id,
                "axes": list(quantity.axes),
                "values": _array_to_wire(quantity.values),
                "metadata": _to_wire_value(quantity.metadata),
            }
            for quantity in result.quantities
        ],
        "diagnostics": _to_wire_value(result.diagnostics),
    }


def system_frequency_result_from_dict(raw: dict[str, Any]) -> SystemFrequencyResult:
    version = int(raw.get("schema_version", 0))
    if version not in SUPPORTED_SYSTEM_RESULT_VERSIONS:
        raise ValueError(f"Unsupported system result schema_version {version}.")
    result = SystemFrequencyResult(
        freq_hz=float(raw["freq_hz"]),
        quantities=tuple(
            QuantityResult(
                id=str(item["id"]),
                quantity=str(item["quantity"]),
                unit=str(item["unit"]),
                target_id=None if item.get("target_id") is None else str(item["target_id"]),
                axes=tuple(str(value) for value in item.get("axes", ())),
                values=_array_from_wire(item["values"]),
                metadata=dict(item.get("metadata", {})),
            )
            for item in raw.get("quantities", ())
        ),
        excitation_port_ids=tuple(str(value) for value in raw.get("excitation_port_ids", ())),
        diagnostics=dict(raw.get("diagnostics", {})),
    )
    validate_system_frequency_result(result)
    return result


def _compiled_mesh_from_dict(raw: dict[str, Any]) -> CompiledMesh:
    return CompiledMesh(
        id=str(raw["id"]),
        name=str(raw["name"]),
        file=str(raw["file"]),
        purpose=MeshPurpose(str(raw["purpose"])),
        scale_to_m=float(raw["scale_to_m"]),
        translation_m=tuple(float(value) for value in raw.get("translation_m", (0.0, 0.0, 0.0))),
    )


def _resolved_group_from_dict(raw: dict[str, Any]) -> ResolvedPhysicalGroup:
    return ResolvedPhysicalGroup(
        mesh_id=str(raw["mesh_id"]),
        dimension=int(raw["dimension"]),
        tag=int(raw["tag"]),
        name=None if raw.get("name") is None else str(raw["name"]),
    )


def _compiled_region_from_dict(raw: dict[str, Any]) -> CompiledRegion:
    return CompiledRegion(
        id=str(raw["id"]),
        name=str(raw["name"]),
        kind=AcousticRegionKind(str(raw["kind"])),
        mesh_ids=tuple(str(value) for value in raw.get("mesh_ids", ())),
        volume_groups=tuple(_resolved_group_from_dict(item) for item in raw.get("volume_groups", ())),
        sound_speed_m_per_s=float(raw["sound_speed_m_per_s"]),
        density_kg_per_m3=float(raw["density_kg_per_m3"]),
        loss_model=dict(raw.get("loss_model", {})),
    )


def _compiled_boundary_from_dict(raw: dict[str, Any]) -> CompiledBoundary:
    return CompiledBoundary(
        id=str(raw["id"]),
        name=str(raw["name"]),
        region_id=str(raw["region_id"]),
        kind=BoundaryKind(str(raw["kind"])),
        group=_resolved_group_from_dict(raw["group"]),
        parameters=dict(raw.get("parameters", {})),
    )


def _compiled_interface_from_dict(raw: dict[str, Any]) -> CompiledInterface:
    topology = raw["topology"]
    return CompiledInterface(
        id=str(raw["id"]),
        name=str(raw["name"]),
        bounded_boundary_id=str(raw["bounded_boundary_id"]),
        unbounded_boundary_id=str(raw["unbounded_boundary_id"]),
        topology=InterfaceTopologyMap(
            fem_vertex_indices=tuple(int(value) for value in topology.get("fem_vertex_indices", ())),
            fem_to_bem_vertex_indices=tuple(int(value) for value in topology.get("fem_to_bem_vertex_indices", ())),
            fem_face_indices=tuple(int(value) for value in topology.get("fem_face_indices", ())),
            bem_face_indices=tuple(int(value) for value in topology.get("bem_face_indices", ())),
            normal_sign=tuple(int(value) for value in topology.get("normal_sign", ())),
            max_coordinate_error=float(topology["max_coordinate_error"]),
            fem_facets_on_tetra_boundary=int(topology["fem_facets_on_tetra_boundary"]),
            bem_boundary_edges=int(topology["bem_boundary_edges"]),
        ),
    )


def _component_from_dict(raw: dict[str, Any]) -> PhysicalComponent:
    return PhysicalComponent(
        id=str(raw["id"]),
        name=str(raw["name"]),
        kind=ComponentKind(str(raw["kind"])),
        boundary_ids=tuple(str(value) for value in raw.get("boundary_ids", ())),
        parameters=dict(raw.get("parameters", {})),
    )


def _excitation_port_from_dict(raw: dict[str, Any]) -> ExcitationPort:
    return ExcitationPort(
        id=str(raw["id"]),
        name=str(raw["name"]),
        component_id=str(raw["component_id"]),
        kind=ExcitationPortKind(str(raw["kind"])),
    )


def _assumption_from_dict(raw: dict[str, Any]) -> PhysicsAssumption:
    return PhysicsAssumption(
        status=AssumptionStatus(str(raw["status"])),
        statement=str(raw["statement"]),
    )


def _array_to_wire(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    if array.dtype.kind not in {"f", "i", "u", "c", "b"}:
        raise ValueError(f"Unsupported result array dtype: {array.dtype}")
    little_endian_dtype = array.dtype.newbyteorder("<")
    contiguous = np.ascontiguousarray(array, dtype=little_endian_dtype)
    return {
        "encoding": "base64",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "order": "C",
        "byte_order": "little",
        "content_base64": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
    }


def _array_from_wire(raw: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(str(raw["dtype"]))
    if dtype.kind not in {"f", "i", "u", "c", "b"}:
        raise ValueError(f"Unsupported result array dtype: {dtype}")
    shape = tuple(int(value) for value in raw.get("shape", ()))
    if any(value < 0 for value in shape):
        raise ValueError(f"Result array shape must be nonnegative: {shape}.")
    expected_size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if "content_base64" in raw:
        if str(raw.get("encoding", "")) != "base64":
            raise ValueError("Result array binary payload must use base64 encoding.")
        if str(raw.get("order", "")) != "C":
            raise ValueError("Result array binary payload must use row-major order.")
        if str(raw.get("byte_order", "")) != "little":
            raise ValueError("Result array binary payload must use little-endian byte order.")
        try:
            payload = base64.b64decode(str(raw["content_base64"]), validate=True)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise ValueError("Result array contains invalid base64 data.") from exc
        expected_nbytes = expected_size * dtype.itemsize
        if len(payload) != expected_nbytes:
            raise ValueError(
                f"Result array payload contains {len(payload)} bytes, expected {expected_nbytes} "
                f"for dtype {dtype} and shape {shape}."
            )
        wire_dtype = dtype.newbyteorder("<")
        values = np.frombuffer(payload, dtype=wire_dtype, count=expected_size)
        native_dtype = dtype.newbyteorder("=")
        return np.array(values, dtype=native_dtype, copy=True).reshape(shape)

    # Schema-v1 compatibility for decimal real/imag list payloads.
    if dtype.kind == "c":
        real_dtype = np.empty((), dtype=dtype).real.dtype
        real = np.asarray(raw.get("real", ()), dtype=real_dtype)
        imag = np.asarray(raw.get("imag", ()), dtype=real_dtype)
        values = (real + 1j * imag).astype(dtype, copy=False)
    else:
        values = np.asarray(raw.get("real", ()), dtype=dtype)
    if values.size != expected_size:
        raise ValueError(
            f"Result array payload contains {values.size} values, expected {expected_size} for shape {shape}."
        )
    return values.reshape(shape)


def _to_wire_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _to_wire_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_wire_value(item) for item in value]
    return value


def _validate_json_value(value: Any, *, label: str) -> None:
    try:
        json.dumps(_to_wire_value(value), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"System contract {label} must contain finite JSON values.") from exc
