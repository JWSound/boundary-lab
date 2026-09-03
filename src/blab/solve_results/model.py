"""Solver-independent, array-oriented storage for a solved physical system.

``SystemFrequencyResult`` remains the streaming/wire contract.  This module
assembles those per-frequency messages into one self-describing in-memory run
whose raw complex quantities can feed plots, reports, and spatial viewers.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable

import numpy as np

from blab.phasor import SOLVER_PHASOR_CONVENTION
from blab.physical_model import CompiledPhysicalSystem
from blab.system_contract import QuantityResult, SystemFrequencyResult, validate_system_frequency_result

RESULT_MODEL_VERSION = 1
PHASOR_CONVENTION = SOLVER_PHASOR_CONVENTION
HORIZONTAL_POLAR_DOMAIN_ID = "observation:horizontal-polar"
VERTICAL_POLAR_DOMAIN_ID = "observation:vertical-polar"
SPHERE_DOMAIN_ID = "observation:sphere"
RADIATOR_DOMAIN_ID = "components:radiators"
TRANSDUCER_DOMAIN_ID = "components:electrodynamic-transducers"
FEM_VOLUME_DOMAIN_ID = "domain:fem-volume"
BEM_BOUNDARY_DOMAIN_ID = "domain:bem-boundary"
HORIZONTAL_POLAR_PRESSURE_ID = "acoustic:pressure:horizontal-polar"
VERTICAL_POLAR_PRESSURE_ID = "acoustic:pressure:vertical-polar"
SPHERE_PRESSURE_ID = "acoustic:pressure:sphere"
FEM_NODAL_PRESSURE_ID = "acoustic:pressure:fem-nodes"
BEM_BOUNDARY_PRESSURE_ID = "acoustic:pressure:bem-boundary"
BEM_BOUNDARY_NEUMANN_ID = "acoustic:normal-derivative:bem-boundary"
RADIATION_IMPEDANCE_ID = "acoustic:radiation-impedance"
DIAPHRAGM_VELOCITY_ID = "mechanical:diaphragm-velocity"
VOICE_COIL_CURRENT_ID = "electrical:voice-coil-current"


def _readonly_array(values: Any, *, copy_values: bool = True) -> np.ndarray:
    array = np.array(values, copy=copy_values)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ResultDomain:
    """Frequency-invariant coordinates/topology referenced by result fields."""

    id: str
    kind: str
    dimensions: tuple[str, ...]
    coordinates: dict[str, np.ndarray] = field(default_factory=dict)
    topology: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> ResultDomain:
        return ResultDomain(
            id=self.id,
            kind=self.kind,
            dimensions=tuple(self.dimensions),
            coordinates={name: _readonly_array(values) for name, values in self.coordinates.items()},
            topology={name: _readonly_array(values) for name, values in self.topology.items()},
            metadata=copy.deepcopy(self.metadata),
        )


@dataclass(frozen=True)
class SolveProvenance:
    """Information needed to interpret a run independently of current UI state."""

    backend_id: str
    solve_kind: str
    solver_options: dict[str, Any] = field(default_factory=dict)
    phasor_convention: str = PHASOR_CONVENTION
    started_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class SolvedQuantity:
    """One physical quantity stacked over the run's frequency dimension."""

    id: str
    quantity: str
    unit: str
    dimensions: tuple[str, ...]
    values: np.ndarray
    domain_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    available_frequency_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))


@dataclass(frozen=True)
class SolvedSystem:
    """Canonical immutable snapshot of a complete or partial solve run."""

    run_id: str
    provenance: SolveProvenance
    frequencies_hz: np.ndarray
    excitation_ids: tuple[str, ...]
    domains: dict[str, ResultDomain]
    quantities: dict[str, SolvedQuantity]
    completion_mask: np.ndarray
    diagnostics_by_frequency: tuple[dict[str, Any] | None, ...]
    status: str
    compiled_system: CompiledPhysicalSystem | None = None
    schema_version: int = RESULT_MODEL_VERSION
    finished_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def solved_count(self) -> int:
        return int(np.count_nonzero(self.completion_mask))

    @property
    def complete(self) -> bool:
        return bool(self.completion_mask.size and np.all(self.completion_mask))

    def quantity(self, quantity_id: str) -> SolvedQuantity:
        try:
            return self.quantities[quantity_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.quantities)) or "(none)"
            raise KeyError(f"Unknown solved quantity {quantity_id!r}; available: {available}") from exc

    def quantities_for(self, quantity: str) -> tuple[SolvedQuantity, ...]:
        return tuple(item for item in self.quantities.values() if item.quantity == quantity)


@dataclass
class _QuantityBuffer:
    specification: QuantityResult
    values: np.ndarray
    available_frequency_mask: np.ndarray


class SolvedSystemBuilder:
    """Accumulate streamed frequency results into stable, contiguous arrays."""

    def __init__(
        self,
        *,
        frequencies_hz: Iterable[float],
        excitation_ids: Iterable[str],
        provenance: SolveProvenance,
        domains: Iterable[ResultDomain] = (),
        compiled_system: CompiledPhysicalSystem | None = None,
        run_id: str | None = None,
    ) -> None:
        frequencies = np.asarray(tuple(frequencies_hz), dtype=np.float64)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("A solved-system builder requires a non-empty one-dimensional frequency axis.")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
            raise ValueError("Solved-system frequencies must be finite and greater than zero.")
        if np.unique(frequencies).size != frequencies.size:
            raise ValueError("Solved-system frequencies must be unique.")

        excitation_ids = tuple(str(value) for value in excitation_ids)
        if len(excitation_ids) != len(set(excitation_ids)):
            raise ValueError("Solved-system excitation ids must be unique.")
        domain_items = tuple(domains)
        domain_map = {domain.id: domain.snapshot() for domain in domain_items}
        if len(domain_map) != len(domain_items):
            raise ValueError("Solved-system domain ids must be unique.")

        self.run_id = run_id or uuid.uuid4().hex
        self.provenance = provenance
        self.frequencies_hz = frequencies
        self.excitation_ids = excitation_ids
        self.domains = domain_map
        self.compiled_system = compiled_system
        self.completion_mask = np.zeros(frequencies.size, dtype=bool)
        self.diagnostics_by_frequency: list[dict[str, Any] | None] = [None] * frequencies.size
        self._quantities: dict[str, _QuantityBuffer] = {}
        self._finalized = False

    @property
    def solved_count(self) -> int:
        return int(np.count_nonzero(self.completion_mask))

    def add(self, result: SystemFrequencyResult) -> int:
        """Add or replace one streamed result and return its canonical index."""

        if self._finalized:
            raise RuntimeError("A finalized solved-system builder cannot accept more results.")
        validate_system_frequency_result(result)
        if tuple(result.excitation_port_ids) != self.excitation_ids:
            raise ValueError(
                "Solved-system excitation ids changed during the run: "
                f"expected {self.excitation_ids}, received {tuple(result.excitation_port_ids)}."
            )
        frequency_index = self._frequency_index(result.freq_hz)
        seen_ids: set[str] = set()
        pending: list[tuple[QuantityResult, _QuantityBuffer]] = []
        for quantity in result.quantities:
            seen_ids.add(quantity.id)
            buffer = self._quantities.get(quantity.id)
            if buffer is None:
                buffer = self._new_quantity_buffer(quantity)
            self._validate_quantity(buffer.specification, quantity)
            pending.append((quantity, buffer))

        for quantity, buffer in pending:
            self._quantities.setdefault(quantity.id, buffer)
            buffer.values[frequency_index] = np.asarray(quantity.values)
            buffer.available_frequency_mask[frequency_index] = True

        for quantity_id, buffer in self._quantities.items():
            if quantity_id not in seen_ids:
                buffer.available_frequency_mask[frequency_index] = False
        self.diagnostics_by_frequency[frequency_index] = copy.deepcopy(result.diagnostics)
        self.completion_mask[frequency_index] = True
        return frequency_index

    def snapshot(self, *, status: str) -> SolvedSystem:
        """Return an independent immutable copy while leaving the builder usable."""

        return self._solved_system(status=status, copy_values=True)

    def finalize(self, *, status: str) -> SolvedSystem:
        """Transfer the accumulated arrays into an immutable solved-system view.

        Finalization freezes the builder's existing NumPy buffers instead of
        duplicating every spatial field.  The builder becomes read-only after
        this call, so those shared buffers cannot subsequently be modified.
        """

        if self._finalized:
            raise RuntimeError("A solved-system builder may only be finalized once.")
        self._finalized = True
        return self._solved_system(status=status, copy_values=False)

    def _solved_system(self, *, status: str, copy_values: bool) -> SolvedSystem:
        quantities: dict[str, SolvedQuantity] = {}
        for quantity_id, buffer in self._quantities.items():
            specification = buffer.specification
            quantities[quantity_id] = SolvedQuantity(
                id=quantity_id,
                quantity=specification.quantity,
                unit=specification.unit,
                dimensions=("frequency", *specification.axes),
                values=_readonly_array(buffer.values, copy_values=copy_values),
                domain_id=specification.target_id,
                metadata=copy.deepcopy(specification.metadata),
                available_frequency_mask=_readonly_array(
                    buffer.available_frequency_mask,
                    copy_values=copy_values,
                ),
            )
        return SolvedSystem(
            run_id=self.run_id,
            provenance=self.provenance,
            frequencies_hz=_readonly_array(self.frequencies_hz, copy_values=copy_values),
            excitation_ids=self.excitation_ids,
            domains=dict(self.domains),
            quantities=quantities,
            completion_mask=_readonly_array(self.completion_mask, copy_values=copy_values),
            diagnostics_by_frequency=tuple(copy.deepcopy(self.diagnostics_by_frequency)),
            status=str(status),
            compiled_system=self.compiled_system,
        )

    def _frequency_index(self, frequency_hz: float) -> int:
        frequency = float(frequency_hz)
        distances = np.abs(self.frequencies_hz - frequency)
        index = int(np.argmin(distances))
        tolerance = max(1e-5, abs(frequency) * 2e-6)
        if distances[index] > tolerance:
            raise ValueError(
                f"Frequency {frequency:g} Hz is not on the requested solved-system axis; "
                f"nearest is {self.frequencies_hz[index]:g} Hz."
            )
        return index

    def _new_quantity_buffer(self, quantity: QuantityResult) -> _QuantityBuffer:
        values = np.asarray(quantity.values)
        if values.dtype.kind not in {"b", "i", "u", "f", "c"}:
            raise ValueError(f"Solved quantity {quantity.id!r} uses unsupported dtype {values.dtype}.")
        shape = (self.frequencies_hz.size, *values.shape)
        if values.dtype.kind in {"f", "c"}:
            fill_value: Any = np.nan
        else:
            fill_value = 0
        storage = np.full(shape, fill_value, dtype=values.dtype)
        return _QuantityBuffer(
            specification=QuantityResult(
                id=quantity.id,
                quantity=quantity.quantity,
                unit=quantity.unit,
                values=np.empty(values.shape, dtype=values.dtype),
                target_id=quantity.target_id,
                axes=tuple(quantity.axes),
                metadata=copy.deepcopy(quantity.metadata),
            ),
            values=storage,
            available_frequency_mask=np.zeros(self.frequencies_hz.size, dtype=bool),
        )

    @staticmethod
    def _validate_quantity(expected: QuantityResult, received: QuantityResult) -> None:
        expected_values = np.asarray(expected.values)
        received_values = np.asarray(received.values)
        expected_signature = (
            expected.quantity,
            expected.unit,
            expected.target_id,
            tuple(expected.axes),
            expected_values.shape,
            expected_values.dtype,
        )
        received_signature = (
            received.quantity,
            received.unit,
            received.target_id,
            tuple(received.axes),
            received_values.shape,
            received_values.dtype,
        )
        if expected_signature != received_signature:
            raise ValueError(
                f"Solved quantity {received.id!r} changed shape or semantics during the run: "
                f"expected {expected_signature}, received {received_signature}."
            )
