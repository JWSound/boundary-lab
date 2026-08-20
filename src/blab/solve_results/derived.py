"""Lazy derived quantities for canonical solved-system data."""

from __future__ import annotations

import copy

import numpy as np

from blab.physical_model import ExcitationPortKind
from blab.solve_results.model import (
    VOICE_COIL_CURRENT_ID,
    SolvedQuantity,
    SolvedSystem,
)

REFERENCE_PRESSURE_PA = 20.0e-6


def pressure_spl_db(pressure: np.ndarray) -> np.ndarray:
    """Return absolute SPL from a complex pressure phasor."""

    values = np.asarray(pressure)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (20.0 * np.log10(np.abs(values) / REFERENCE_PRESSURE_PA)).astype(np.float32, copy=False)


def phase_deg(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.angle(np.asarray(values))).astype(np.float32, copy=False)


def velocity_to_excursion(
    velocity: SolvedQuantity,
    frequencies_hz: np.ndarray,
    *,
    quantity_id: str = "mechanical:diaphragm-excursion",
) -> SolvedQuantity:
    """Integrate velocity phasors under Boundary Lab's exp(-i omega t) convention."""

    if not velocity.dimensions or velocity.dimensions[0] != "frequency":
        raise ValueError("Velocity quantity must use frequency as its first dimension.")
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.shape != (np.asarray(velocity.values).shape[0],):
        raise ValueError("Velocity and frequency dimensions do not match.")
    denominator_shape = (frequencies.size,) + (1,) * (np.asarray(velocity.values).ndim - 1)
    denominator = (-1j * 2.0 * np.pi * frequencies).reshape(denominator_shape)
    values = (np.asarray(velocity.values) / denominator).astype(np.asarray(velocity.values).dtype, copy=False)
    values.setflags(write=False)
    return SolvedQuantity(
        id=quantity_id,
        quantity="diaphragm_excursion",
        unit="m",
        dimensions=velocity.dimensions,
        values=values,
        domain_id=velocity.domain_id,
        metadata=copy.deepcopy(velocity.metadata),
        available_frequency_mask=velocity.available_frequency_mask,
    )


def electrical_input_impedance(
    solved: SolvedSystem,
    *,
    current_quantity_id: str = VOICE_COIL_CURRENT_ID,
) -> SolvedQuantity:
    """Derive each driven transducer's input impedance from its basis current."""

    if solved.compiled_system is None:
        raise ValueError("Electrical input impedance requires the compiled physical-system snapshot.")
    current = solved.quantity(current_quantity_id)
    if current.dimensions != ("frequency", "excitation", "transducer"):
        raise ValueError("Voice-coil current has unexpected dimensions.")
    component_ids = tuple(str(value) for value in current.metadata.get("component_ids", ()))
    component_index = {component_id: index for index, component_id in enumerate(component_ids)}
    ports_by_id = {port.id: port for port in solved.compiled_system.excitation_ports}
    selected: list[tuple[int, int, str]] = []
    for excitation_index, port_id in enumerate(solved.excitation_ids):
        port = ports_by_id.get(port_id)
        if port is None or port.kind != ExcitationPortKind.VOLTAGE:
            continue
        if port.component_id not in component_index:
            raise ValueError(f"Voltage port {port_id!r} has no matching transducer current column.")
        selected.append((excitation_index, component_index[port.component_id], port_id))
    if not selected:
        raise ValueError("The solved system contains no voltage excitation basis.")

    voltages = np.asarray(
        [_diagnostic_reference_voltage(diagnostics) for diagnostics in solved.diagnostics_by_frequency],
        dtype=float,
    )
    values = np.empty((solved.frequencies_hz.size, len(selected)), dtype=np.asarray(current.values).dtype)
    for output_index, (excitation_index, transducer_index, _port_id) in enumerate(selected):
        basis_current = np.asarray(current.values)[:, excitation_index, transducer_index]
        with np.errstate(divide="ignore", invalid="ignore"):
            values[:, output_index] = voltages / basis_current
    values.setflags(write=False)
    mask = np.asarray(current.available_frequency_mask).copy()
    mask &= np.isfinite(voltages)
    mask.setflags(write=False)
    return SolvedQuantity(
        id="electrical:input-impedance",
        quantity="electrical_input_impedance",
        unit="ohm",
        dimensions=("frequency", "excitation"),
        values=values,
        metadata={"excitation_port_ids": [item[2] for item in selected]},
        available_frequency_mask=mask,
    )


def _diagnostic_reference_voltage(diagnostics: dict | None) -> float:
    if not diagnostics:
        return np.nan
    try:
        value = float(diagnostics["transducer_reference_voltage_v"])
    except (KeyError, TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) and value > 0.0 else np.nan
