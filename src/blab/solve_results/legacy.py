"""Compatibility projection from legacy exterior-BEM results."""

from __future__ import annotations

from typing import Any

import numpy as np

from blab.solve_results.model import (
    HORIZONTAL_POLAR_DOMAIN_ID,
    HORIZONTAL_POLAR_PRESSURE_ID,
    RADIATION_IMPEDANCE_ID,
    RADIATOR_DOMAIN_ID,
    SPHERE_DOMAIN_ID,
    SPHERE_PRESSURE_ID,
    VERTICAL_POLAR_DOMAIN_ID,
    VERTICAL_POLAR_PRESSURE_ID,
    ResultDomain,
)
from blab.solvers.base import FrequencyResult
from blab.system_contract import QuantityResult, SystemFrequencyResult


def legacy_excitation_ids(result: FrequencyResult) -> tuple[str, ...]:
    if result.channel_names is None:
        return ()
    names = tuple(str(value) for value in np.asarray(result.channel_names).tolist())
    if len(names) != len(set(names)):
        raise ValueError("Legacy result channel names must be unique to form an excitation basis.")
    return names


def legacy_result_domains(dataset: Any) -> tuple[ResultDomain, ...]:
    angles = np.asarray(dataset.polar_angle_deg, dtype=np.float32)
    domains = [
        ResultDomain(
            id=HORIZONTAL_POLAR_DOMAIN_ID,
            kind="polar_observation",
            dimensions=("observation",),
            coordinates={"angle_deg": angles},
            metadata={"plane": "horizontal"},
        ),
        ResultDomain(
            id=VERTICAL_POLAR_DOMAIN_ID,
            kind="polar_observation",
            dimensions=("observation",),
            coordinates={"angle_deg": angles},
            metadata={"plane": "vertical"},
        ),
        ResultDomain(
            id=RADIATOR_DOMAIN_ID,
            kind="component_collection",
            dimensions=("radiator",),
            coordinates={"name": np.asarray(dataset.radiator_names).astype(str)},
        ),
    ]
    if (
        dataset.sphere_r_distance_m is not None
        and dataset.sphere_theta_polar_rad is not None
        and dataset.sphere_phi_azimuth_rad is not None
    ):
        radius = np.asarray(dataset.sphere_r_distance_m, dtype=np.float32)
        theta = np.asarray(dataset.sphere_theta_polar_rad, dtype=np.float32)
        phi = np.asarray(dataset.sphere_phi_azimuth_rad, dtype=np.float32)
        sin_theta = np.sin(theta)
        points = np.column_stack(
            [
                radius * sin_theta * np.cos(phi),
                radius * sin_theta * np.sin(phi),
                radius * np.cos(theta),
            ]
        ).astype(np.float32, copy=False)
        domains.append(
            ResultDomain(
                id=SPHERE_DOMAIN_ID,
                kind="spherical_observation",
                dimensions=("observation",),
                coordinates={
                    "points_m": points,
                    "r_distance_m": radius,
                    "theta_polar_rad": theta,
                    "phi_azimuth_rad": phi,
                },
            )
        )
    return tuple(domains)


def legacy_result_to_system_result(result: FrequencyResult) -> SystemFrequencyResult:
    """Express one legacy plot result as canonical physical quantities."""

    excitation_ids = legacy_excitation_ids(result)
    quantities: list[QuantityResult] = []
    if result.horizontal_pressure is not None and result.vertical_pressure is not None:
        quantities.extend(
            [
                QuantityResult(
                    id=HORIZONTAL_POLAR_PRESSURE_ID,
                    quantity="exterior_pressure",
                    unit="Pa",
                    values=np.asarray(result.horizontal_pressure),
                    target_id=HORIZONTAL_POLAR_DOMAIN_ID,
                    axes=("excitation", "observation"),
                ),
                QuantityResult(
                    id=VERTICAL_POLAR_PRESSURE_ID,
                    quantity="exterior_pressure",
                    unit="Pa",
                    values=np.asarray(result.vertical_pressure),
                    target_id=VERTICAL_POLAR_DOMAIN_ID,
                    axes=("excitation", "observation"),
                ),
            ]
        )
        if result.sphere_pressure is not None:
            quantities.append(
                QuantityResult(
                    id=SPHERE_PRESSURE_ID,
                    quantity="exterior_pressure",
                    unit="Pa",
                    values=np.asarray(result.sphere_pressure),
                    target_id=SPHERE_DOMAIN_ID,
                    axes=("excitation", "observation"),
                )
            )
    else:
        quantities.extend(_legacy_spl_quantities(result))

    impedance = np.asarray(result.impedance)
    if impedance.ndim == 2 and impedance.shape[1] == 2:
        complex_impedance = impedance[:, 0] + 1j * impedance[:, 1]
        quantities.append(
            QuantityResult(
                id=RADIATION_IMPEDANCE_ID,
                quantity="radiation_impedance",
                unit="Pa*s/m^3",
                values=complex_impedance.astype(np.complex64, copy=False),
                target_id=RADIATOR_DOMAIN_ID,
                axes=("radiator",),
            )
        )

    diagnostics: dict[str, Any] = {
        "timings": {
            "assembly_s": float(result.timings.assembly_s),
            "solve_s": float(result.timings.solve_s),
            "field_s": float(result.timings.field_s),
        }
    }
    if result.diagnostics is not None:
        diagnostics["solver"] = {
            "convergence_info": result.diagnostics.convergence_info,
            "message": result.diagnostics.message,
        }
    return SystemFrequencyResult(
        freq_hz=float(result.freq_hz),
        quantities=tuple(quantities),
        excitation_port_ids=excitation_ids,
        diagnostics=diagnostics,
    )


def _legacy_spl_quantities(result: FrequencyResult) -> list[QuantityResult]:
    quantities = []
    for quantity_id, target_id, values, normalized in (
        (
            "acoustic:spl:horizontal-polar",
            HORIZONTAL_POLAR_DOMAIN_ID,
            result.horizontal_spl_db if result.horizontal_spl_db is not None else result.horizontal_spl_norm_db,
            False,
        ),
        (
            "acoustic:spl:vertical-polar",
            VERTICAL_POLAR_DOMAIN_ID,
            result.vertical_spl_db if result.vertical_spl_db is not None else result.vertical_spl_norm_db,
            False,
        ),
        (
            "acoustic:spl-normalized:horizontal-polar",
            HORIZONTAL_POLAR_DOMAIN_ID,
            result.horizontal_spl_norm_db,
            True,
        ),
        (
            "acoustic:spl-normalized:vertical-polar",
            VERTICAL_POLAR_DOMAIN_ID,
            result.vertical_spl_norm_db,
            True,
        ),
    ):
        quantities.append(
            QuantityResult(
                id=quantity_id,
                quantity="sound_pressure_level",
                unit="dB re 20 uPa",
                values=np.asarray(values, dtype=np.float32),
                target_id=target_id,
                axes=("observation",),
                metadata={"normalized": normalized},
            )
        )
    if result.sphere_spl_norm_db is not None:
        quantities.append(
            QuantityResult(
                id="acoustic:spl-normalized:sphere",
                quantity="sound_pressure_level",
                unit="dB re 20 uPa",
                values=np.asarray(result.sphere_spl_norm_db, dtype=np.float32),
                target_id=SPHERE_DOMAIN_ID,
                axes=("observation",),
                metadata={"normalized": True},
            )
        )
    return quantities
