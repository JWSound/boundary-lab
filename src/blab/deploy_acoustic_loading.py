"""Qt-free active acoustic load recovery for Deploy's coupled sweeps."""

from __future__ import annotations

from typing import Any

import numpy as np

from blab.deploy_solve import DeployPackageData

ACOUSTIC_LOADING_KEYS = (
    "resistance",
    "reactance",
    "pressure_real_pa",
    "pressure_imag_pa",
)


def normalized_acoustic_loading(
    package: DeployPackageData, request: dict[str, Any], result: dict[str, Any], frequency_hz: float
) -> dict[str, list[float | None]]:
    """Recover opposing load as dimensionless R/X and complex RMS pressure (Pa).

    Undefined active impedances are explicit nulls, never zero-valued samples.
    Pressure is force/effective area, retains zero-velocity samples, and uses
    the same standard-audio (+iwt) convention as the impedance output.
    """
    descriptors = request.get("transducers", [])
    output = {key: [None] * len(descriptors) for key in ACOUSTIC_LOADING_KEYS}
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        return output
    physical = package.manifest.get("physical_system", {})
    components = [c for c in physical.get("components", []) if c.get("kind") == "electrodynamic_transducer"]
    if not components:
        return output
    areas = physical.get("metadata", {}).get("acoustic_impedance_normalization", {})
    medium = package.manifest.get("medium", {})
    rho_c = float(medium.get("density_kg_per_m3", np.nan)) * float(medium.get("sound_speed_m_per_s", np.nan))
    diagnostics = result.get("diagnostics", {})
    velocities = diagnostics.get("transducer_velocity", [])
    currents = diagnostics.get("transducer_current", [])
    offset = 0
    omega = 2 * np.pi * frequency_hz
    for raw_v, raw_i in zip(velocities, currents):
        v = np.asarray(raw_v["real"]) + 1j * np.asarray(raw_v["imag"])
        current = np.asarray(raw_i["real"]) + 1j * np.asarray(raw_i["imag"])
        if v.shape != (len(components),) or current.shape != v.shape:
            return output
        finite_velocity = np.abs(v[np.isfinite(v)])
        threshold = max(1e-12, float(np.max(finite_velocity, initial=0)) * 1e-8)
        for index, component in enumerate(components):
            target = offset + index
            if target >= len(descriptors):
                return output
            area = float(areas.get(component["id"], {}).get("effective_area_m2", np.nan))
            scale = rho_c * area
            if not np.isfinite(area) or area <= 0 or not np.isfinite(v[index]):
                continue
            parameters = component.get("parameters", {})
            try:
                zm = float(parameters["rms_n_s_per_m"]) + 1j * (
                    1 / (omega * float(parameters["cms_m_per_n"])) - omega * float(parameters["mmd_kg"])
                )
                load_force = float(parameters["bl_n_per_a"]) * current[index] - zm * v[index]
            except (KeyError, ValueError, TypeError, ZeroDivisionError):
                continue
            pressure = load_force / area
            if np.isfinite(pressure):
                output["pressure_real_pa"][target] = float(pressure.real)
                output["pressure_imag_pa"][target] = float(-pressure.imag)
            # Pressure is meaningful even with a stationary diaphragm. Only
            # impedance needs the velocity and medium-normalization guards.
            if not np.isfinite(scale) or scale <= 0 or abs(v[index]) <= threshold:
                continue
            impedance = load_force / v[index] / scale
            if np.isfinite(impedance):
                output["resistance"][target] = float(impedance.real)
                output["reactance"][target] = float(-impedance.imag)
        offset += len(components)
    return output
