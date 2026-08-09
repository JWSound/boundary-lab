"""Canonical in-memory models for completed and streaming solve results."""

from blab.solve_results.derived import (
    electrical_input_impedance,
    phase_deg,
    pressure_spl_db,
    velocity_to_excursion,
)
from blab.solve_results.legacy import legacy_result_domains, legacy_result_to_system_result
from blab.solve_results.model import (
    DIAPHRAGM_VELOCITY_ID,
    HORIZONTAL_POLAR_DOMAIN_ID,
    HORIZONTAL_POLAR_PRESSURE_ID,
    RADIATION_IMPEDANCE_ID,
    RESULT_MODEL_VERSION,
    SPHERE_DOMAIN_ID,
    SPHERE_PRESSURE_ID,
    TRANSDUCER_DOMAIN_ID,
    VERTICAL_POLAR_DOMAIN_ID,
    VERTICAL_POLAR_PRESSURE_ID,
    VOICE_COIL_CURRENT_ID,
    ResultDomain,
    SolvedQuantity,
    SolvedSystem,
    SolvedSystemBuilder,
    SolveProvenance,
)

__all__ = [
    "RESULT_MODEL_VERSION",
    "DIAPHRAGM_VELOCITY_ID",
    "HORIZONTAL_POLAR_DOMAIN_ID",
    "HORIZONTAL_POLAR_PRESSURE_ID",
    "RADIATION_IMPEDANCE_ID",
    "SPHERE_DOMAIN_ID",
    "SPHERE_PRESSURE_ID",
    "TRANSDUCER_DOMAIN_ID",
    "VERTICAL_POLAR_DOMAIN_ID",
    "VERTICAL_POLAR_PRESSURE_ID",
    "VOICE_COIL_CURRENT_ID",
    "ResultDomain",
    "SolveProvenance",
    "SolvedQuantity",
    "SolvedSystem",
    "SolvedSystemBuilder",
    "electrical_input_impedance",
    "legacy_result_domains",
    "legacy_result_to_system_result",
    "phase_deg",
    "pressure_spl_db",
    "velocity_to_excursion",
]
