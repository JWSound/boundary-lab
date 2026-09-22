"""Public positive-time data and explicit legacy ingestion boundaries."""

import numpy as np
import pytest

from blab.phasor import LEGACY_PHASOR_CONVENTION as MINUS
from blab.phasor import SOLVER_PHASOR_CONVENTION as PLUS
from blab.system_contract import (
    QuantityResult,
    SystemFrequencyResult,
    canonicalize_phasor_result,
    system_frequency_result_from_dict,
    system_frequency_result_to_dict,
)


def test_all_complex_quantities_normalize_once():
    values = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    result = SystemFrequencyResult(
        100.0,
        tuple(
            QuantityResult(name, name, "unit", values)
            for name in ("pressure", "normal_derivative", "velocity", "current", "impedance")
        ),
        diagnostics={"phasor_convention": MINUS},
    )
    normalized = canonicalize_phasor_result(result)
    for q in normalized.quantities:
        np.testing.assert_array_equal(q.values, values.conj())
        assert q.values.dtype == values.dtype
    restored = system_frequency_result_from_dict(system_frequency_result_to_dict(normalized))
    for q in restored.quantities:
        np.testing.assert_array_equal(q.values, values.conj())
    assert restored.diagnostics["phasor_convention"] == PLUS
    assert result.diagnostics["phasor_convention"] == MINUS
    with pytest.raises(ValueError, match="Unsupported phasor"):
        canonicalize_phasor_result(SystemFrequencyResult(100.0, (), diagnostics={"phasor_convention": "unknown"}))


def test_legacy_source_pressure_converts_but_standard_impedance_does_not():
    from blab.protocol import frequency_result_from_dict, frequency_result_to_dict

    raw = {
        "freq_hz": 100.0,
        "horizontal_spl_norm_db": [0.0],
        "vertical_spl_norm_db": [0.0],
        "impedance": [[6.0, 2.0]],
        "channel_names": ["drive"],
        "horizontal_pressure": {"real": [[1.0]], "imag": [[2.0]]},
    }
    result = frequency_result_from_dict(raw)
    np.testing.assert_array_equal(result.horizontal_pressure, [[1 - 2j]])
    np.testing.assert_array_equal(result.impedance, [[6.0, 2.0]])
    restored = frequency_result_from_dict(frequency_result_to_dict(result))
    np.testing.assert_array_equal(restored.horizontal_pressure, result.horizontal_pressure)


def test_builder_normalizes_legacy_provenance_and_arrays_together():
    from blab.solve_results.model import SolvedSystemBuilder, SolveProvenance

    builder = SolvedSystemBuilder(
        frequencies_hz=[100.0],
        excitation_ids=["drive"],
        provenance=SolveProvenance("beat_cpu", "exterior_bem", phasor_convention=MINUS),
    )
    builder.add(
        SystemFrequencyResult(
            100.0,
            (QuantityResult("p", "pressure", "Pa", np.array([1 + 2j]), axes=("excitation",)),),
            excitation_port_ids=("drive",),
        )
    )
    assert builder.provenance.phasor_convention == PLUS
    np.testing.assert_array_equal(builder.snapshot(status="completed").quantities["p"].values, [[1 - 2j]])
