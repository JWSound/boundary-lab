"""Public positive-time data and explicit legacy ingestion boundaries."""

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from blab.deploy_solve import _load_deploy_package_data
from blab.phasor import LEGACY_PHASOR_CONVENTION as MINUS
from blab.phasor import SOLVER_PHASOR_CONVENTION as PLUS
from blab.phasor import convert_phasor
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


def npz(**arrays):
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


@pytest.mark.parametrize("convention", [MINUS, PLUS, None])
def test_old_and_new_package_traces_and_rom_normalize_once(tmp_path, convention):
    value = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    manifest = {
        "schema": "boundary-lab-speaker-package",
        "schema_version": 1,
        "frequencies_hz": [100.0],
        "files": {
            "fixed_sources": {"path": "fixed.npz", "geometry_mesh": "mesh.msh"},
            "coupled_model": {"path": "rom.npz", "representation": "parity_petrov_galerkin_rom"},
        },
    }
    if convention is not None:
        manifest["phasor_convention"] = convention
    names = ("k", "c", "d", "b", "e", "velocity", "current", "velocity_drive", "current_drive")
    members = {
        "manifest.json": json.dumps(manifest).encode(),
        "mesh.msh": b"geometry",
        "fixed.npz": npz(
            triangles=np.array([[0, 1, 2]]), points_m=np.eye(3), pressure_pa=value, normal_derivative_pa_per_m=2 * value
        ),
        "rom.npz": npz(frequencies_hz=np.array([100.0]), **{name: value for name in names}),
    }
    members["checksums.json"] = json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}).encode()
    path = tmp_path / "test.blabsp"
    with zipfile.ZipFile(path, "w") as archive:
        for key, payload in members.items():
            archive.writestr(key, payload)
    before = path.read_bytes()
    data = _load_deploy_package_data(path)
    expected = value if convention == PLUS else value.conj()
    np.testing.assert_array_equal(data.pressure, expected)
    np.testing.assert_array_equal(data.normal, 2 * expected)
    for name in names:
        np.testing.assert_array_equal(data.coupled_model["arrays"][name], expected)
    np.testing.assert_array_equal(data.coupled_model["arrays"]["frequencies_hz"], [100.0])
    np.testing.assert_array_equal(convert_phasor(data.pressure, data.manifest["phasor_convention"]), expected)
    assert path.read_bytes() == before


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
