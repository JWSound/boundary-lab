import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import blab.headless as headless_module
from blab.headless import (
    HeadlessProject,
    HeadlessResultWriter,
    load_headless_solve_spec,
    resolve_headless_backend,
)
from blab.physical_model import (
    CompiledPhysicalSystem,
    ExcitationPort,
    ExcitationPortKind,
    PhysicalSolveKind,
    PhysicalSystem,
)
from blab.project_cli import _build_arg_parser
from blab.solve_results import ResultDomain
from blab.system_contract import OutputRequest, QuantityResult, SystemFrequencyResult, SystemSolveRequest
from blab.system_solve import SystemUiSolveRequest, canonicalize_observation_result
from blab.ui.project_state import ProjectPreferencesState


def test_headless_request_parses_explicit_frequencies_and_probes(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frequencies_hz": [40, 50, 63],
                "include_project_observations": False,
                "excitation_port_ids": ["excitation:woofer"],
                "probes": [
                    {
                        "id": "farfield",
                        "coordinate_frame": "project",
                        "points_m": [[0, 0, 3], [1, 0, 3]],
                    }
                ],
                "retain": ["bem_boundary_traces"],
                "solver_options": {"validation_diagnostics": True},
            }
        ),
        encoding="utf-8",
    )

    spec = load_headless_solve_spec(request_path)

    assert spec.frequencies_hz == (40.0, 50.0, 63.0)
    assert spec.excitation_port_ids == ("excitation:woofer",)
    assert spec.probes[0].points_m.shape == (2, 3)
    assert spec.retain == ("bem_boundary_traces",)
    assert spec.include_project_observations is False


def test_headless_request_rejects_duplicate_frequencies(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"schema_version": 1, "frequencies_hz": [100, 100]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_headless_solve_spec(request_path)


def test_canonical_observation_result_preserves_complex_excitation_basis() -> None:
    prepared = _prepared_request()
    raw = SystemFrequencyResult(
        freq_hz=100.0,
        excitation_port_ids=("excitation:source",),
        quantities=(
            QuantityResult(
                id="ui:exterior-pressure",
                quantity="exterior_pressure",
                unit="Pa",
                values=np.asarray([[1 + 2j, 3 + 4j]], dtype=np.complex64),
                axes=("excitation", "observation"),
            ),
        ),
    )

    canonical = canonicalize_observation_result(prepared, raw)

    assert canonical.quantities[0].id == "acoustic:pressure:probe:test"
    np.testing.assert_array_equal(
        canonical.quantities[0].values,
        np.asarray([[1 + 2j, 3 + 4j]], dtype=np.complex64),
    )


def test_result_writer_persists_partial_complex_frequency_result(tmp_path: Path) -> None:
    project_file = tmp_path / "sample.blab.json"
    project_file.write_text('{"schema_version": 8}', encoding="utf-8")
    project = HeadlessProject(
        path=project_file,
        payload={"schema_version": 8},
        physical_system=PhysicalSystem("system:test", "Test", (), (), ()),
        preferences=ProjectPreferencesState(),
        symmetry="off",
        component_channel_by_id={},
    )
    prepared = _prepared_request()
    writer = HeadlessResultWriter(
        tmp_path / "run",
        project=project,
        prepared=prepared,
        backend_id="beat_cpu",
        public_request={"schema_version": 1},
    )
    result = canonicalize_observation_result(
        prepared,
        SystemFrequencyResult(
            freq_hz=100.0,
            excitation_port_ids=("excitation:source",),
            quantities=(
                QuantityResult(
                    id="ui:exterior-pressure",
                    quantity="exterior_pressure",
                    unit="Pa",
                    values=np.asarray([[1 + 2j, 3 + 4j]], dtype=np.complex64),
                    axes=("excitation", "observation"),
                ),
            ),
            diagnostics={"relative_residual": 1e-5},
        ),
    )

    writer.write_result(result)
    writer.finish(status="complete")

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / "run" / "frequencies" / "000000.json").read_text(encoding="utf-8"))
    arrays = np.load(tmp_path / "run" / "frequencies" / "000000.npz")
    assert manifest["status"] == "complete"
    assert manifest["completion_mask"] == [True]
    assert metadata["quantities"][0]["id"] == "acoustic:pressure:probe:test"
    assert arrays["q0000"].dtype == np.complex64
    np.testing.assert_array_equal(arrays["q0000"], np.asarray([[1 + 2j, 3 + 4j]], dtype=np.complex64))


def test_project_cli_exposes_validate_and_solve_commands() -> None:
    parser = _build_arg_parser("blab project")

    validate = parser.parse_args(["validate", "speaker.blab.json", "--json"])
    solve = parser.parse_args(["solve", "speaker.blab.json", "--events", "ndjson"])
    export = parser.parse_args(
        ["export-speaker", "speaker.blab.json", "--output", "speaker.blabsp", "--fidelity", "fixed"]
    )

    assert validate.project_command == "validate"
    assert validate.json is True
    assert validate.backend == "beat_auto"
    assert solve.project_command == "solve"
    assert solve.events == "ndjson"
    assert export.project_command == "export-speaker"
    assert export.output == Path("speaker.blabsp")
    assert export.fidelity == "fixed"


def test_auto_backend_prefers_functional_cuda(monkeypatch) -> None:
    monkeypatch.setattr(headless_module, "_command_available", lambda _command: True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(headless_module.subprocess, "run", fake_run)

    assert resolve_headless_backend("beat_auto", julia_executable="custom-julia") == "beat_cuda"
    assert calls[0][0][0] == "custom-julia"
    assert "CUDA.functional()" in calls[0][0][-1]


def test_auto_backend_falls_back_to_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(headless_module, "_command_available", lambda _command: True)
    monkeypatch.setattr(
        headless_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )

    assert resolve_headless_backend() == "beat_cpu"


def test_explicit_backend_does_not_probe_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        headless_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("explicit backend should not probe CUDA"),
    )

    assert resolve_headless_backend("beat_cpu") == "beat_cpu"
    assert resolve_headless_backend("beat_cuda") == "beat_cuda"
    assert resolve_headless_backend("beat_rocm") == "beat_rocm"
    assert resolve_headless_backend("beat_cpu_condensed") == "beat_cpu_condensed"


def _prepared_request() -> SystemUiSolveRequest:
    port = ExcitationPort(
        id="excitation:source",
        name="Source",
        component_id="component:source",
        kind=ExcitationPortKind.NORMAL_VELOCITY,
    )
    compiled = CompiledPhysicalSystem(
        id="system:test",
        name="Test",
        meshes=(),
        regions=(),
        boundaries=(),
        interfaces=(),
        components=(),
        excitation_ports=(port,),
        assumptions=(),
    )
    request = SystemSolveRequest(
        compiled_system=compiled,
        frequencies_hz=(100.0,),
        excitation_port_ids=(port.id,),
        outputs=(
            # The backend returns one compact array; canonicalization splits it.
            OutputRequest(
                id="ui:exterior-pressure",
                quantity="exterior_pressure",
                options={
                    "points_m": [[0, 0, 1], [0, 0, 2]],
                    "observation_domains": [
                        {
                            "id": "observation:probe:test",
                            "quantity_id": "acoustic:pressure:probe:test",
                            "offset": 0,
                            "count": 2,
                        }
                    ],
                },
            ),
        ),
    )
    return SystemUiSolveRequest(
        request=request,
        backend_id="beat_cpu",
        solve_kind=PhysicalSolveKind.EXTERIOR_BEM,
        polar_angle_deg=np.empty(0),
        excitation_channel_names=np.asarray(["main"]),
        excitation_component_names=np.asarray(["Source"]),
        horizontal_count=0,
        vertical_count=0,
        result_domains=(
            ResultDomain(
                id="observation:probe:test",
                kind="point_probe",
                dimensions=("observation",),
                coordinates={"points_m": np.asarray([[0, 0, 1], [0, 0, 2]], dtype=float)},
            ),
        ),
    )
