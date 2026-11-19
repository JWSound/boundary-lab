import csv
import json
from io import StringIO
from xml.etree import ElementTree

import numpy as np
import pytest

from blab.cli import main
from blab.solve_results.model import (
    DIAPHRAGM_VELOCITY_ID,
    FEM_NODAL_PRESSURE_ID,
    FEM_VOLUME_DOMAIN_ID,
    HORIZONTAL_POLAR_DOMAIN_ID,
    HORIZONTAL_POLAR_PRESSURE_ID,
    RADIATION_IMPEDANCE_ID,
)


@pytest.fixture
def run(tmp_path):
    def save(path, data):
        (tmp_path / path).write_text(json.dumps(data), encoding="utf-8")

    ids = ["a", "b"]
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    # Deliberately reverse frequency-result excitation order relative to the manifest.
    entries = [
        (
            HORIZONTAL_POLAR_PRESSURE_ID,
            "exterior_pressure",
            "Pa",
            ["excitation", "observation"],
            HORIZONTAL_POLAR_DOMAIN_ID,
            [[2, 2], [1, 1]],
        ),
        (DIAPHRAGM_VELOCITY_ID, "diaphragm_velocity", "m/s", ["excitation", "transducer"], "drivers", [[0.25], [1]]),
        (
            FEM_NODAL_PRESSURE_ID,
            "fem_nodal_pressure",
            "Pa",
            ["excitation", "fem_node"],
            FEM_VOLUME_DOMAIN_ID,
            [np.ones(4), 1 + points[:, 0]],
        ),
        (RADIATION_IMPEDANCE_ID, "radiation_impedance", "N*s/m", ["radiator"], "radiators", [12 + 6j]),
    ]
    metadata, arrays = [], {}
    for i, (qid, kind, unit, axes, domain, vals) in enumerate(entries):
        key = str(i)
        arrays[key] = np.array(vals, dtype=complex)
        metadata.append(
            dict(
                id=qid,
                quantity=kind,
                unit=unit,
                axes=axes,
                target_id=domain,
                key=key,
                shape=arrays[key].shape,
                dtype=str(arrays[key].dtype),
                metadata={"component_ids": ["one"]} if qid == DIAPHRAGM_VELOCITY_ID else {},
            )
        )
    np.savez(tmp_path / "sample.npz", **arrays)
    save(
        "sample.json",
        dict(
            freq_hz=100,
            excitation_port_ids=ids[::-1],
            quantities=metadata,
            diagnostics={"transducer_reference_voltage_v": 2.83},
        ),
    )
    save(
        "manifest.json",
        dict(
            schema="boundary-lab-headless-result",
            schema_version=2,
            status="interrupted",
            backend_id="beat_cpu",
            solve_kind="exterior_bem",
            phasor_convention="exp(+i omega t)",
            frequencies_hz=[80, 100],
            completion_mask=[False, True],
            results=[None, dict(freq_hz=100, metadata_file="sample.json", arrays_file="sample.npz")],
            solver_options={"symmetry": "off"},
            project_sha256="test",
            meshes=[],
            excitation_port_ids=ids,
            domains_metadata_file="domains.json",
            domains_file="domains.npz",
            compiled_system_file="system.json",
            project_file="project.json",
        ),
    )
    save(
        "project.json",
        dict(
            schema_version=10,
            component_channel_by_id={"one": "woofer", "two": "port"},
            channel_config_by_name={"woofer": {"voltage_v": 5.66, "polarity": -1}, "port": {"voltage_v": 0}},
            project_preferences={"normalized_channel_correction": False},
            max_spl_limits_by_channel={"woofer": {"xmax_mm": 1, "pmax_w": 100}},
            observation_planes=[dict(id="slice", center_m=[0.5, 0.5, 0.1], width_m=1, height_m=1, resolution_m=0.5)],
        ),
    )
    save(
        "system.json",
        dict(
            excitation_ports=[
                dict(id="a", kind="voltage", component_id="one"),
                dict(id="b", kind="normal_velocity", component_id="two"),
            ],
            components=[
                dict(id="one", kind="electrodynamic_transducer", parameters={"re_ohm": 6}),
                dict(id="two", kind="ideal_velocity_source", parameters={}),
            ],
            regions=[dict(kind="bounded_air", density_kg_per_m3=2, sound_speed_m_per_s=3)],
            metadata={"acoustic_impedance_normalization": {"two": {"effective_area_m2": 2}}},
        ),
    )
    np.savez(
        tmp_path / "domains.npz",
        angles=[-5, 5],
        points=points,
        cells=[[0, 1, 2, 3]],
        node_regions=[0] * 4,
        cell_regions=[0],
    )
    save(
        "domains.json",
        {
            "domains": [
                {"id": HORIZONTAL_POLAR_DOMAIN_ID, "coordinates": {"angle_deg": "angles"}, "topology": {}},
                {
                    "id": FEM_VOLUME_DOMAIN_ID,
                    "coordinates": {"points_m": "points", "region_index": "node_regions"},
                    "topology": {"tetrahedra": "cells", "region_index": "cell_regions"},
                    "metadata": {"density_kg_per_m3": [2]},
                },
            ]
        },
    )
    return tmp_path


def test_read_only_query_and_synthesis(run, capsys):
    def call(command, *args):
        main(["results", command, str(run), *args])
        return json.loads(capsys.readouterr().out)

    summary = call("inspect")
    assert summary["completion_mask"] == [False, True]
    assert summary["quantities"][DIAPHRAGM_VELOCITY_ID]["frequencies_hz"] == [100]
    query = (HORIZONTAL_POLAR_PRESSURE_ID, "--index", "observation", "1")
    assert call("query", *query, "--excitation", "a")["samples"][0]["values"]["real"] == 1
    assert call("query", *query, "--response", "system")["samples"][0]["values"]["real"] == pytest.approx(0)
    assert call("query", *query, "--response", "channel:woofer")["samples"][0]["values"]["real"] == pytest.approx(-2)
    main(["results", "query", str(run), *query, "--format", "csv"])
    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert [(r["excitation_port_id"], float(r["real"])) for r in rows] == [("b", 2), ("a", 1)]
    peak = call("query", DIAPHRAGM_VELOCITY_ID, "--response", "system", "--view", "peak-excursion")
    assert peak["samples"][0]["values"] == pytest.approx([1.75 * np.sqrt(2) / (2 * np.pi * 100) * 1000])
    with pytest.raises(SystemExit):
        call("query", *query, "--frequency", "80")
    assert "No committed sample" in capsys.readouterr().err


def test_desktop_metrics_and_interior_field(run, capsys):
    def call(command, *args):
        main(["results", command, str(run), *args])
        return json.loads(capsys.readouterr().out)

    sample = call("report", "normalized-impedance")["samples"][0]
    assert sample["values"] == {"real": [1.0], "imag": [0.5]}
    sample = call("report", "maximum-spl")["samples"][0]
    gain = 2 * np.pi * 100 / (1000 * np.sqrt(2))
    assert sample["values"] == pytest.approx([20 * np.log10(gain / 20e-6)])
    plane = call("plane", "--plane-id", "slice", "--frequency", "100", "--excitation", "a")
    fields = plane["fields"]
    assert plane["shape_yx"] == [3, 3]
    assert fields["pressure_pa"]["real"][1] == pytest.approx(1.5)
    assert fields["particle_velocity_m_per_s"]["imag"][1][0] == pytest.approx(1 / (400 * np.pi))
    assert fields["reactive_in_plane_intensity_w_per_m2"][1][0] == pytest.approx(-1.5 / (400 * np.pi))
    assert fields["pressure_pa"]["real"][-1] is None
    for command in (
        ["report", str(run), "maximum-spl"],
        ["plane", str(run), "--plane-id", "slice", "--frequency", "100"],
    ):
        main(["results", *command, "--format", "svg"])
        assert ElementTree.fromstring(capsys.readouterr().out).tag.endswith("}svg")


def test_saved_flat_target_is_explicit_and_basis_subset_is_checked(run, capsys):
    project_file = run / "project.json"
    project = json.loads(project_file.read_text())
    project["project_preferences"]["normalized_channel_correction"] = True
    project_file.write_text(json.dumps(project))
    args = ["results", "query", str(run), HORIZONTAL_POLAR_PRESSURE_ID, "--response", "system"]
    main(args)
    # Port basis (2 Pa) is normalized to 1 Pa; the inverted woofer still contributes -2 Pa.
    assert json.loads(capsys.readouterr().out)["samples"][0]["values"]["real"] == pytest.approx([-1, -1])
    main([*args, "--uncorrected"])
    assert json.loads(capsys.readouterr().out)["samples"][0]["values"]["real"] == pytest.approx([0, 0])
    sample_file = run / "sample.json"
    sample = json.loads(sample_file.read_text())
    sample["quantities"] = [q for q in sample["quantities"] if q["id"] != HORIZONTAL_POLAR_PRESSURE_ID]
    sample_file.write_text(json.dumps(sample))
    args[3] = DIAPHRAGM_VELOCITY_ID
    with pytest.raises(SystemExit):
        main(args)
    assert "requires horizontal polar pressure" in capsys.readouterr().err
    system_file = run / "system.json"
    system = json.loads(system_file.read_text())
    system["excitation_ports"].append(dict(id="missing", kind="voltage", component_id="one"))
    system_file.write_text(json.dumps(system))
    with pytest.raises(SystemExit):
        main([*args, "--uncorrected"])
    assert "lacks excitation ports" in capsys.readouterr().err
