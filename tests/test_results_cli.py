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


def test_catalog_uses_saved_drives_and_reordered_electrical_rows(run, capsys):
    from blab.solve_results.model import VOICE_COIL_CURRENT_ID

    with np.load(run / "sample.npz") as archive:
        arrays = dict(archive)
    arrays["current"] = np.array([[0.1], [0.5]], dtype=complex)
    np.savez(run / "sample.npz", **arrays)
    path = run / "sample.json"
    meta = json.loads(path.read_text())
    meta["quantities"].append(
        dict(
            id=VOICE_COIL_CURRENT_ID,
            axes=["excitation", "transducer"],
            shape=[2, 1],
            unit="A",
            quantity="voice_coil_current",
            target_id="drivers",
            key="current",
            metadata={"component_ids": ["one"]},
        )
    )
    path.write_text(json.dumps(meta))
    for metric, field, expected in [
        ("electrical-impedance", "magnitude", 2.83 / 0.5),
        ("real-input-power", "real_power", (-5.66) * (-2 * 0.5 + 0.1)),
    ]:
        main(["results", "report", str(run), metric])
        data = json.loads(capsys.readouterr().out)
        assert data["trace_names"][0] == "woofer"
        assert data["quantities"][field]["values"][0][0] == pytest.approx(expected)
    main(["results", "report", str(run), "on-axis-frequency-response"])
    axis = json.loads(capsys.readouterr().out)
    assert axis["trace_names"] == ["Sum", "woofer", "port"]
    assert axis["quantities"]["spl_db"]["values"][1][0] == pytest.approx(20 * np.log10(2 / 20e-6))
    with pytest.raises(SystemExit):
        main(["results", "report", str(run), "group-delay"])
    assert "three distinct" in capsys.readouterr().err


def test_combined_selects_only_external_points_for_bem(run, monkeypatch, capsys):
    pytest.importorskip("pyvista")
    from blab.solve_results.model import BEM_BOUNDARY_DOMAIN_ID, BEM_BOUNDARY_NEUMANN_ID, BEM_BOUNDARY_PRESSURE_ID

    def update(path, change):
        payload = json.loads(path.read_text())
        change(payload)
        path.write_text(json.dumps(payload))

    with np.load(run / "domains.npz") as archive:
        arrays = dict(archive)
    arrays["faces"] = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    np.savez(run / "domains.npz", **arrays)
    update(
        run / "domains.json",
        lambda p: p["domains"].append(
            dict(
                id=BEM_BOUNDARY_DOMAIN_ID,
                coordinates={"points_m": "points"},
                topology={"triangles": "faces"},
                metadata={"symmetry": "off"},
            )
        ),
    )
    update(
        run / "system.json",
        lambda p: p["regions"].append(dict(kind="unbounded_air", density_kg_per_m3=2, sound_speed_m_per_s=345)),
    )
    with np.load(run / "sample.npz") as archive:
        arrays = dict(archive)
    quantities = []
    for key, axis in [(BEM_BOUNDARY_PRESSURE_ID, "bem_node"), (BEM_BOUNDARY_NEUMANN_ID, "bem_face")]:
        arrays[key] = np.ones((2, 4), dtype=complex)
        quantities.append(
            dict(id=key, key=key, axes=["excitation", axis], shape=[2, 4], unit="Pa", quantity=key, metadata={})
        )
    np.savez(run / "sample.npz", **arrays)
    update(run / "sample.json", lambda p: p["quantities"].extend(quantities))

    def field(request, **kwargs):
        assert kwargs["backend_id"] == "beat_cpu"
        assert request.sound_speed_m_per_s == 345
        assert (request.points_m.sum(axis=1) > 1.0).all()
        return (
            np.tile([3j, 1, 0, 0], (len(request.points_m), 1))
            if request.pressure_gradient
            else np.full(len(request.points_m), 3j)
        )

    monkeypatch.setattr("blab.solve_results.plane_query.evaluate_bem_field", field)
    definition = run / "plane.json"
    definition.write_text(
        json.dumps(
            dict(id="s", type="combined", center_m=[0.9, 0.1, 0.1], width_m=0.6, height_m=0.01, resolution_m=0.1)
        )
    )
    args = ["results", "plane", str(run), "--plane-file", str(definition), "--frequency", "100", "--excitation", "a"]
    main(args)
    fields = json.loads(capsys.readouterr().out)["fields"]
    interior = np.array(fields["interior_mask"])
    exterior = np.array(fields["exterior_mask"])
    assert interior.any() and exterior.any()
    assert not (interior & exterior).any()
    assert np.array(fields["pressure_pa"]["imag"])[exterior].tolist() == [3] * exterior.sum()
    velocity = np.asarray(fields["particle_velocity_m_per_s"]["imag"], dtype=float)
    assert velocity[exterior, 0] == pytest.approx(1 / (400 * np.pi))
    main([*args, "--pressure-only"])
    fields = json.loads(capsys.readouterr().out)["fields"]
    assert np.isnan(np.asarray(fields["particle_velocity_m_per_s"]["imag"], dtype=float)[exterior]).all()


def test_directivity_reports_share_desktop_basis_and_return_labelled_axes(run, capsys):
    from blab.solve_results.model import (
        SPHERE_DOMAIN_ID,
        SPHERE_PRESSURE_ID,
        VERTICAL_POLAR_DOMAIN_ID,
        VERTICAL_POLAR_PRESSURE_ID,
    )

    # One nonzero source gives an omnidirectional response on both cuts and a tetrahedral sphere.
    with np.load(run / "sample.npz") as archive:
        arrays = dict(archive)
    meta = json.loads((run / "sample.json").read_text())
    h = meta["quantities"][0]

    def pressure(key, count):
        arrays[key] = np.vstack((np.zeros(count), np.ones(count))).astype(complex)

    count = 37
    pressure(h["key"], count)
    h["shape"] = [2, count]
    for qid, domain, n in [
        (VERTICAL_POLAR_PRESSURE_ID, VERTICAL_POLAR_DOMAIN_ID, count),
        (SPHERE_PRESSURE_ID, SPHERE_DOMAIN_ID, 4),
    ]:
        pressure(qid, n)
        meta["quantities"].append(h | {"id": qid, "key": qid, "target_id": domain, "shape": [2, n]})
    np.savez(run / "sample.npz", **arrays)
    with np.load(run / "domains.npz") as archive:
        coords = dict(archive)
    dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    coords.update(
        angles=np.linspace(-180, 180, count),
        radius=np.ones(4),
        theta=np.arccos(dirs[:, 2]),
        phi=np.arctan2(dirs[:, 1], dirs[:, 0]),
    )
    np.savez(run / "domains.npz", **coords)
    domains = json.loads((run / "domains.json").read_text())
    domains["domains"].extend(
        [
            {"id": VERTICAL_POLAR_DOMAIN_ID, "coordinates": {"angle_deg": "angles"}, "topology": {}},
            {
                "id": SPHERE_DOMAIN_ID,
                "coordinates": {"r_distance_m": "radius", "theta_polar_rad": "theta", "phi_azimuth_rad": "phi"},
                "topology": {},
            },
        ]
    )
    (run / "domains.json").write_text(json.dumps(domains))
    manifest = json.loads((run / "manifest.json").read_text())
    frequencies = [100, 200, 300]
    manifest.update(frequencies_hz=frequencies, completion_mask=[True] * 3, results=[])
    for hz in frequencies:
        filename = f"{hz}.json"
        (run / filename).write_text(json.dumps(meta | {"freq_hz": hz}))
        manifest["results"].append({"freq_hz": hz, "metadata_file": filename, "arrays_file": "sample.npz"})
    (run / "manifest.json").write_text(json.dumps(manifest))
    for metric, extra in [
        ("group-delay", []),
        ("spinorama", ["--spherical"]),
        ("horizontal-isobar", ["--frequency", "100"]),
        ("vertical-isobar", ["--frequency", "100"]),
        ("balloon", ["--frequency", "100"]),
    ]:
        args = ["results", "report", str(run), metric, *extra]
        main(args)
        data = json.loads(capsys.readouterr().out)
        if metric == "group-delay":
            assert data["quantities"]["group_delay"]["unit"] == "ms"
        elif metric == "spinorama":
            assert data["trace_names"][-1] == "Spherical DI"
            np.testing.assert_allclose(data["quantities"]["level"]["values"], 0, atol=1e-5)
        else:
            assert data["axes"] == (["frequency", "direction"] if metric == "balloon" else ["angle", "frequency"])
            np.testing.assert_allclose(data["quantities"]["normalized_spl"]["values"], 0, atol=1e-5)
        main([*args, "--format", "svg"])
        assert ElementTree.fromstring(capsys.readouterr().out).tag.endswith("}svg")
