"""macOS GUI, native Gmsh, and CPU/Metal solves without external Ath or Wine."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--cpu-reference", type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["QT_QPA_PLATFORM"] = "cocoa"
    import gmsh
    import numpy as np
    import pyvista as pv
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from blab.paths import USER_ROOT
    from blab.ui.main_window import MainWindow
    from blab.ui.mesh_preview import MeshPreview

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(args.output / "settings"))
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    if args.backend == "metal":
        window.preferences.solve_backend = "beat_metal"
        failures = []
        window.backend_health.availability_failed.connect(failures.append)
        window.backend_health.check_selected_backend()
        deadline = time.monotonic() + 920
        while window.backend_health._probe is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert window.backend_health._metal_available, failures
        assert not failures, failures
    window.grab().save(str(args.output / "window.png"))
    for preview in window.findChildren(MeshPreview):
        if preview.viewer is not None:
            preview.viewer.close()
    window.close()
    plot = pv.Plotter(off_screen=True, window_size=(320, 240))
    plot.add_mesh(pv.Sphere())
    plot.screenshot(str(args.output / "vtk.png"))
    plot.close()
    gmsh.initialize()
    try:
        gmsh.model.add("packaging-probe")
        gmsh.model.occ.addSphere(0, 0, 0, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(str(args.output / "sphere.msh"))
    finally:
        gmsh.finalize()
    report = {
        "gui": True,
        "vtk": True,
        "gmsh": True,
        "external_ath_required": False,
        "backend": args.backend,
        "solves": [],
        "cpu_comparison_relative_errors": {},
    }
    if args.solve:
        fixture = args.output / "fixture"
        shutil.copytree(USER_ROOT / "examples/Simple_Sealed", fixture)
        original = fixture / "simple_sealed.blab.json"
        exterior = json.loads(original.read_text())
        system = exterior["physical_system"]
        system["meshes"] = [m for m in system["meshes"] if m["purpose"] == "bem_surface"]
        system["regions"] = [r for r in system["regions"] if r["kind"] == "unbounded_air"]
        system["boundaries"] = [b for b in system["boundaries"] if b["region_id"] == "region:exterior-air"]
        system["components"][0]["boundary_ids"] = ["boundary:region-exterior-air:woofer"]
        system["components"][0]["kind"] = "ideal_velocity_source"
        system["components"][0]["parameters"] = {"motion_profile": "uniform"}
        system["excitation_ports"][0]["kind"] = "normal_velocity"
        exterior["imported_meshes"] = [exterior["imported_meshes"][0]]
        exterior_path = fixture / "exterior.blab.json"
        exterior_path.write_text(json.dumps(exterior), encoding="utf-8")
        request = args.output / "request.json"
        request.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "frequencies_hz": [500.0],
                    "include_project_observations": False,
                    "retain": ["bem_boundary_traces"],
                    "probes": [{"id": "on_axis", "coordinate_frame": "project", "points_m": [[0, 0, 2]]}],
                }
            )
        )
        for name, project in (("exterior", exterior_path), ("coupled", original)):
            common = [str(project), "--backend", "beat_" + args.backend, "--request", str(request)]
            cli = [sys.executable, "-B", "-X", "utf8", "-m", "blab.cli", "project"]
            subprocess.run(cli + ["validate"] + common + ["--json"], check=True, timeout=180)
            output = args.output / name
            subprocess.run(
                cli + ["solve"] + common + ["--julia-threads", "2", "--output", str(output)], check=True, timeout=1800
            )
            manifest = json.loads((output / "manifest.json").read_text())
            assert all(manifest["completion_mask"]), manifest["status"]
            assert manifest["engine_runs"], "Missing engine provenance"
            assert manifest["backend_id"] == "beat_" + args.backend
            assert any(run["execution"].get("backend") == args.backend for run in manifest["engine_runs"]), (
                "Wrong backend provenance"
            )
            for run in manifest["engine_runs"]:
                assert run["engine"]["version"] == "0.4.0"
                assert len(run["engine"]["source_sha256"]) == 64
            with np.load(output / "frequencies/000000.npz") as values:
                assert any(np.iscomplexobj(values[k]) for k in values.files), "Complex results lost"
                assert all(np.isfinite(values[k]).all() for k in values.files), "Nonfinite result"
            if args.cpu_reference:
                with (
                    np.load(output / "frequencies/000000.npz") as actual,
                    np.load(args.cpu_reference / name / "frequencies/000000.npz") as reference,
                ):
                    assert set(actual.files) == set(reference.files)
                    for key in actual.files:
                        relative_error = np.linalg.norm(actual[key] - reference[key]) / max(
                            np.linalg.norm(reference[key]), 1e-12
                        )
                        report["cpu_comparison_relative_errors"][f"{name}/{key}"] = float(relative_error)
                        assert relative_error < 0.001, (name, key, relative_error)
            report["solves"].append(name)
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
