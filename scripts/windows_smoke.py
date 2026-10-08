"""Installed-payload qualification, invoked explicitly by the release verifier."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--solve", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["QT_QPA_PLATFORM"] = "windows"
    import numpy as np
    import pyvista as pv
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from blab.ath import run_ath
    from blab.paths import APP_ROOT, USER_ROOT
    from blab.ui.main_window import MainWindow
    from blab.ui.mesh_preview import MeshPreview

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(args.output / "settings"))
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    window.grab().save(str(args.output / "window.png"))
    for preview in window.findChildren(MeshPreview):
        if preview.viewer is not None:
            preview.viewer.close()
    window.close()
    plot = pv.Plotter(off_screen=True, window_size=(320, 240))
    plot.add_mesh(pv.Sphere())
    plot.screenshot(str(args.output / "vtk.png"))
    plot.close()
    example = USER_ROOT / "examples/Ath_SingleSourceWaveguide/singlesourcewaveguide.blab.json"
    config = json.loads(example.read_text())["generator_documents"][0]["source"]["text"]
    console_python = sys.executable
    try:
        # The desktop launcher uses pythonw; test the exact meshing-child variant.
        sys.executable = str(Path(console_python).with_name("pythonw.exe"))
        generated = run_ath(
            ath_exe=APP_ROOT / "ath/ath202608.exe", config_text=config, run_root=args.output / "geometry", timeout_s=180
        )
    finally:
        sys.executable = console_python
    assert generated is not None
    report = {"gui": True, "vtk": True, "ath_gmsh": True, "backend": args.backend, "solves": []}
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
            for run in manifest["engine_runs"]:
                assert run["engine"]["version"] == "0.4.0"
                assert len(run["engine"]["source_sha256"]) == 64
            with np.load(output / "frequencies/000000.npz") as values:
                assert any(np.iscomplexobj(values[k]) for k in values.files), "Complex results lost"
                assert all(np.isfinite(values[k]).all() for k in values.files), "Nonfinite result"
            report["solves"].append(name)
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
