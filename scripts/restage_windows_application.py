"""Reuse a pinned payload's dependencies in a NEW directory while rebuilding the app."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path

from build_windows_runtime import ROOT, copy_tracked, digest, run, stage_gmsh_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if output.exists():
        parser.error("Output must be a new directory.")
    manifest = json.loads((base / "runtime-manifest.json").read_text())
    lock = json.loads((ROOT / "packaging/runtime-lock.json").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    if manifest["components"] != lock:
        parser.error("Runtime lock changed; perform a full build.")
    if lock["beat_requirement"] not in project["project"]["dependencies"]:
        parser.error("Runtime BEAT pin must match pyproject.toml.")
    # Verify dependency pins without importing from the base runtime.
    from importlib.metadata import distributions

    installed = {
        d.metadata["Name"].lower().replace("_", "-"): d.version
        for d in distributions(path=[str(base / "runtime/python/Lib/site-packages")])
    }
    for line in (ROOT / "packaging/requirements-win.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            if installed.get(name.lower()) != version:
                parser.error(f"Dependency pin changed: {name}; perform a full build.")
    output.mkdir(parents=True)
    # Copy only inventoried dependency files, checking each hash during the copy.
    files = {}
    for name, expected in manifest["files"].items():
        parts = Path(name).parts
        if parts[0] != "runtime" or "blab" in parts or any(p.startswith("boundary_lab-") for p in parts):
            continue
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with (base / name).open("rb") as source, target.open("wb") as destination:
            checksum = hashlib.sha256()
            while data := source.read(1024 * 1024):
                checksum.update(data)
                destination.write(data)
        if checksum.hexdigest() != expected:
            raise RuntimeError(f"Base dependency changed: {name}")
        files[name] = expected
    wheels = ROOT / "build" / (output.name + "-app-wheel")
    wheels.mkdir(parents=True, exist_ok=False)
    run(sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", wheels, ROOT)
    site = output / "runtime/python/Lib/site-packages"
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--no-compile",
        "--target",
        site,
        *wheels.glob("*.whl"),
    )
    base_wheels = ROOT / "build" / (base.name + "-wheels")
    stage_gmsh_library(base_wheels, site)
    for folder in ("assets", "ath", "geometry_providers"):
        copy_tracked(folder, output / "resources" / folder)
    copy_tracked("examples", output / "user-files/examples")
    copy_tracked("docs", output / "user-files/documentation")
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    compiler = Path(os.environ["SystemRoot"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    run(
        compiler,
        "/nologo",
        "/target:winexe",
        "/platform:x64",
        "/reference:System.Windows.Forms.dll",
        f"/win32icon:{ROOT / 'assets/256.ico'}",
        f"/out:{output / 'Boundary Lab.exe'}",
        ROOT / "packaging/Launcher.cs",
    )
    for path in output.rglob("*"):
        if path.is_file():
            name = path.relative_to(output).as_posix()
            if name not in files:
                files[name] = digest(path)
    manifest["files"] = files
    manifest["version"] = project["project"]["version"]
    manifest["wheels"] = {k: v for k, v in manifest["wheels"].items() if not k.startswith("boundary_lab-")}
    manifest["wheels"].update({p.name: digest(p) for p in wheels.glob("*.whl")})
    manifest["runtime_id"] = "win-x64-" + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    (output / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Restaged:", output)


if __name__ == "__main__":
    main()
