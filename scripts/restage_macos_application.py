"""Rebuild only application code into a new copy of a verified unsigned runtime."""

import argparse
import hashlib
import json
import shutil
import tomllib
from pathlib import Path

from build_macos_installer import verify_resources
from build_macos_runtime import inventory
from build_windows_runtime import ROOT, copy_tracked, digest, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.app.resolve()
    resources = source / "Contents/Resources"
    manifest = json.loads((resources / "runtime-manifest.json").read_text())
    if manifest.get("signing") or manifest["components"] != json.loads(
        (ROOT / "packaging/runtime-lock.json").read_text()
    ):
        raise ValueError("Require an unsigned app with current runtime pins")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    if manifest.get("platform") != "darwin" or manifest.get("arch") != "arm64" or manifest["version"] != version:
        raise ValueError("Require a macOS ARM64 app with the same application version; otherwise rebuild")
    verify_resources(resources, manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    app = output / source.name
    shutil.copytree(source, app, symlinks=True)
    resources = app / "Contents/Resources"
    python = resources / "runtime/python/bin/python3"
    wheels = output / "wheels"
    run(python, "-I", "-B", "-m", "pip", "wheel", "--no-deps", "--wheel-dir", wheels, ROOT)
    wheel = next(wheels.glob("boundary_lab-*.whl"))
    run(
        python,
        "-I",
        "-B",
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--no-compile",
        "--force-reinstall",
        wheel,
    )
    for folder in ("assets", "geometry_providers"):
        shutil.rmtree(resources / "resources" / folder)
        copy_tracked(folder, resources / "resources" / folder)
    shutil.rmtree(resources / "user-files/documentation")
    copy_tracked("docs", resources / "user-files/documentation")
    files, links = inventory(resources)
    manifest.update(files=files, symlinks=links)
    manifest["wheels"][wheel.name] = digest(wheel)
    manifest["runtime_id"] = (
        "mac-arm64-" + hashlib.sha256(json.dumps([files, links], sort_keys=True).encode()).hexdigest()[:16]
    )
    (resources / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(app)


if __name__ == "__main__":
    main()
