"""Build an Apple Silicon CPU app, excluding external Ath and Wine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import plistlib
import shutil
import sys
import tarfile
import tomllib
from pathlib import Path

from build_windows_runtime import ROOT, copy_tracked, digest, download, run


def extract(archive, destination):
    def safe(member, directory):
        if Path(member.name).name.startswith("._") or "__MACOSX" in Path(member.name).parts:
            return None
        return tarfile.data_filter(member, directory)

    with tarfile.open(archive) as source:
        source.extractall(destination, filter=safe)


def inventory(root):
    files, links = {}, {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            if not path.resolve().is_relative_to(root.resolve()):
                raise RuntimeError(f"Escaping link: {name}")
            links[name] = os.readlink(path)
        elif path.is_file() and name != "runtime-manifest.json":
            files[name] = digest(path)
    return files, links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin" or platform.machine() != "arm64" or sys.version_info < (3, 12):
        parser.error("Use native Apple Silicon Python 3.12+.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((ROOT / "packaging/runtime-lock.json").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    if lock["beat_requirement"] not in project["dependencies"]:
        raise ValueError("Application and runtime engine pins disagree")
    app = output / "Boundary Lab.app"
    contents = app / "Contents"
    resources = contents / "Resources"
    runtime = resources / "runtime"
    runtime.mkdir(parents=True)
    work = output / "staging"
    work.mkdir()
    target = lock["macos_arm64"]
    cache = ROOT / "build/downloads"
    extract(download(target["python"], cache), runtime)
    extract(download(target["julia"], cache), work)
    shutil.move(work / f"julia-{target['julia']['version']}", runtime / "julia")
    python = runtime / "python/bin/python3"
    wheels = work / "wheels"
    wheels.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("JULIA", "PYTHON", "BLAB_", "DYLD_"))}
    run(python, "-I", "-m", "pip", "wheel", "--no-deps", "--wheel-dir", wheels, ROOT, lock["beat_requirement"], env=env)
    run(
        python,
        "-I",
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--dest",
        wheels,
        "-r",
        ROOT / "packaging/requirements-macos.txt",
        env=env,
    )
    run(
        python,
        "-I",
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--no-compile",
        *sorted(wheels.glob("*.whl")),
        env=env,
    )
    run(python, "-I", "-m", "pip", "check", env=env)
    site = runtime / "python/lib/python3.13/site-packages"
    env.update(
        JULIA_DEPOT_PATH=str(work / "depot"),
        JULIA_LOAD_PATH="@:@stdlib",
        JULIA_PKG_PRECOMPILE_AUTO="0",
        JULIA_CPU_TARGET="generic",
    )
    run(
        runtime / "julia/bin/julia",
        "--startup-file=no",
        ROOT / "packaging/stage_depot.jl",
        runtime / "julia-depot",
        "0.0",
        site / "beat_engine/julia_local",
        env=env,
    )
    # Deliberately whitelist resources: no ath/ executable or Wine distribution.
    for folder in ("assets", "geometry_providers"):
        copy_tracked(folder, resources / "resources" / folder)
    copy_tracked("examples", resources / "user-files/examples")
    copy_tracked("docs", resources / "user-files/documentation")
    shutil.copy2(ROOT / "LICENSE", resources / "LICENSE")
    executables = contents / "MacOS"
    executables.mkdir()
    run(
        "xcrun",
        "clang",
        "-arch",
        "arm64",
        "-mmacosx-version-min=14.0",
        "-framework",
        "Cocoa",
        ROOT / "packaging/Launcher.m",
        "-o",
        executables / "Boundary Lab",
    )
    iconset = work / "BoundaryLab.iconset"
    iconset.mkdir()
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            name = f"icon_{size}x{size}" + ("@2x" if scale == 2 else "") + ".png"
            run(
                "sips",
                "-s",
                "format",
                "png",
                "-z",
                str(size * scale),
                str(size * scale),
                ROOT / "assets/256.ico",
                "--out",
                iconset / name,
            )
    run("iconutil", "-c", "icns", iconset, "-o", resources / "BoundaryLab.icns")
    metadata = dict(
        CFBundleIdentifier="org.boundarylab.desktop",
        CFBundleName="Boundary Lab",
        CFBundleDisplayName="Boundary Lab",
        CFBundleExecutable="Boundary Lab",
        CFBundlePackageType="APPL",
        CFBundleShortVersionString=project["version"],
        CFBundleVersion="1",
        CFBundleIconFile="BoundaryLab.icns",
        LSMinimumSystemVersion="14.0",
        NSHighResolutionCapable=True,
    )
    (contents / "Info.plist").write_bytes(plistlib.dumps(metadata))
    files, links = inventory(resources)
    identity = hashlib.sha256(json.dumps([files, links], sort_keys=True).encode()).hexdigest()[:16]
    manifest = dict(
        schema_version=1,
        platform="darwin",
        arch="arm64",
        runtime_id="mac-arm64-" + identity,
        backends=["cpu"],
        version=project["version"],
        components=lock,
        files=files,
        symlinks=links,
        wheels={p.name: digest(p) for p in wheels.glob("*.whl")},
    )
    (resources / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {app}", flush=True)


if __name__ == "__main__":
    main()
