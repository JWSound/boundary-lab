"""Stage a self-contained Windows x64 runtime from pinned official archives/wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(item: dict, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / item["url"].rsplit("/", 1)[1]
    if not target.exists():
        partial = target.with_suffix(".partial")
        print("Downloading", item["url"], flush=True)
        with urllib.request.urlopen(item["url"], timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(target)
    if digest(target) != item["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {target}")
    return target


def run(*args, **kwargs):
    print("Running", *map(str, args), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def copy_tracked(folder: str, destination: Path) -> None:
    files = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", folder], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    for filename in filter(None, files):
        source = ROOT / filename
        target = destination / source.relative_to(ROOT / folder)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def validate_cuda_inventory(files) -> None:
    names = {Path(name).name.lower() for name in files}
    required = ("cublas64_", "cublaslt64_", "cudart64_", "cusolver64_", "cusparse64_", "cudss64_", "nvjitlink")
    missing = [prefix for prefix in required if not any(n.startswith(prefix) and n.endswith(".dll") for n in names)]
    if "ptxas.exe" not in names:
        missing.append("ptxas.exe")
    if missing:
        raise RuntimeError("Incomplete CUDA bundle: " + ", ".join(missing))


def stage_gmsh_library(wheels: Path, site: Path) -> None:
    # pip --target can omit the wheel's .data/data/lib DLL. Keep it beside gmsh.py,
    # which is one of upstream's supported lookup locations.
    with zipfile.ZipFile(next(wheels.glob("gmsh-*.whl"))) as archive:
        libraries = [name for name in archive.namelist() if name.endswith(".dll")]
        if len(libraries) != 1:
            raise RuntimeError(f"Unexpected Gmsh native library inventory: {libraries}")
        with archive.open(libraries[0]) as source, (site / Path(libraries[0]).name).open("wb") as target:
            shutil.copyfileobj(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu", help="CUDA includes CPU fallback")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--download-cache", type=Path, default=ROOT / "build/downloads")
    args = parser.parse_args()
    if sys.platform != "win32" or sys.version_info[:2] != (3, 13):
        parser.error("Use Python 3.13 x64 on Windows.")
    output = (args.output or ROOT / f"build/windows-{args.backend}").resolve()
    if output.exists():
        parser.error("Use a new output directory; completed/failed builds are retained for diagnosis.")
    lock = json.loads((ROOT / "packaging/runtime-lock.json").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    if lock["beat_requirement"] not in project["project"]["dependencies"]:
        parser.error("Runtime BEAT pin must match pyproject.toml.")
    output.mkdir(parents=True)
    runtime = output / "runtime"
    python = runtime / "python"
    python.mkdir(parents=True)
    with zipfile.ZipFile(download(lock["python"], args.download_cache)) as archive:
        archive.extractall(python)
    unpack = output / "julia-unpack"
    with zipfile.ZipFile(download(lock["julia"], args.download_cache)) as archive:
        archive.extractall(unpack)
    shutil.move(str(unpack / f"julia-{lock['julia']['version']}"), runtime / "julia")
    unpack.rmdir()
    wheels = ROOT / "build" / (output.name + "-wheels")
    wheels.mkdir(parents=True, exist_ok=False)
    run(sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", wheels, ROOT, lock["beat_requirement"])
    run(
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--dest",
        wheels,
        "-r",
        ROOT / "packaging/requirements-win.txt",
    )
    site = python / "Lib/site-packages"
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
        *sorted(wheels.glob("*.whl")),
    )
    stage_gmsh_library(wheels, site)
    (python / "python313._pth").write_text("python313.zip\n.\nLib/site-packages\nimport site\n", encoding="utf-8")
    run(python / "python.exe", "-I", "-B", "-c", "import gmsh; gmsh.initialize(); gmsh.finalize()")
    backends = ["cpu", "cuda"] if args.backend == "cuda" else ["cpu"]
    engine = site / "beat_engine"
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith(("JULIA", "BLAB_", "PYTHON"))}
    # A fresh build depot, never the developer's ~/.julia. Retain outside the payload.
    environment.update(
        JULIA_DEPOT_PATH=str(ROOT / "build" / (output.name + "-build-depot")) + ";",
        JULIA_PKG_PRECOMPILE_AUTO="0",
        JULIA_CPU_TARGET="generic",
        JULIA_LOAD_PATH="@;@stdlib",
    )
    run(
        runtime / "julia/bin/julia.exe",
        "--startup-file=no",
        ROOT / "packaging/stage_depot.jl",
        runtime / "julia-depot",
        lock["cuda_runtime"],
        engine / "julia_local",
        *([engine / "julia_cuda"] if args.backend == "cuda" else []),
        env=environment,
    )
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
    files = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    if args.backend == "cuda":
        validate_cuda_inventory(files)
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    manifest = dict(
        schema_version=1,
        runtime_id="win-x64-" + identity,
        backends=backends,
        version=project["project"]["version"],
        components=lock,
        files=files,
        wheels={p.name: digest(p) for p in sorted(wheels.glob("*.whl"))},
    )
    (output / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Complete:", output, flush=True)


if __name__ == "__main__":
    main()
