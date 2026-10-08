"""Qualify relocated/installed Windows payloads offline, checking resource immutability."""

import argparse
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from build_windows_runtime import digest, validate_cuda_inventory

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--solve", action="store_true")
    args = parser.parse_args()
    root = args.payload.resolve()
    if args.destination:
        if args.destination.exists():
            parser.error("Use a fresh relocation directory.")
        shutil.copytree(root, args.destination)
        root = args.destination.resolve()
    manifest = json.loads((root / "runtime-manifest.json").read_text())
    if args.backend not in manifest["backends"]:
        parser.error("Requested backend is not bundled.")
    if "cuda" in manifest["backends"]:
        validate_cuda_inventory(manifest["files"])
    scratch = ROOT / "build" / ("qualification-" + uuid.uuid4().hex[:8])
    scratch.mkdir(parents=True)
    # Exercise the same configure_runtime entrypoint with redirected Known Folder
    # results. No real Documents, settings, or user Julia depot is touched.
    runner = scratch / "run.py"
    runner.write_text(
        "from pathlib import Path\nfrom blab.desktop_runtime import configure_runtime\n"
        + f"configure_runtime(Path({str(root)!r}), documents=Path({str(scratch / 'Documents été')!r}), "
        + f"local_data=Path({str(scratch / 'Local AppData')!r}))\n"
        + f"import runpy\nrunpy.run_path({str(ROOT / 'scripts/windows_smoke.py')!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        HTTP_PROXY="http://127.0.0.1:9",
        HTTPS_PROXY="http://127.0.0.1:9",
        ALL_PROXY="http://127.0.0.1:9",
        NO_PROXY="",
        BLAB_JULIA_EXE="invalid-julia",
        PYTHONPATH="invalid-python",
        JULIA_PROJECT="invalid-project",
    )
    command = [
        root / "runtime/python/python.exe",
        "-I",
        "-B",
        "-X",
        "utf8",
        runner,
        "--output",
        scratch / "Documents été/Boundary Lab/runs/smoke",
        "--backend",
        args.backend,
    ]
    if args.solve:
        command.append("--solve")
    with (scratch / "qualification.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            list(map(str, command)), cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=4200
        )
    if result.returncode:
        raise RuntimeError(f"Qualification failed ({result.returncode}); see {scratch / 'qualification.log'}")
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Installed file changed: {name}")
    unexpected = (
        {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        - set(manifest["files"])
        - {"runtime-manifest.json", "unins000.exe", "unins000.dat", "unins000.msg"}
    )
    if unexpected:
        raise RuntimeError(f"Files created in installation: {sorted(unexpected)[:10]}")
    report = dict(
        runtime_id=manifest["runtime_id"],
        backend=args.backend,
        solve=args.solve,
        resources_unchanged=True,
        log=str(scratch / "qualification.log"),
    )
    (scratch / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
