"""Test a signed DMG in a fresh spaced installation with cold/warm offline solves."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from build_macos_installer import verify_resources
from build_windows_runtime import ROOT, digest, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    mount = output / "Mounted Image"
    mount.mkdir()
    run("hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", mount, args.dmg.resolve())
    app = output / "Boundary Lab.app"
    try:
        run("ditto", mount / "Boundary Lab.app", app)
    finally:
        run("hdiutil", "detach", mount)
    run("codesign", "--verify", "--deep", "--strict", app)
    resources = app / "Contents/Resources"
    manifest = json.loads((resources / "runtime-manifest.json").read_text())
    if not manifest.get("signing"):
        raise RuntimeError("Expected signed runtime")
    verify_resources(resources, manifest)
    python = resources / "runtime/python/bin/python3"
    runner = output / "run.py"
    runner.write_text(
        "from pathlib import Path\nfrom blab.desktop_runtime import configure_runtime\n"
        + f"configure_runtime(Path({str(resources)!r}), documents=Path({str(output / 'Documents été')!r}), local_data=Path({str(output / 'Application Support')!r}))\n"
        + f"import runpy\nrunpy.run_path({str(ROOT / 'scripts/macos_smoke.py')!r}, run_name='__main__')\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith(("PYTHON", "JULIA", "BLAB_", "QT_", "QML", "DYLD_", "LD_LIBRARY_PATH"))
    }
    env.update(
        HTTP_PROXY="http://127.0.0.1:9",
        HTTPS_PROXY="http://127.0.0.1:9",
        ALL_PROXY="http://127.0.0.1:9",
        NO_PROXY="",
        PYTHONPATH="/invalid-python",
        BLAB_JULIA_EXE="/invalid-julia",
        JULIA_PROJECT="/invalid-project",
    )
    launcher_home = output / "Launcher Home"
    launcher_home.mkdir()
    run(app / "Contents/MacOS/Boundary Lab", "--initialize", env=dict(env, HOME=str(launcher_home)), cwd=output)
    if not (launcher_home / "Documents/Boundary Lab/runs").is_dir():
        raise RuntimeError("Native launcher did not initialize its isolated user files")
    for backend in manifest["backends"]:
        if backend not in ("cpu", "metal"):
            raise RuntimeError(f"Unsupported qualification backend: {backend}")
        for phase in ("cold", "warm"):
            label = f"{backend}-{phase}"
            command = [python, "-I", "-B", runner, "--output", output / label, "--backend", backend, "--solve"]
            if backend == "metal":
                command += ["--cpu-reference", output / f"cpu-{phase}"]
            with (output / f"{label}.log").open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=output,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=4200,
                )
            if result.returncode:
                raise RuntimeError(f"{label} qualification failed; see {output / (label + '.log')}")
            verify_resources(resources, manifest)
            print(f"{label} GUI, VTK, Gmsh and solves passed", flush=True)
    (output / "verification.json").write_text(
        json.dumps(
            dict(
                runtime_id=manifest["runtime_id"],
                dmg_sha256=digest(args.dmg),
                backends=manifest["backends"],
                cold=True,
                warm=True,
                resources_unchanged=True,
                ath_wine_bundled=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
