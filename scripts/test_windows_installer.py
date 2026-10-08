"""Guarded silent install, installed-runtime solve qualification, and uninstall."""

import argparse
import json
import subprocess
import sys
import winreg
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{557D58CB-ED76-465A-A8DE-19C03801A536}_is1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installer", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    target = args.directory.resolve()
    build_root = (ROOT / "build").resolve()
    if not target.is_relative_to(build_root) or target == build_root or target.exists():
        parser.error("Use a new test installation directory strictly inside this repository's build directory.")
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            key = winreg.OpenKey(hive, UNINSTALL_KEY, access=winreg.KEY_READ | winreg.KEY_WOW64_64KEY)
        except FileNotFoundError:
            continue
        else:
            key.Close()
            parser.error("Boundary Lab is already installed; refusing to change an existing installation.")
    installer = args.installer.resolve()
    log = build_root / (target.name + "-install.log")
    installed = False
    try:
        subprocess.run(
            [
                str(installer),
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                f"/DIR={target}",
                "/GROUP=Boundary Lab Evaluation",
                f"/LOG={log}",
            ],
            check=True,
            timeout=900,
        )
        installed = True
        subprocess.run([str(target / "Boundary Lab.exe"), "--initialize"], check=True, timeout=120)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/verify_windows_runtime.py"),
                "--payload",
                str(target),
                "--backend",
                args.backend,
                "--solve",
            ],
            check=True,
            timeout=4500,
        )
        report = {"installer": str(installer), "installed": str(target), "backend": args.backend, "qualified": True}
    finally:
        uninstaller = target / "unins000.exe"
        if installed and uninstaller.is_file() and uninstaller.resolve().is_relative_to(build_root):
            subprocess.run(
                [str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], check=True, timeout=300
            )
    if (target / "Boundary Lab.exe").exists():
        raise RuntimeError("Uninstaller left the application executable behind.")
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY):
            raise RuntimeError("Uninstaller left its registration behind.")
    except FileNotFoundError:
        pass
    # The installer and bootstrap must agree on the redirected Documents location.
    sys.path.insert(0, str(ROOT / "src"))
    from blab.desktop_runtime import documents_directory

    user = documents_directory() / "Boundary Lab"
    assert (user / "examples/Simple_Sealed/simple_sealed.blab.json").is_file()
    assert (user / "documentation/Boundary Lab Guide.pdf").is_file()
    report.update(uninstalled=True, documents_preserved=True)
    (build_root / (target.name + "-lifecycle.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
