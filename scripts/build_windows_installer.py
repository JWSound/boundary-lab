"""Compile a local unsigned candidate and record its checksum; never publish."""

import argparse
import json
import subprocess
from pathlib import Path

from build_windows_runtime import ROOT, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--iscc", type=Path, required=True)
    args = parser.parse_args()
    payload = args.payload.resolve()
    manifest = json.loads((payload / "runtime-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(payload / name) != expected:
            raise RuntimeError(f"Payload changed after staging: {name}")
    flavor = "cuda" if "cuda" in manifest["backends"] else "cpu"
    version = manifest["version"]
    subprocess.run(
        [
            str(args.iscc.resolve()),
            f"/DPayload={payload}",
            f"/DAppVersion={version}",
            f"/DFlavor={flavor}",
            str(ROOT / "packaging/windows.iss"),
        ],
        check=True,
    )
    installer = ROOT / "dist" / f"Boundary-Lab-{version}-windows-x64-{flavor}.exe"
    checksum = digest(installer)
    installer.with_suffix(".exe.sha256").write_text(f"{checksum}  {installer.name}\n")
    print(
        json.dumps(
            {
                "installer": str(installer),
                "bytes": installer.stat().st_size,
                "sha256": checksum,
                "signed": False,
                "runtime_id": manifest["runtime_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
