"""Copy, sign and package a macOS app; never modify the staged input."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path

from build_macos_runtime import inventory
from build_windows_runtime import ROOT, digest, run


def macho_type(path):
    with path.open("rb") as stream:
        header = stream.read(32)
        if len(header) < 16:
            return None
        magic = header[:4]
        if magic in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
            offset = struct.unpack(
                ">I" if magic[-1] == 0xBE else ">Q", header[16:20] if magic[-1] == 0xBE else header[16:24]
            )[0]
            stream.seek(offset)
            header = stream.read(32)
            magic = header[:4]
        if len(header) >= 16 and magic in (
            b"\xfe\xed\xfa\xce",
            b"\xfe\xed\xfa\xcf",
            b"\xce\xfa\xed\xfe",
            b"\xcf\xfa\xed\xfe",
        ):
            return struct.unpack(">I" if magic[0] == 0xFE else "<I", header[12:16])[0]
    return None


def verify_resources(resources, manifest):
    files, links = inventory(resources)
    if files != manifest["files"] or links != manifest.get("symlinks", {}):
        raise RuntimeError("Runtime inventory changed")
    if (resources / "resources/ath").exists() or any(
        "wine" in part.lower()
        for name in files
        for part in Path(name).parts
        if part.lower().endswith((".exe", ".dylib"))
    ):
        raise RuntimeError("External Ath/Wine must not be bundled")


def signing_order(app):
    """Seal nested code before signing any enclosing framework executable."""
    native = []
    bundles = []
    for path in app.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file() and macho_type(path) in (2, 6, 8):
            native.append(path)
        elif path.is_dir() and path.suffix in (".framework", ".app"):
            bundles.append(path)
    return native, sorted(native + bundles, key=lambda p: len(p.parts), reverse=True)


def sign(*args):
    for attempt in range(4):
        result = subprocess.run(["codesign", *map(str, args)], capture_output=True, text=True)
        if result.returncode == 0:
            return
        if "timestamp service is not available" not in result.stderr or attempt == 3:
            raise RuntimeError(result.stderr)
        print("Timestamp service unavailable; retrying", flush=True)
        time.sleep(5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    args = parser.parse_args()
    if args.identity == "-":
        parser.error("Developer ID Application identity required")
    source = args.app.resolve()
    resources = source / "Contents/Resources"
    original = (resources / "runtime-manifest.json").read_bytes()
    manifest = json.loads(original)
    if manifest.get("signing"):
        raise ValueError("Use unsigned staging input")
    verify_resources(resources, manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    volume = output / "volume"
    volume.mkdir()
    app = volume / source.name
    shutil.copytree(source, app, symlinks=True)
    resources = app / "Contents/Resources"
    native, targets = signing_order(app)
    for index, path in enumerate(targets):
        print(f"Signing {index + 1}/{len(targets)}: {path.relative_to(app)}", flush=True)
        command = ["codesign", "--force", "--timestamp", "--options", "runtime", "--sign", args.identity]
        if path == resources / "runtime/julia/bin/julia":
            command += ["--entitlements", ROOT / "packaging/entitlements-julia.plist"]
        sign(*command[1:], path)
    (resources / "runtime-manifest-unsigned.json").write_bytes(original)
    files, links = inventory(resources)
    manifest["signing"] = dict(
        identity=args.identity,
        source_runtime_id=manifest["runtime_id"],
        source_manifest_sha256=hashlib.sha256(original).hexdigest(),
        native_files=[str(p.relative_to(resources)) for p in native if p.is_relative_to(resources)],
    )
    manifest.update(files=files, symlinks=links)
    manifest["runtime_id"] = (
        "mac-arm64-signed-" + hashlib.sha256(json.dumps([files, links], sort_keys=True).encode()).hexdigest()[:16]
    )
    (resources / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sign("--force", "--timestamp", "--options", "runtime", "--sign", args.identity, app)
    run("codesign", "--verify", "--deep", "--strict", app)
    verify_resources(resources, manifest)
    (volume / "Applications").symlink_to("/Applications")
    dmg = output / f"Boundary-Lab-{manifest['version']}-mac-arm64.dmg"
    run("hdiutil", "create", "-volname", "Boundary Lab", "-srcfolder", volume, "-ov", "-format", "UDZO", dmg)
    sign("--timestamp", "--sign", args.identity, dmg)
    (output / "SHA256SUMS-unstapled.txt").write_text(f"{digest(dmg)}  {dmg.name}\n")
    print(f"Signed candidate (not yet notarized): {dmg}", flush=True)


if __name__ == "__main__":
    main()
