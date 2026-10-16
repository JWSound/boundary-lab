"""Submit an installed-qualified DMG, staple acceptance and record final checksums."""

import argparse
import json
import subprocess
from pathlib import Path

from build_windows_runtime import digest, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmg", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--profile", required=True, help="Existing notarytool Keychain profile")
    args = parser.parse_args()
    dmg = args.dmg.resolve()
    qualification = json.loads((args.qualification / "verification.json").read_text())
    if not all(qualification.get(k) for k in ("cold", "warm", "resources_unchanged")):
        raise RuntimeError("Installed qualification must pass first")
    receipt = dmg.parent / "notarization-submission.json"
    if receipt.exists():
        submission = json.loads(receipt.read_text())
        if digest(dmg) not in (submission["submitted_sha256"], submission.get("stapled_sha256")):
            raise RuntimeError("DMG no longer matches its submission")
    else:
        if digest(dmg) != qualification["dmg_sha256"]:
            raise RuntimeError("DMG differs from the installed-qualified artifact")
        response = subprocess.check_output(
            ["xcrun", "notarytool", "submit", str(dmg), "--keychain-profile", args.profile, "--output-format", "json"],
            text=True,
        )
        submission = json.loads(response)
        submission["submitted_sha256"] = digest(dmg)
        receipt.write_text(json.dumps(submission, indent=2))
    subprocess.run(
        ["xcrun", "notarytool", "wait", submission["id"], "--keychain-profile", args.profile, "--timeout", "20m"],
        check=False,
    )
    info = json.loads(
        subprocess.check_output(
            [
                "xcrun",
                "notarytool",
                "info",
                submission["id"],
                "--keychain-profile",
                args.profile,
                "--output-format",
                "json",
            ],
            text=True,
        )
    )
    (dmg.parent / "notarization-result.json").write_text(json.dumps(info, indent=2))
    if info["status"] != "In Progress":
        run(
            "xcrun",
            "notarytool",
            "log",
            submission["id"],
            "--keychain-profile",
            args.profile,
            dmg.parent / "notarization-log.json",
        )
    if info["status"] != "Accepted":
        raise RuntimeError(f"Notarization status: {info['status']}; see saved results or resume this command")
    run("xcrun", "stapler", "staple", dmg)
    submission["stapled_sha256"] = digest(dmg)
    receipt.write_text(json.dumps(submission, indent=2))
    run("xcrun", "stapler", "validate", dmg)
    run("codesign", "--verify", "--strict", dmg)
    run("spctl", "--assess", "--type", "open", "--context", "context:primary-signature", "--verbose=2", dmg)
    run("spctl", "--assess", "--type", "execute", "--verbose=2", args.qualification / "Boundary Lab.app")
    (dmg.parent / "SHA256SUMS.txt").write_text(f"{digest(dmg)}  {dmg.name}\n")
    print(f"Notarized and qualified: {dmg}")


if __name__ == "__main__":
    main()
