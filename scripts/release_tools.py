"""Small, dependency-free gates used by the release workflows."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(tag):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:rc\d+)?", version) or tag != "v" + version:
        raise ValueError("Release tag must be v + package version (stable or rcN).")
    notes = ROOT / "docs/releases" / f"{version}.md"
    if not notes.is_file():
        raise ValueError(f"Missing release notes: {notes}")
    dependencies = project.get("dependencies", [])
    beat = next((d for d in dependencies if d.startswith("beat-engine")), None)
    if beat and not re.fullmatch(
        r"beat-engine @ https://github.com/JWSound/BEAT_Engine/releases/download/v\d+\.\d+\.\d+(?:rc\d+)?/beat_engine-[^/]+\.whl#sha256=[0-9a-f]{64}",
        beat,
    ):
        raise ValueError("Releases require a published BEAT wheel with a SHA-256 pin.")
    desktop = ROOT / "desktop/package.json"
    if desktop.exists():
        package = json.loads(desktop.read_text(encoding="utf-8"))
        lock = json.loads((ROOT / "desktop/package-lock.json").read_text(encoding="utf-8"))
        npm_version = re.sub(r"rc(\d+)$", r"-rc.\1", version)
        if (
            package["version"] != npm_version
            or lock["version"] != npm_version
            or lock["packages"][""]["version"] != npm_version
        ):
            raise ValueError("Python, npm and npm lock versions must agree.")
        runtime = json.loads((ROOT / "packaging/runtime-lock.json").read_text(encoding="utf-8"))
        if runtime["beat_requirement"] != beat:
            raise ValueError("Application and bundled runtime BEAT pins must agree.")
    init = ROOT / "src/beat_engine/__init__.py"
    if init.exists() and f'__version__ = "{version}"' not in init.read_text(encoding="utf-8"):
        raise ValueError("Engine runtime version disagrees with package metadata.")
    print(f"Validated {tag}")


def require_ci(workflow):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    repo = os.environ["GITHUB_REPOSITORY"]
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs?head_sha={sha}&event=push&branch=main&status=success&per_page=100"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
            "User-Agent": "Boundary-release",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        runs = json.load(response)["workflow_runs"]
    if not any(r["head_sha"] == sha and r["head_branch"] == "main" and r["conclusion"] == "success" for r in runs):
        raise RuntimeError("Release commit has no successful main push CI run.")
    print(f"CI passed for {sha}")


def checksums(directory):
    directory = Path(directory)
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    if not files:
        raise ValueError("No release artifacts found.")
    lines = []
    for path in files:
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").hexdigest()
        lines.append(f"{value}  {path.name}\n")
    (directory / "SHA256SUMS.txt").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["check", "require-ci", "checksums"])
    parser.add_argument("value")
    args = parser.parse_args()
    {"check": check, "require-ci": require_ci, "checksums": checksums}[args.operation](args.value)
