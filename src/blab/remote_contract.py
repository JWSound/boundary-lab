"""Portable, versioned physical-system jobs; no client paths reach the worker."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from blab.solvers.coupled_backend import validate_solve_plan
from blab.system_contract import (
    SystemSolveRequest,
    system_solve_request_from_dict,
    system_solve_request_to_dict,
)

REMOTE_VERSION = 1
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_ENTRIES = 1024


def _validate_envelope(envelope: dict) -> None:
    if not isinstance(envelope, dict) or set(envelope) != {
        "protocol",
        "version",
        "backend_id",
        "observation_planes",
        "request",
        "assets",
    }:
        raise ValueError("Invalid remote job envelope fields.")
    if (
        envelope["protocol"] != "blab-remote"
        or type(envelope["version"]) is not int
        or envelope["version"] != REMOTE_VERSION
    ):
        raise ValueError("Unsupported remote job protocol/version.")
    if envelope["backend_id"] != "beat_cpu" or envelope["observation_planes"] is not False:
        raise ValueError("Remote v1 supports CPU jobs without observation planes only.")
    request = envelope["request"]
    if not isinstance(request, dict) or set(request) != {
        "schema_version",
        "compiled_system",
        "frequencies_hz",
        "excitation_port_ids",
        "outputs",
        "solver_options",
    }:
        raise ValueError("Invalid remote solve request fields.")
    # Runtime paths and future executable options must not silently enter this API.
    allowed_options = {
        "quadrature_order",
        "singular_order",
        "validation_diagnostics",
        "cache_frequency_invariant",
        "static_condensation",
        "symmetry",
        "precision",
        "bem_backend",
        "transducer_reference_voltage_v",
    }
    if set(request["solver_options"]) - allowed_options:
        raise ValueError("Unsupported remote solver_options.")
    for output in request["outputs"]:
        if set(output.get("options", {})) - {"points_m", "observation_domains"}:
            raise ValueError("Unsupported remote output options.")
    system_solve_request_from_dict(request)


def build_job_bundle(request: SystemSolveRequest) -> bytes:
    """Snapshot required meshes, replacing their file references with content hashes."""
    raw = system_solve_request_to_dict(request)
    assets = {}
    for mesh in raw["compiled_system"]["meshes"]:
        path = Path(mesh["file"])
        if path.stat().st_size > MAX_BUNDLE_BYTES:
            raise ValueError("Mesh exceeds remote bundle size limit.")
        data = path.read_bytes()
        name = f"assets/{hashlib.sha256(data).hexdigest()}.msh"
        assets[name] = data
        mesh["file"] = name
    envelope = {
        "protocol": "blab-remote",
        "version": REMOTE_VERSION,
        "backend_id": "beat_cpu",
        "observation_planes": False,
        "request": raw,
        "assets": [
            {"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(assets.items())
        ],
    }
    _validate_envelope(envelope)
    manifest = json.dumps(envelope, allow_nan=False).encode()
    if len(assets) + 1 > MAX_ENTRIES or len(manifest) + sum(map(len, assets.values())) > MAX_BUNDLE_BYTES:
        raise ValueError("Remote bundle exceeds size/entry limit.")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("job.json", manifest)
        for name, data in assets.items():
            archive.writestr(name, data)
    bundle = stream.getvalue()
    if len(bundle) > MAX_BUNDLE_BYTES:
        raise ValueError("Remote bundle exceeds upload limit.")
    return bundle


def stage_job_bundle(bundle: bytes, directory: Path) -> SystemSolveRequest:
    """Validate the entire archive before creating an exclusively owned job directory."""
    if len(bundle) > MAX_BUNDLE_BYTES:
        raise ValueError("Remote bundle exceeds upload limit.")
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(entries) > MAX_ENTRIES or len(names) != len(set(names)):
            raise ValueError("Duplicate or excessive archive entries.")
        if sum(entry.file_size for entry in entries) > MAX_BUNDLE_BYTES:
            raise ValueError("Remote bundle exceeds expanded size limit.")
        envelope = json.loads(archive.read("job.json"))
        _validate_envelope(envelope)
        assets = envelope["assets"]
        if not isinstance(assets, list):
            raise ValueError("assets must be an array.")
        contents = {}
        for asset in assets:
            name = asset["name"]
            if not isinstance(name, str) or not re.fullmatch(r"assets/[0-9a-f]{64}\.msh", name):
                raise ValueError("Invalid asset name.")
            if name in contents:
                raise ValueError("Duplicate asset manifest entry.")
            data = archive.read(name)
            digest = hashlib.sha256(data).hexdigest()
            if (
                type(asset["size"]) is not int
                or len(data) != asset["size"]
                or digest != asset["sha256"]
                or name != f"assets/{digest}.msh"
            ):
                raise ValueError("Asset size/hash mismatch.")
            contents[name] = data
        if set(names) != {"job.json", *contents}:
            raise ValueError("Unexpected or missing archive entries.")
        raw = envelope["request"]
        meshes = raw["compiled_system"]["meshes"]
        if {mesh["file"] for mesh in meshes} != set(contents):
            raise ValueError("Every mesh must reference a bundled asset; unused assets are rejected.")
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "assets").mkdir()
        (directory / "job.json").write_text(json.dumps(envelope), encoding="utf-8")
        for name, data in contents.items():
            (directory / name).write_bytes(data)
        for mesh in meshes:
            mesh["file"] = str(directory / mesh["file"])
        request = system_solve_request_from_dict(raw)
        validate_solve_plan(request)
        return request
