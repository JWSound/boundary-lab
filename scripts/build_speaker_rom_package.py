"""Build an exploratory parity Petrov--Galerkin package from an exact Level 3 package."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from blab.deploy_solve import DeploySolveCache, prepare_deploy_coupled_request
from blab.solvers.beat_engine_backend import DEFAULT_BEAT_ENGINE_CUDA_PROJECT, BeatEngineWorkerProcess
from blab.solvers.coupled_backend import DEFAULT_COUPLED_SOLVER_SCRIPT
from blab.system_contract import system_frequency_result_from_dict

ROM_QUANTITIES = (
    "speaker_rom_k",
    "speaker_rom_c",
    "speaker_rom_d",
    "speaker_rom_b",
    "speaker_rom_e",
    "speaker_rom_velocity",
    "speaker_rom_current",
    "speaker_rom_velocity_drive",
    "speaker_rom_current_drive",
)


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Exact frequency-parametric Level 3 package.")
    parser.add_argument("output", type=Path, help="Output parity-ROM .blabsp package.")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--training", type=int, default=96)
    parser.add_argument("--validation", type=int, default=24)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    if args.rank <= 0 or args.training < args.rank or args.validation <= 0:
        parser.error("rank must be positive, training >= rank, and validation positive")

    source = args.source.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
    frequencies = [float(value) for value in manifest.get("frequencies_hz", ())]
    if not frequencies:
        raise ValueError("Source speaker package contains no frequencies.")
    physical_metadata = manifest.get("physical_system", {}).get("metadata", {})
    expansion = physical_metadata.get("speaker_export_symmetry_expansion", {})
    source_symmetry = str(expansion.get("source_symmetry", "off"))

    payload = {
        "packagePath": str(source),
        "frequencyHz": frequencies[0],
        "backend": "cuda",
        "fidelity": "coupled",
        "sources": [
            {
                "id": "rom-training-cabinet",
                "positionX": 0.0,
                "positionHeightM": 1.0,
                "positionZ": 0.0,
                "pitchDeg": 0.0,
                "yawDeg": 0.0,
                "rollDeg": 0.0,
                "levelDb": 0.0,
                "delayMs": 0.0,
                "polarity": 1,
            }
        ],
        "rigidObjects": [],
        "observation": {
            "widthM": 1.0,
            "depthM": 1.0,
            "centerXM": 0.0,
            "nearM": 2.0,
            "heightM": 1.2,
            "pitchDeg": 0.0,
            "yawDeg": 0.0,
            "rollDeg": 0.0,
            "columns": 2,
            "rows": 2,
        },
    }
    results = []
    with tempfile.TemporaryDirectory(prefix=".blab-speaker-rom-", dir=output.parent) as temp_dir:
        request_path, request = prepare_deploy_coupled_request(
            payload,
            temp_dir,
            cache=DeploySolveCache(),
        )
        request["frequencies_hz"] = frequencies
        request["outputs"] = [
            {"id": f"rom:{quantity}", "quantity": quantity, "target_ids": [], "options": {}}
            for quantity in ROM_QUANTITIES
        ]
        request["solver_options"]["speaker_rom"] = {
            "rank_per_sector": args.rank,
            "training_count_per_sector": args.training,
            "validation_count_per_sector": args.validation,
            "symmetry": source_symmetry,
        }
        request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
        worker = BeatEngineWorkerProcess(
            julia_executable=args.julia,
            solver_script=DEFAULT_COUPLED_SOLVER_SCRIPT,
            julia_threads="auto",
            julia_project=DEFAULT_BEAT_ENGINE_CUDA_PROJECT,
        )
        try:
            for event in worker.submit(request_path):
                event_type = str(event.get("type", ""))
                if event_type == "result":
                    raw = event.get("result")
                    if isinstance(raw, dict):
                        results.append(system_frequency_result_from_dict(raw))
                elif event_type == "failed":
                    raise RuntimeError(str(event.get("error", "Speaker ROM build failed.")))
        finally:
            worker.terminate()

    if len(results) != len(frequencies):
        raise RuntimeError(f"Expected {len(frequencies)} ROM frequency results, received {len(results)}.")
    arrays_by_quantity: dict[str, list[np.ndarray]] = {name: [] for name in ROM_QUANTITIES}
    metadata: dict[str, object] | None = None
    for result in results:
        quantities = {quantity.quantity: quantity for quantity in result.quantities}
        for name in ROM_QUANTITIES:
            arrays_by_quantity[name].append(np.asarray(quantities[name].values, dtype=np.complex64))
        if metadata is None:
            metadata = dict(quantities["speaker_rom_k"].metadata)
    assert metadata is not None
    rom_bytes = _npz_bytes(
        frequencies_hz=np.asarray(frequencies, dtype=np.float64),
        **{name.removeprefix("speaker_rom_"): np.stack(values, axis=0) for name, values in arrays_by_quantity.items()},
    )

    with zipfile.ZipFile(source, "r") as archive:
        members = {
            name: archive.read(name) for name in archive.namelist() if name not in {"manifest.json", "checksums.json"}
        }
    exact_declaration = dict(manifest["files"]["coupled_model"])
    members["data/coupled-rom.npz"] = rom_bytes
    manifest["files"]["coupled_exact_reference"] = exact_declaration
    manifest["files"]["coupled_model"] = {
        "path": "data/coupled-rom.npz",
        "representation": "parity_petrov_galerkin_rom",
        "format_version": 1,
        "symmetry_mode": metadata["symmetry_mode"],
        "image_count": metadata["image_count"],
        "rank_per_sector": args.rank,
        "sector_names": metadata["sector_names"],
        "sector_signs": metadata["sector_signs"],
        "node_orbits": metadata["node_orbits"],
        "face_orbits": metadata["face_orbits"],
        "input_ports": exact_declaration.get("input_ports", []),
        "equations": metadata["equations"],
        "validation": [
            {"frequency_hz": frequencies[index], "sectors": results[index].quantities[0].metadata["validation"]}
            for index in range(len(results))
        ],
        "matrix_dimensions": {
            "k": ["frequency", "parity_sector", "reduced_row", "reduced_column"],
            "c": ["frequency", "parity_sector", "reduced_row", "boundary_node_orbit"],
            "d": ["frequency", "parity_sector", "boundary_face_orbit", "reduced_column"],
            "b": ["frequency", "parity_sector", "reduced_row", "input_port"],
            "e": ["frequency", "parity_sector", "boundary_face_orbit", "input_port"],
            "velocity": ["frequency", "parity_sector", "transducer", "reduced_column"],
            "current": ["frequency", "parity_sector", "transducer", "reduced_column"],
            "velocity_drive": ["frequency", "parity_sector", "transducer", "input_port"],
            "current_drive": ["frequency", "parity_sector", "transducer", "input_port"],
        },
    }
    capabilities = list(manifest.get("capabilities", ()))
    if "parity_petrov_galerkin_rom" not in capabilities:
        capabilities.append("parity_petrov_galerkin_rom")
    manifest["capabilities"] = capabilities
    members["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    members["checksums.json"] = (
        json.dumps(
            {name: hashlib.sha256(payload_bytes).hexdigest() for name, payload_bytes in sorted(members.items())},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, payload_bytes in members.items():
            archive.writestr(name, payload_bytes)
    temporary.replace(output)
    print(f"Wrote {output} ({output.stat().st_size / (1024**2):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
