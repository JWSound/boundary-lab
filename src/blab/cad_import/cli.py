"""``blab cad-import``: mesh a STEP file into a FEM/BEM assembly with the CAD import pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .pipeline import AXES, SYMMETRY_LABELS, CadImportSettings, mesh_step


def _build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Mesh a STEP file: solids become FEM air bodies, surface bodies the BEM exterior. "
        "Closed shells of surfaces are exterior unless --closed-shell makes them air.",
    )
    parser.add_argument("step", help="STEP file (units are taken from the file)")
    parser.add_argument("--fmax", type=float, required=True, help="Maximum frequency in Hz")
    parser.add_argument(
        "--elements-per-wavelength", type=float, default=5.0, help="Elements per wavelength at --fmax (default: 5)"
    )
    parser.add_argument("--output", required=True, help="Output folder for <mesh id>.msh files (metres)")
    parser.add_argument(
        "--sound-speed", type=float, default=343.0, help="Sound speed in m/s for the target size (default: 343)"
    )
    parser.add_argument("--forward", choices=tuple(AXES), default="+Z", help="CAD axis mapped to +Z (default: +Z)")
    parser.add_argument("--up", choices=tuple(AXES), default="+Y", help="CAD axis mapped to +Y (default: +Y)")
    parser.add_argument(
        "--closed-shell",
        action="append",
        default=[],
        metavar="NAME=air|exterior",
        help="Treat a closed shell of surfaces as an air volume or exterior surface (default: exterior); repeatable",
    )
    return parser


def _closed_shells(parser, values) -> dict[str, str]:
    roles = {}
    for value in values:
        name, _, role = value.rpartition("=")
        if not name or role not in ("air", "exterior"):
            parser.error(f"--closed-shell takes NAME=air or NAME=exterior, not '{value}'.")
        roles[name] = role
    return roles


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    parser = _build_arg_parser(prog)
    args = parser.parse_args(argv)
    try:
        settings = CadImportSettings(
            fmax_hz=args.fmax,
            sound_speed_m_per_s=args.sound_speed,
            elements_per_wavelength=args.elements_per_wavelength,
            forward=args.forward,
            up=args.up,
            closed_shells=_closed_shells(parser, args.closed_shell),
        )
        result = mesh_step(args.step, settings, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    geometry = result.geometry
    low, high = geometry.bbox_m
    size_mm = " x ".join(f"{(b - a) * 1000:.1f}" for a, b in zip(low, high, strict=True))
    print(f"Target element size: {result.target_size_m * 1000:.1f} mm")
    print(f"Bounding box: {size_mm} mm")
    for mesh in result.meshes:
        volumes = f"volume groups {', '.join(mesh.volume_groups)}; " if mesh.volume_groups else ""
        print(f"Wrote {mesh.path.name} ({mesh.purpose}): {volumes}surface groups {', '.join(mesh.surface_groups)}")
    for shell in geometry.closed_shells:
        print(f"Closed shell: {shell.name} ({shell.role})")
    for interface in geometry.interfaces:
        print(f"Interface: {interface.name} ({interface.mesh_id} <-> exterior)")
    print(f"Suggested symmetry: {SYMMETRY_LABELS[geometry.symmetry]}")
    for warning in geometry.warnings:
        print(f"Warning: {warning}")
