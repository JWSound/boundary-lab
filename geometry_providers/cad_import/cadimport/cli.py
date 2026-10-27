"""``python -m cadimport.cli``: mesh a STEP file into a FEM/BEM assembly with the CAD import pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from blab.defaults import DEFAULT_SOUND_SPEED_M_PER_S

from .pipeline import SYMMETRY_LABELS, CadImportSettings, mesh_step


def _build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Mesh a STEP file: solids become FEM air bodies, surface bodies the BEM exterior. "
        "Closed shells of surfaces are guessed air bodies (or, when the file has no other exterior "
        "surface, the largest one exterior) unless --closed-shell says otherwise.",
    )
    parser.add_argument("step", help="STEP file (units are taken from the file)")
    parser.add_argument("--fmax", type=float, required=True, help="Maximum frequency in Hz")
    parser.add_argument(
        "--elements-per-wavelength",
        type=float,
        default=6.0,
        help="FEM elements per wavelength at --fmax, for air bodies (default: 6)",
    )
    parser.add_argument(
        "--bem-elements-per-wavelength",
        type=float,
        help="BEM elements per wavelength at --fmax, for the exterior (default: --elements-per-wavelength)",
    )
    parser.add_argument("--output", required=True, help="Output folder for <mesh id>.msh files (metres)")
    parser.add_argument(
        "--sound-speed",
        type=float,
        default=DEFAULT_SOUND_SPEED_M_PER_S,
        help=f"Sound speed in m/s for the target size (default: {DEFAULT_SOUND_SPEED_M_PER_S:g})",
    )
    parser.add_argument(
        "--rotate",
        default="0,0,0",
        metavar="X,Y,Z",
        help="Rotate by these degrees about the X, Y and Z axes, in that order (write --rotate=-90,0,0)",
    )
    parser.add_argument(
        "--translate",
        default="0,0,0",
        metavar="X,Y,Z",
        help="Then translate by these millimetres (write --translate=-30,0,0)",
    )
    parser.add_argument(
        "--closed-shell",
        action="append",
        default=[],
        metavar="NAME=air|exterior",
        help="Treat a closed shell of surfaces as an air body or exterior surface (default: guessed); repeatable",
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


def _triple(parser, option, value) -> tuple[float, float, float]:
    try:
        values = tuple(float(v) for v in value.split(","))
    except ValueError:
        values = ()
    if len(values) != 3:
        parser.error(f"{option} takes X,Y,Z, not '{value}'.")
    return values


def main(argv: Sequence[str] | None = None, prog: str = "python -m cadimport.cli") -> None:
    parser = _build_arg_parser(prog)
    args = parser.parse_args(argv)
    try:
        settings = CadImportSettings(
            fmax_hz=args.fmax,
            sound_speed_m_per_s=args.sound_speed,
            elements_per_wavelength=args.elements_per_wavelength,
            bem_elements_per_wavelength=args.bem_elements_per_wavelength,
            closed_shells=_closed_shells(parser, args.closed_shell),
            rotation_deg=_triple(parser, "--rotate", args.rotate),
            translation_m=tuple(v / 1000 for v in _triple(parser, "--translate", args.translate)),
        )
        result = mesh_step(args.step, settings, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    geometry = result.geometry
    low, high = geometry.bbox_m
    size_mm = " x ".join(f"{(b - a) * 1000:.1f}" for a, b in zip(low, high, strict=True))
    print(f"FEM target element size: {settings.sizing.target_size_m * 1000:.1f} mm")
    print(f"BEM target element size: {settings.sizing.bem_target_size_m * 1000:.1f} mm")
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


if __name__ == "__main__":
    main()
