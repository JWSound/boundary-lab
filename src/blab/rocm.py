"""Discover and configure a local ROCm SDK installation."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

_SOURCE_LABELS = {
    "argument": "command-line path",
    "BLAB_ROCM_PATH": "BLAB_ROCM_PATH",
    "config": "Boundary Lab configuration",
    "HIP_PATH": "HIP_PATH",
    "ROCM_PATH": "ROCM_PATH",
    "ROCM_HOME": "ROCM_HOME",
    "program_files": "AMD Windows installation",
}


@dataclass(frozen=True)
class RocmInstallation:
    root: Path
    source: str
    valid: bool
    missing: tuple[str, ...] = ()

    @property
    def source_label(self) -> str:
        return _SOURCE_LABELS.get(self.source, self.source)

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["root"] = str(self.root)
        payload["source_label"] = self.source_label
        return payload


def default_rocm_config_path(environ: Mapping[str, str] | None = None) -> Path:
    environment = os.environ if environ is None else environ
    override = _environment_value(environment, "BLAB_ROCM_CONFIG")
    if override:
        return Path(_clean_path_text(override)).expanduser()

    local_app_data = _environment_value(environment, "LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "Boundary Lab" / "rocm_path.txt"
    return Path.home() / ".config" / "boundary-lab" / "rocm_path.txt"


def load_configured_rocm_path(config_path: str | Path | None = None) -> Path | None:
    path = default_rocm_config_path() if config_path is None else Path(config_path)
    try:
        value = _clean_path_text(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        return None
    return Path(value).expanduser() if value else None


def save_configured_rocm_path(root: str | Path, config_path: str | Path | None = None) -> Path:
    path = default_rocm_config_path() if config_path is None else Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{Path(root).expanduser()}\n", encoding="utf-8")
    return path


def clear_configured_rocm_path(config_path: str | Path | None = None) -> bool:
    path = default_rocm_config_path() if config_path is None else Path(config_path)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def inspect_rocm_root(root: str | Path, *, source: str = "argument") -> RocmInstallation:
    candidate = Path(_clean_path_text(str(root))).expanduser()
    missing: list[str] = []
    if not candidate.is_dir():
        return RocmInstallation(candidate, source, False, ("SDK directory",))

    library_dirs = tuple(
        path for path in (candidate, candidate / "bin", candidate / "lib", candidate / "lib64") if path.is_dir()
    )
    required_libraries = {
        "HIP runtime (amdhip64)": ("amdhip64*.dll", "libamdhip64.so*"),
        "rocBLAS": ("rocblas*.dll", "librocblas.so*"),
        "rocSOLVER": ("rocsolver*.dll", "librocsolver.so*"),
        "rocSPARSE": ("rocsparse*.dll", "librocsparse.so*"),
    }
    for label, patterns in required_libraries.items():
        if not _contains_pattern(library_dirs, patterns):
            missing.append(label)

    executable_dirs = (candidate / "bin", candidate / "hip" / "bin")
    executable_names = ("hipconfig.exe", "hipconfig", "hipInfo.exe", "hipInfo")
    if not any((directory / name).is_file() for directory in executable_dirs for name in executable_names):
        missing.append("hipconfig or hipInfo")

    return RocmInstallation(candidate.resolve(), source, not missing, tuple(missing))


def rocm_candidates(
    *,
    explicit_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
    program_files_roots: Sequence[str | Path] | None = None,
) -> tuple[RocmInstallation, ...]:
    environment = os.environ if environ is None else environ
    raw_candidates: list[tuple[str, Path]] = []
    if explicit_path is not None:
        raw_candidates.append(("argument", Path(_clean_path_text(str(explicit_path))).expanduser()))
    else:
        for variable in ("BLAB_ROCM_PATH",):
            value = _environment_value(environment, variable)
            if value:
                raw_candidates.append((variable, Path(_clean_path_text(value)).expanduser()))

        selected_config_path = default_rocm_config_path(environment) if config_path is None else config_path
        configured = load_configured_rocm_path(selected_config_path)
        if configured is not None:
            raw_candidates.append(("config", configured))

        for variable in ("HIP_PATH", "ROCM_PATH", "ROCM_HOME"):
            value = _environment_value(environment, variable)
            if value:
                raw_candidates.append((variable, Path(_clean_path_text(value)).expanduser()))

        standard_roots = (
            _windows_program_files_rocm_roots(environment)
            if program_files_roots is None
            else tuple(Path(root) for root in program_files_roots)
        )
        raw_candidates.extend(("program_files", root) for root in standard_roots)

    candidates: list[RocmInstallation] = []
    seen: set[str] = set()
    for source, root in raw_candidates:
        key = os.path.normcase(os.path.abspath(root))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(inspect_rocm_root(root, source=source))
    return tuple(candidates)


def discover_rocm(
    *,
    explicit_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
    program_files_roots: Sequence[str | Path] | None = None,
) -> RocmInstallation | None:
    return next(
        (
            candidate
            for candidate in rocm_candidates(
                explicit_path=explicit_path,
                environ=environ,
                config_path=config_path,
                program_files_roots=program_files_roots,
            )
            if candidate.valid
        ),
        None,
    )


def _windows_program_files_rocm_roots(environment: Mapping[str, str]) -> tuple[Path, ...]:
    parents: list[Path] = []
    for variable in ("ProgramW6432", "ProgramFiles"):
        value = _environment_value(environment, variable)
        if value:
            parent = Path(value) / "AMD" / "ROCm"
            if parent not in parents:
                parents.append(parent)

    roots: list[Path] = []
    for parent in parents:
        try:
            children = [path for path in parent.iterdir() if path.is_dir()]
        except OSError:
            continue
        roots.extend(sorted(children, key=_version_sort_key, reverse=True))
    return tuple(roots)


def _version_sort_key(path: Path) -> tuple[tuple[int, ...], str]:
    numbers = tuple(int(part) for part in re.findall(r"\d+", path.name))
    return numbers, path.name.casefold()


def _contains_pattern(directories: Sequence[Path], patterns: Sequence[str]) -> bool:
    return any(any(directory.glob(pattern)) for directory in directories for pattern in patterns)


def _environment_value(environment: Mapping[str, str], name: str) -> str:
    if name in environment:
        return str(environment[name]).strip()
    folded_name = name.casefold()
    return next((str(value).strip() for key, value in environment.items() if key.casefold() == folded_name), "")


def _clean_path_text(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def _build_arg_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Discover and configure an AMD ROCm SDK for Boundary Lab.")
    commands = parser.add_subparsers(dest="command", required=True)

    detect = commands.add_parser("detect", help="Find and validate an existing ROCm SDK")
    detect.add_argument("--path", help="Validate only this SDK root")
    detect.add_argument("--config", type=Path, help="Override the user configuration file")
    detect.add_argument("--json", action="store_true", help="Print machine-readable diagnostics")
    detect.add_argument("--root-file", type=Path, help="Write the selected SDK root to this file")

    configure = commands.add_parser("configure", help="Validate and remember a custom ROCm SDK root")
    configure.add_argument("path", help="ROCm SDK root to remember")
    configure.add_argument("--config", type=Path, help="Override the user configuration file")

    clear = commands.add_parser("clear", help="Forget the saved custom ROCm SDK root")
    clear.add_argument("--config", type=Path, help="Override the user configuration file")
    return parser


def _detect(args: argparse.Namespace) -> int:
    candidates = rocm_candidates(explicit_path=args.path, config_path=args.config)
    selected = next((candidate for candidate in candidates if candidate.valid), None)
    if args.json:
        print(
            json.dumps(
                {
                    "selected": None if selected is None else selected.payload(),
                    "candidates": [candidate.payload() for candidate in candidates],
                },
                indent=2,
            )
        )
    elif selected is not None:
        print(f"ROCm SDK: {selected.root}")
        print(f"Detected from: {selected.source_label}")
    else:
        print("No functional ROCm SDK layout was found.", file=sys.stderr)
        for candidate in candidates:
            print(f"  {candidate.root}: missing {', '.join(candidate.missing)}", file=sys.stderr)

    if selected is not None and args.root_file is not None:
        args.root_file.parent.mkdir(parents=True, exist_ok=True)
        args.root_file.write_text(f"{selected.root}\n", encoding="utf-8")
    return 0 if selected is not None else 1


def _configure(args: argparse.Namespace) -> int:
    installation = inspect_rocm_root(args.path)
    if not installation.valid:
        print(f"Invalid ROCm SDK root: {installation.root}", file=sys.stderr)
        print(f"Missing: {', '.join(installation.missing)}", file=sys.stderr)
        return 1
    config_path = save_configured_rocm_path(installation.root, args.config)
    print(f"Saved ROCm SDK: {installation.root}")
    print(f"Configuration: {config_path}")
    return 0


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    args = _build_arg_parser(prog).parse_args(argv)
    if args.command == "detect":
        status = _detect(args)
    elif args.command == "configure":
        status = _configure(args)
    else:
        removed = clear_configured_rocm_path(args.config)
        print("Saved ROCm SDK configuration removed." if removed else "No saved ROCm SDK configuration was present.")
        status = 0
    raise SystemExit(status)


if __name__ == "__main__":
    main()
