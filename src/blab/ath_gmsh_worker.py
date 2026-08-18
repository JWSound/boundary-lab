"""Cancellable subprocess worker for meshing Ath blab-mode GEO output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from blab.ath import ath_run_result_to_payload, build_ath_mesh_artifacts, mesh_ath_geo_text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--mirror-axes", default="")
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        geo_text = args.geo.read_text(encoding="utf-8")
        raw_mesh = mesh_ath_geo_text(
            geo_text,
            output_dir=args.output_dir,
            case_name=args.case_name,
        )
        result = build_ath_mesh_artifacts(
            raw_mesh,
            output_dir=args.output_dir,
            case_name=args.case_name,
            config_path=args.config,
            geo_path=args.geo,
            log_path=args.log,
            mirror_axes=tuple(args.mirror_axes),
        )
        args.result.write_text(
            json.dumps(ath_run_result_to_payload(result), separators=(",", ":")),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
