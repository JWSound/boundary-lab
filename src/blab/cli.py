from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import import_module


COMMAND_MODULES = {
    "clean": "blab.mesh_clean",
    "gui": "blab.gui",
    "solve": "blab.solver",
    "prepare": "blab.postprocess",
    "plot": "blab.plotting",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blab",
        description="Run Boundary Lab mesh cleaning, solving, preparation, and plotting commands.",
    )
    parser.add_argument("command", choices=tuple(COMMAND_MODULES), help="Workflow command to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the command")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    module = import_module(COMMAND_MODULES[args.command])
    module.main(args.args, prog=f"bemps {args.command}")


if __name__ == "__main__":
    main()
