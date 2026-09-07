"""Retirement notice for the legacy HTTP solve server."""

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    raise SystemExit(
        "The legacy HTTP solve server is retired. "
        "Open an existing project in the GUI to migrate it, then use "
        "'blab project validate <project.blab.json> --json' and "
        "'blab project solve <project.blab.json>'. Physical-system solves use BEAT Engine."
    )


if __name__ == "__main__":
    main()
