"""Export rendered plot figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_plot_png(figure: Any, output_path: str | Path, *, dpi: int | float) -> Path:
    path = Path(output_path)
    if path.suffix == "":
        path = path.with_suffix(".png")
    figure.savefig(path, dpi=dpi)
    return path
