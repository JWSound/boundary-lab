"""Provider-neutral generated-mesh post-processing helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from blab.generators.base import GeneratedGeometry
from blab.mesh_clean import AREA_TOL, MERGE_TOL, clean_mesh_file


def ensure_reduced_geometry(
    result: GeneratedGeometry,
    *,
    output_path: Path | None = None,
    merge_tol: float = MERGE_TOL,
    area_tol: float = AREA_TOL,
) -> GeneratedGeometry:
    """Materialize a cleaned, unmirrored mesh for native-symmetry solvers."""
    if result.reduced_cleaned_mesh_path is not None and result.reduced_cleaned_mesh_path.exists():
        return result
    reduced_path = output_path or result.mesh_path.with_name(
        f"{result.mesh_path.stem}_clean_reduced{result.mesh_path.suffix}"
    )
    clean_mesh_file(
        str(result.mesh_path),
        str(reduced_path),
        merge_tol=merge_tol,
        area_tol=area_tol,
        mirror_x=False,
        mirror_axes=(),
        binary=False,
    )
    return replace(result, reduced_cleaned_mesh_path=reduced_path)
