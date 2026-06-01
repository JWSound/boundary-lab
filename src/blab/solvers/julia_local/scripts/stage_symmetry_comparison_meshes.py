from __future__ import annotations

from pathlib import Path

import meshio

from blab.mesh_clean import AREA_TOL, MERGE_TOL, clean_mesh_file


ROOT = Path(__file__).resolve().parents[1]
TEST_MESHES = ROOT / "test_meshes"
STAGED = TEST_MESHES / "staged_symmetry"


def stage_mesh(source_name: str, mode: str) -> None:
    source = TEST_MESHES / source_name
    stem = source.stem
    reduced = STAGED / f"{stem}_reduced_clean.msh"
    full = STAGED / f"{stem}_full_clean.msh"
    axes = tuple(mode)

    STAGED.mkdir(parents=True, exist_ok=True)
    clean_mesh_file(
        str(source),
        str(reduced),
        merge_tol=MERGE_TOL,
        area_tol=AREA_TOL,
        mirror_x=False,
        mirror_axes=(),
        binary=False,
    )
    clean_mesh_file(
        str(source),
        str(full),
        merge_tol=MERGE_TOL,
        area_tol=AREA_TOL,
        mirror_x=False,
        mirror_axes=axes,
        binary=False,
    )

    reduced_mesh = meshio.read(reduced)
    full_mesh = meshio.read(full)
    print(
        {
            "source": source_name,
            "mode": mode,
            "reduced": str(reduced.relative_to(ROOT)),
            "full": str(full.relative_to(ROOT)),
            "reduced_vertices": len(reduced_mesh.points),
            "full_vertices": len(full_mesh.points),
            "reduced_faces": len(reduced_mesh.cells_dict["triangle"]),
            "full_faces": len(full_mesh.cells_dict["triangle"]),
        }
    )


def main() -> None:
    stage_mesh("sample_half.msh", "x")
    stage_mesh("sample_quarter.msh", "xy")


if __name__ == "__main__":
    main()
