from pathlib import Path

import pytest

from blab.viewport_model import viewport_model_members


def test_viewport_rotation_units_materials_and_negative_indices(tmp_path: Path):
    obj = tmp_path / "cabinet.obj"
    obj.write_text("mtllib cabinet finish.mtl\nv 100 200 300\nv 0 0 0\nv 100 0 0\n"
                   "vn 0 0 1\nusemtl wood\nf -3//1 -2//1 -1//1\n")
    (tmp_path / "cabinet finish.mtl").write_text("newmtl wood\nKd 0.5 0.2 0.1\nmap_Kd texture.png\n")
    members, descriptor = viewport_model_members(obj, 0.01)
    text = members["viewport/model.obj"].decode()
    assert "v 1 3 -2" in text
    assert "vn 0 1 -0" in text
    assert "f -3//1 -2//1 -1//1" in text
    assert "mtllib materials.mtl" in text
    assert "map_Kd" not in members["viewport/materials.mtl"].decode()
    assert descriptor["warnings"] == ["Texture maps are not supported; material colors will be used."]
    assert descriptor["bounds_max_m"] == [1, 3, 0]


def test_viewport_material_fallback_and_missing_library(tmp_path: Path):
    obj = tmp_path / "cabinet.obj"
    obj.write_text("mtllib missing.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    _, descriptor = viewport_model_members(obj, 1)
    assert "Missing material" in descriptor["warnings"][0]
    (tmp_path / "cabinet.mtl").write_text("newmtl default\nKd 0.2 0.2 0.2\n")
    _, descriptor = viewport_model_members(obj, 1)
    assert descriptor["materials_detected"] == ["cabinet.mtl"]
    assert descriptor["warnings"] == []


@pytest.mark.parametrize("text", ["v nan 0 0", "v 0 0 0\nf 1 2 3", "v 0 0 0\nf 0 1 1"])
def test_viewport_rejects_invalid_geometry(tmp_path: Path, text: str):
    obj = tmp_path / "bad.obj"
    obj.write_text(text)
    with pytest.raises(ValueError, match="Invalid viewport model"):
        viewport_model_members(obj, 1)


def test_viewport_rejects_external_material(tmp_path: Path):
    obj = tmp_path / "bad.obj"
    obj.write_text("mtllib ../outside.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    with pytest.raises(ValueError, match="OBJ directory"):
        viewport_model_members(obj, 1)
