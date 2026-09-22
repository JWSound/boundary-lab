import json
import shutil
import sys
from pathlib import Path

import pytest

from blab.generators import catalog as catalog_module
from blab.generators.base import GenerationRequest, complete_generation
from blab.generators.catalog import ProviderCatalog
from blab.generators.registry import create_generator, generator_info


def package(root, name="example", **overrides):
    folder = root / name
    folder.mkdir(parents=True)
    manifest = (
        dict(
            manifest_version=1,
            id="test.example",
            name="Example",
            version="1.0",
            provider_api=1,
            source_schema_version=1,
            backend="backend:create_backend",
        )
        | overrides
    )
    (folder / "provider.json").write_text(json.dumps(manifest))
    (folder / "backend.py").write_text("raise RuntimeError('code was executed')\n")
    return folder


def test_scan_and_disabled_lookup_never_execute_code(tmp_path, monkeypatch):
    package(tmp_path)
    catalog = ProviderCatalog([tmp_path])
    monkeypatch.setattr(catalog_module, "_catalog", catalog)
    assert not catalog.scan()[0].error
    assert catalog.status(catalog.packages[0]) == "Disabled"
    with pytest.raises(ValueError, match="disabled"):
        create_generator("test.example")
    catalog.enabled.add("test.example")
    assert generator_info("test.example").label == "Example"
    with pytest.raises(ValueError, match="code was executed"):
        create_generator("test.example")
    catalog.scan()
    assert "code was executed" in catalog.packages[0].error


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider_api": 2},
        {"manifest_version": True},
        {"source_schema_version": 0},
        {"backend": "../backend:factory"},
        {"id": "ATH"},
        {"mesh_scale_factor": 0},
        {"default_source": []},
    ],
)
def test_incompatible_manifests_remain_visible(tmp_path, overrides):
    package(tmp_path, **overrides)
    records = ProviderCatalog([tmp_path]).scan()
    assert len(records) == 1 and records[0].error


def test_duplicates_and_builtin_override_are_conflicts(tmp_path):
    package(tmp_path, "first")
    package(tmp_path, "second")
    package(tmp_path, "third", id="ath")
    records = ProviderCatalog([tmp_path]).scan()
    assert all("Conflicting" in record.error for record in records)


def test_relative_imports_are_private_and_loaded_changes_require_restart(tmp_path):
    first = package(tmp_path, "first", id="test.first")
    second = package(tmp_path, "second", id="test.second")
    for folder, value in ((first, 1), (second, 2)):
        (folder / "backend.py").write_text("from .helper import VALUE\ndef create_backend(): return VALUE\n")
        (folder / "helper.py").write_text(f"VALUE = {value}\n")
    catalog = ProviderCatalog([tmp_path], enabled=("test.first", "test.second"))
    before = list(sys.path)
    catalog.scan()
    assert catalog.factory("test.first", "backend")() == 1
    assert catalog.factory("test.second", "backend")() == 2
    assert sys.path == before
    (first / "helper.py").write_text("VALUE = 3\n")
    catalog.scan()
    with pytest.raises(ValueError, match="restart"):
        catalog.factory("test.first", "backend")
    assert catalog.factory("test.second", "backend")() == 2


def test_example_package_generates_and_stages_without_mesh_files(tmp_path, monkeypatch):
    from blab.generators.application import stage_generation
    from blab.project.model import ProjectDocument, new_generator_document

    root = Path(__file__).resolve().parents[1] / "examples" / "geometry_providers" / "example_box"
    shutil.copytree(root, tmp_path / "example_box")
    catalog = ProviderCatalog([tmp_path], enabled=("example.box",))
    catalog.scan()
    monkeypatch.setattr(catalog_module, "_catalog", catalog)
    source = catalog.source_defaults("example.box")
    document = new_generator_document("box", provider_id="example.box", source=source)
    project = ProjectDocument(generator_documents=(document,), active_generator_document_id=document.id)
    request = GenerationRequest("example.box", document.id, "box", source, tmp_path / "unused", "box")
    response = create_generator("example.box").create_session(request).generate()
    completed = complete_generation(request, response)
    candidate = stage_generation(project, {}, completed)
    assert candidate.physical_system.components
    assert candidate.physical_system.meshes[0].mesh_data.digest == response.geometry.mesh_data.digest
    assert not (tmp_path / "unused").exists()
