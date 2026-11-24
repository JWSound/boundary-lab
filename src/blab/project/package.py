"""Version-one ZIP project container with a JSON document and optional numerical results."""

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# Readers return ordinary filesystem paths to existing mesh consumers.
_workspace = None


def _asset_fields(project):
    for mesh in project.get("imported_meshes", []):
        yield mesh, ("source_file", "cleaned_file")
    for mesh in (project.get("physical_system") or {}).get("meshes", []):
        yield mesh, ("file",)
    for document in project.get("generator_documents", []):
        artifact = document.get("artifact")
        if artifact:
            yield artifact, ("mesh_path", "cleaned_mesh_path", "reduced_cleaned_mesh_path", "source_path")
            for mesh in artifact.get("meshes", []):
                yield mesh, ("mesh_path",)


def write_package(path, payload, include_results=True, session=None):
    project = copy.deepcopy(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED, allowZip64=True) as archive:
            assets = {}

            def pack(filename):
                source = Path(filename).resolve()
                if source not in assets:
                    member = f"assets/{len(assets)}/{source.name}"
                    archive.write(source, member)
                    assets[source] = member
                return assets[source]

            for record, keys in _asset_fields(project):
                for key in keys:
                    if record.get(key) and Path(record[key]).is_file():
                        record[key] = pack(record[key])
            for doc in project.get("generator_documents", []):
                if doc.get("artifact"):
                    doc["artifact"]["output_dir"] = "."
            archive.writestr("project.json", json.dumps(project, ensure_ascii=False))
            archive.writestr(
                "manifest.json",
                json.dumps({"format": "boundary-lab-project", "version": 1, "include_results": include_results}),
            )
            if include_results and session is not None:
                from blab.project.result_io import write_session

                write_session(archive, session, pack)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_package(path, *, with_results=False, asset_parent=None):
    from blab.project.io import migrate_project_payload, resolve_project_paths

    global _workspace
    if asset_parent is None:
        if _workspace is None:
            _workspace = tempfile.TemporaryDirectory(prefix="boundary-lab-projects-")
        asset_parent = _workspace.name
    root = Path(tempfile.mkdtemp(prefix="project-assets-", dir=asset_parent)).resolve()
    try:
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            if (manifest.get("format"), manifest.get("version")) != ("boundary-lab-project", 1):
                raise ValueError("Unsupported Boundary Lab package version.")
            project = migrate_project_payload(json.loads(archive.read("project.json")))

            def unpack(name):
                if not name.startswith("assets/"):
                    return name
                target = (root / name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Invalid packaged asset path.")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                return str(target)

            for record, keys in _asset_fields(project):
                for key in keys:
                    if record.get(key):
                        record[key] = unpack(record[key])
            session = None
            if with_results and "results.json" in archive.namelist():
                from blab.project.result_io import read_session

                session = read_session(archive, unpack)
            return resolve_project_paths(project, root), manifest["include_results"], session
    except Exception:
        shutil.rmtree(root)
        raise
