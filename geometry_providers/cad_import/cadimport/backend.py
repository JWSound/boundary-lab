"""Cancellable meshing in a worker process, and the System entries of the resulting assembly."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from threading import Event
from uuid import uuid4

import blab
from blab.generators import GeneratedGeometry, GeneratedMesh, GenerationResponse, generated_mesh_id
from blab.generators.base import GenerationCancelledError

from .parameters import settings, step_path

BLAB_PATH = str(Path(blab.__file__).resolve().parents[1])


def run_worker(job, output, *, cancelled, started=lambda process: None):
    """Run ``worker.py`` on ``job`` in ``output``; ``cancelled()`` is polled and terminates it."""
    job_path = output / "job.json"
    job_path.write_text(json.dumps(job | {"blab_path": BLAB_PATH, "output": str(output)}, indent=2), encoding="utf-8")
    log_path = output / "meshing.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-X", "utf8", str(Path(__file__).with_name("worker.py")), str(job_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        started(process)
        try:
            while True:
                if cancelled():
                    raise GenerationCancelledError("CAD import cancelled.")
                try:
                    code = process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()
            started(None)
    if code:
        details = log_path.read_text(encoding="utf-8")
        # The pipeline's own rejections are the last line of the traceback.
        lines = [line for line in details.strip().splitlines() if line.startswith("ValueError: ")]
        if lines:
            raise ValueError(lines[-1].removeprefix("ValueError: "))
        raise ValueError(f"CAD import meshing failed. Details: {log_path}\n{details[-2000:]}")


def configuration_patch(request, info):
    """Regions and interfaces every time; boundaries only for groups that have none yet."""
    doc = request.document_id
    system = request.configuration.get("physical_system") or {}
    regions = {r["id"]: r for r in system.get("regions", [])}
    boundaries = {b["id"] for b in system.get("boundaries", [])}
    exterior_region = next((r["id"] for r in regions.values() if r["kind"] == "unbounded_air"), f"{doc}/exterior")
    region_upserts, boundary_upserts, interfaces = [], [], []
    interface_names = {i["name"] for i in info["geometry"]["interfaces"]}
    region_of = {}
    for mesh in info["meshes"]:
        mesh_id = generated_mesh_id(doc, mesh["id"])
        if mesh["purpose"] == "fem_volume":
            region_id = f"{doc}/{mesh['id']}"
            region = dict(
                id=region_id,
                kind="bounded_air",
                mesh_ids=[mesh_id],
                volume_groups=[dict(mesh_id=mesh_id, dimension=3, name=name) for name in mesh["volume_groups"]],
            )
            if region_id not in regions:
                region["name"] = f"{request.mesh_name} {', '.join(mesh['volume_groups'])}"
            region_upserts.append(region)
        else:
            region_id = exterior_region
            if region_id not in regions:
                region_upserts.append(dict(id=region_id, name="Exterior air", kind="unbounded_air", mesh_ids=[mesh_id]))
        region_of[mesh["id"]] = region_id
        for group in mesh["surface_groups"]:
            boundary_id = f"{doc}/{mesh['id']}/{group}"
            if boundary_id in boundaries:
                continue
            boundary_upserts.append(
                dict(
                    id=boundary_id,
                    name=f"{request.mesh_name} {mesh['id']} {group}",
                    region_id=region_id,
                    group=dict(mesh_id=mesh_id, dimension=2, name=group),
                    kind="interface" if group in interface_names else "rigid",
                )
            )
    for interface in info["geometry"]["interfaces"]:
        interfaces.append(
            dict(
                id=f"{doc}/interface/{interface['name']}",
                name=f"{request.mesh_name} {interface['name']}",
                bounded_boundary_id=f"{doc}/{interface['mesh_id']}/{interface['name']}",
                unbounded_boundary_id=f"{doc}/exterior/{interface['name']}",
            )
        )
    return {
        "regions": {"upsert": region_upserts},
        "surface_assignments": {"upsert": boundary_upserts},
        "interfaces": {"upsert": interfaces},
    }


class CadImportSession:
    def __init__(self, request):
        self.request = request
        self.cancelled = Event()
        self.process = None

    def stop(self):
        self.cancelled.set()
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass

    def _set_process(self, process):
        self.process = process

    def generate(self, *, status_callback=None, stop_requested=None):
        def cancelled():
            return self.cancelled.is_set() or bool(stop_requested and stop_requested())

        if cancelled():
            raise GenerationCancelledError("CAD import cancelled.")
        path = step_path(self.request.source)
        cad_settings = settings(self.request.source, self.request.configuration)
        output = self.request.run_root.resolve() / ("cad_import_" + uuid4().hex)
        output.mkdir(parents=True)
        if status_callback:
            status_callback(f"Meshing {path.name} (target element size {cad_settings.target_size_m * 1000:.1f} mm)")
        job = {"mode": "mesh", "step_path": str(path), "settings": asdict(cad_settings)}
        run_worker(job, output, cancelled=cancelled, started=self._set_process)
        info = json.loads((output / "mesh_info.json").read_text(encoding="utf-8"))
        geometry = info["geometry"]
        metadata = dict(
            generator_version="0.1.0",
            units="m",
            step_path=str(path),
            target_size_m=info["target_size_m"],
            bbox_m=geometry["bbox_m"],
            suggested_symmetry=geometry["symmetry"],
            warnings=geometry["warnings"],
        )
        result = GeneratedGeometry(
            self.request.provider_id,
            output,
            None,
            (),
            source_path=path,
            meshes=tuple(GeneratedMesh(m["id"], m["purpose"], mesh_path=output / m["file"]) for m in info["meshes"]),
            provider_metadata=metadata,
            reduced=cad_settings.symmetry != "off",
        )
        return GenerationResponse(self.request.request_id, result, configuration_patch(self.request, info))


class CadImportBackend:
    def create_session(self, request):
        return CadImportSession(request)

    def restore(self, document):
        return None  # The host restores cached assembly resources without executing plugins.


def create_backend(**options):
    return CadImportBackend()
