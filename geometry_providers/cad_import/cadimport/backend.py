"""Cancellable meshing in a worker process, and the System entries of the resulting assembly."""

from threading import Event
from uuid import uuid4

from blab.cad_import import CadImportResult
from blab.generators import GeneratedGeometry, GeneratedMesh, GenerationResponse, generated_mesh_id
from blab.generators.base import GenerationCancelledError

from .jobs import mesh_in_worker
from .parameters import settings, step_path


def configuration_patch(request, result: CadImportResult):
    """Regions and interfaces every time; boundaries only for groups that have none yet."""
    doc = request.document_id
    system = request.configuration.get("physical_system") or {}
    regions = {r["id"]: r for r in system.get("regions", [])}
    boundaries = {b["id"] for b in system.get("boundaries", [])}
    exterior_region = next((r["id"] for r in regions.values() if r["kind"] == "unbounded_air"), f"{doc}/exterior")
    region_upserts, boundary_upserts, interfaces = [], [], []
    interface_names = {i.name for i in result.geometry.interfaces}
    for mesh in result.meshes:
        mesh_id = generated_mesh_id(doc, mesh.id)
        if mesh.purpose == "fem_volume":
            region_id = f"{doc}/{mesh.id}"
            region = dict(
                id=region_id,
                kind="bounded_air",
                mesh_ids=[mesh_id],
                volume_groups=[dict(mesh_id=mesh_id, dimension=3, name=name) for name in mesh.volume_groups],
            )
            if region_id not in regions:
                region["name"] = f"{request.mesh_name} {', '.join(mesh.volume_groups)}"
            region_upserts.append(region)
        else:
            region_id = exterior_region
            if region_id not in regions:
                region_upserts.append(dict(id=region_id, name="Exterior air", kind="unbounded_air", mesh_ids=[mesh_id]))
        for group in mesh.surface_groups:
            boundary_id = f"{doc}/{mesh.id}/{group}"
            if boundary_id in boundaries:
                continue
            boundary_upserts.append(
                dict(
                    id=boundary_id,
                    name=f"{request.mesh_name} {mesh.id} {group}",
                    region_id=region_id,
                    group=dict(mesh_id=mesh_id, dimension=2, name=group),
                    kind="interface" if group in interface_names else "rigid",
                )
            )
    for interface in result.geometry.interfaces:
        interfaces.append(
            dict(
                id=f"{doc}/interface/{interface.name}",
                name=f"{request.mesh_name} {interface.name}",
                bounded_boundary_id=f"{doc}/{interface.mesh_id}/{interface.name}",
                unbounded_boundary_id=f"{doc}/exterior/{interface.name}",
            )
        )
    patch = {
        "regions": {"upsert": region_upserts},
        "surface_assignments": {"upsert": boundary_upserts},
        "interfaces": {"upsert": interfaces},
    }
    return _remove_stale(patch, system, doc, result.meshes)


def _remove_stale(patch, system, doc, meshes):
    """Remove this design's boundaries whose face group is gone, with what depends on them, in ``patch``.

    Their interfaces go, and so do bounded regions whose mesh is gone. Components keep their
    other boundaries; a component left without any is removed with its excitation ports.
    """
    # Every mesh this design generates has an ID starting with this prefix.
    design_mesh_prefix = generated_mesh_id(doc, "")
    current_mesh_ids = {generated_mesh_id(doc, m.id) for m in meshes}
    current_groups = {(generated_mesh_id(doc, m.id), g) for m in meshes for g in m.surface_groups}
    design_boundaries = [
        b for b in system.get("boundaries", []) if b["group"]["mesh_id"].startswith(design_mesh_prefix)
    ]
    design_boundary_ids = {b["id"] for b in design_boundaries}
    stale = {
        b["id"] for b in design_boundaries if (b["group"]["mesh_id"], b["group"].get("name")) not in current_groups
    }
    if not stale:
        return patch
    components, removed_components = [], []
    for component in system.get("components", []):
        kept = [b for b in component["boundary_ids"] if b not in stale]
        if len(kept) == len(component["boundary_ids"]) or not set(component["boundary_ids"]) <= design_boundary_ids:
            continue  # untouched, or shared with other designs: the host rejects the dangling reference
        if kept:
            components.append({"id": component["id"], "boundary_ids": kept})
        else:
            removed_components.append(component["id"])
    patch["surface_assignments"]["remove"] = sorted(stale)
    patch["interfaces"]["remove"] = sorted(
        i["id"] for i in system.get("interfaces", []) if {i["bounded_boundary_id"], i["unbounded_boundary_id"]} & stale
    )
    patch["regions"]["remove"] = sorted(
        r["id"]
        for r in system.get("regions", [])
        if r["kind"] == "bounded_air"
        and r["mesh_ids"]
        and all(m.startswith(design_mesh_prefix) and m not in current_mesh_ids for m in r["mesh_ids"])
    )
    patch["component_assignments"] = {"upsert": components, "remove": removed_components}
    patch["excitation_ports"] = {
        "remove": [p["id"] for p in system.get("excitation_ports", []) if p["component_id"] in removed_components]
    }
    return patch


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
        result = mesh_in_worker(path, cad_settings, output, cancelled=cancelled, started=self._set_process)
        geometry = result.geometry
        metadata = dict(
            generator_version="0.1.0",
            units="m",
            step_path=str(path),
            target_size_m=result.target_size_m,
            bbox_m=[list(corner) for corner in geometry.bbox_m],
            suggested_symmetry=geometry.symmetry,
            warnings=list(geometry.warnings),
        )
        generated = GeneratedGeometry(
            self.request.provider_id,
            output,
            None,
            (),
            source_path=path,
            meshes=tuple(GeneratedMesh(m.id, m.purpose, mesh_path=m.path) for m in result.meshes),
            provider_metadata=metadata,
            reduced=cad_settings.symmetry != "off",
        )
        return GenerationResponse(self.request.request_id, generated, configuration_patch(self.request, result))


class CadImportBackend:
    def create_session(self, request):
        return CadImportSession(request)

    def restore(self, document):
        return None  # The host restores cached assembly resources without executing plugins.


def create_backend(**options):
    return CadImportBackend()
