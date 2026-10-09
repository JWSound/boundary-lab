"""Cancellable backend and scoped physical-system assignments."""

import json
import subprocess
import sys
from pathlib import Path
from threading import Event
from uuid import uuid4

from blab.generators import GeneratedGeometry, GeneratedMesh, GenerationResponse, generated_mesh_id
from blab.generators.base import GenerationCancelledError

from .parameters import DRIVER_DEFAULTS, port_profile, validate


def configuration_patch(request):
    prefix = request.document_id + "/"
    fem, bem = (generated_mesh_id(request.document_id, key) for key in ("interior", "exterior"))
    system = request.configuration.get("physical_system") or {}
    existing_regions = system.get("regions", [])
    exterior = next((r["id"] for r in existing_regions if r["kind"] == "unbounded_air"), prefix + "exterior")
    interior = prefix + "interior"
    regions = []
    if not any(r["id"] == interior for r in existing_regions):
        regions.append(
            dict(
                id=interior,
                name=f"{request.mesh_name} interior",
                kind="bounded_air",
                mesh_ids=[fem],
                volume_groups=[dict(mesh_id=fem, dimension=3, name="air")],
            )
        )
    if not any(r["id"] == exterior for r in existing_regions):
        regions.append(dict(id=exterior, name="Exterior air", kind="unbounded_air", mesh_ids=[bem]))
    boundaries = []
    for mesh, region, side, wall in ((fem, interior, "inside", "inner_wall"), (bem, exterior, "outside", "outer_wall")):
        for group, kind in ((wall, "rigid"), ("driver", "moving"), ("port_mouth", "interface")):
            boundaries.append(
                dict(
                    id=prefix + side + "/" + group,
                    name=f"{request.mesh_name} {side} {group.replace('_', ' ')}",
                    region_id=region,
                    group=dict(mesh_id=mesh, dimension=2, name=group),
                    kind=kind,
                )
            )
    component_id = prefix + "driver"
    component = dict(
        id=component_id,
        name=f"{request.mesh_name} driver",
        kind="electrodynamic_transducer",
        boundary_ids=[prefix + side + "/driver" for side in ("inside", "outside")],
    )
    if not any(c["id"] == component_id for c in system.get("components", [])):
        component["parameters"] = DRIVER_DEFAULTS.copy()
    patch = {
        "regions": {"upsert": regions},
        "surface_assignments": {"upsert": boundaries},
        "interfaces": {
            "upsert": [
                dict(
                    id=prefix + "port_interface",
                    name=f"{request.mesh_name} port exit",
                    bounded_boundary_id=prefix + "inside/port_mouth",
                    unbounded_boundary_id=prefix + "outside/port_mouth",
                )
            ]
        },
        "component_assignments": {"upsert": [component]},
        "excitation_ports": {
            "upsert": [
                dict(
                    id=prefix + "voltage",
                    name=f"{request.mesh_name} voltage",
                    component_id=component_id,
                    kind="voltage",
                )
            ]
        },
    }
    if component_id not in request.configuration.get("component_channels", {}):
        patch["component_channels"] = {"upsert": [{"id": component_id, "channel": "main"}]}
    return patch


class EnclosureSession:
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

    def generate(self, *, status_callback=None, stop_requested=None):
        def check_cancelled():
            if self.cancelled.is_set() or (stop_requested and stop_requested()):
                self.stop()
                raise GenerationCancelledError("Enclosure generation cancelled.")

        check_cancelled()
        p = validate(self.request.source)
        if self.request.configuration.get("symmetry_config", {}).get("mode", "off") != "off":
            raise ValueError("Vented Enclosure requires Symmetry Off in the mesh settings.")
        output = self.request.run_root.resolve() / ("vented_enclosure_" + uuid4().hex)
        output.mkdir(parents=True)
        source = output / "source.json"
        source.write_text(json.dumps(p, indent=2), encoding="utf-8")
        if status_callback:
            status_callback("Meshing enclosure, driver and port (FEM + BEM)")
        log_path = output / "meshing.log"
        with log_path.open("w", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                [sys.executable, "-I", "-B", "-X", "utf8", str(Path(__file__).with_name("worker.py")), str(source)],
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                while True:
                    check_cancelled()
                    try:
                        code = self.process.wait(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                check_cancelled()
            finally:
                if self.process.poll() is None:
                    self.process.terminate()
                self.process.wait()
                self.process = None
        if code:
            details = log_path.read_text(encoding="utf-8")[-2000:]
            raise ValueError(f"Enclosure meshing failed. Details: {log_path}\n{details}")
        metadata = json.loads((output / "mesh_info.json").read_text(encoding="utf-8"))
        metadata.update(
            generator_version="0.1.0",
            units="m",
            source=p,
            port_mouth_diameter_m=2 * port_profile(p)["mouth"][1],
            model="Rigid cabinet, zero-thickness rigidly translating diaphragm, linear air",
        )
        geometry = GeneratedGeometry(
            self.request.provider_id,
            output,
            None,
            (),
            source_path=source,
            meshes=(
                GeneratedMesh("interior", "fem_volume", mesh_path=output / "interior.msh"),
                GeneratedMesh("exterior", "bem_surface", mesh_path=output / "exterior.msh"),
            ),
            provider_metadata=metadata,
        )
        return GenerationResponse(self.request.request_id, geometry, configuration_patch(self.request))


class EnclosureBackend:
    def create_session(self, request):
        return EnclosureSession(request)

    def restore(self, document):
        return None  # The host restores cached assembly resources without executing plugins.


def create_backend(**options):
    return EnclosureBackend()
