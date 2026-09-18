"""A coarse closed box with one driven face, generated entirely in memory.

This demonstrates the API; subdivide the faces for acoustically useful meshes.
"""

import math

import numpy as np

from blab.config import RadiatorConfig
from blab.generators.base import GeneratedGeometry, GenerationCancelledError, GenerationResponse
from blab.mesh_data import MeshData


class BoxSession:
    def __init__(self, request):
        self.request = request
        self.stopped = False

    def stop(self):
        self.stopped = True

    def generate(self, *, status_callback=None, stop_requested=None):
        if self.stopped or (stop_requested and stop_requested()):
            raise GenerationCancelledError("Stopped")
        dimensions = [float(self.request.source[key]) for key in ("width_m", "height_m", "depth_m")]
        if any(not math.isfinite(value) or value <= 0 for value in dimensions):
            raise ValueError("Box dimensions must be positive and finite.")
        if status_callback:
            status_callback("Building box mesh")
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float) * dimensions
        triangles = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                     [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                     [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
        mesh = MeshData(points=points, cells=[("triangle", triangles)],
                        physical_tags=[[1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]],
                        physical_names={"wall": (1, 2), "driver": (2, 2)})
        geometry = GeneratedGeometry(
            provider_id=self.request.provider_id, output_dir=self.request.run_root,
            mesh_path=None, mesh_data=mesh,
            radiators=(RadiatorConfig("driver", 2, mesh=self.request.mesh_name),),
        )
        return GenerationResponse(request_id=self.request.request_id, geometry=geometry)


class BoxBackend:
    def create_session(self, request):
        return BoxSession(request)

    def restore(self, document):
        # Memory artifacts are restored by Boundary Lab without invoking this method.
        return None


def create_backend(**options):
    return BoxBackend()
