"""Project-owned authoring model for observation planes."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np

DEFAULT_PLANE_SIZE_M = 0.25
DEFAULT_PLANE_RESOLUTION_M = 0.01
MIN_PLANE_SIZE_M = 1e-4
MIN_PLANE_RESOLUTION_M = 1e-5
EVALUATION_POINT_WARNING_THRESHOLD = 10_000


class ObservationPlaneType(StrEnum):
    INTERIOR = "interior"
    EXTERIOR = "exterior"
    COMBINED = "combined"


class ObservationPlaneDisplay(StrEnum):
    SPL = "spl"
    PHASE = "phase"
    REAL_PRESSURE = "real_pressure"
    IMAGINARY_PRESSURE = "imaginary_pressure"
    NORMALIZED_SPL = "normalized_spl"


class InteriorRenderingMode(StrEnum):
    SMOOTH_FIELD = "smooth_field"
    ELEMENT_FIELD = "element_field"


@dataclass(frozen=True)
class ObservationPlane:
    """A finite rectangular plane declared before solving.

    Orientation is stored as a unit quaternion in ``(w, x, y, z)`` order.
    Its local X/Y axes span the plane and local Z is its normal.
    """

    id: str
    name: str
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    width_m: float = DEFAULT_PLANE_SIZE_M
    height_m: float = DEFAULT_PLANE_SIZE_M
    resolution_m: float = DEFAULT_PLANE_RESOLUTION_M
    plane_type: ObservationPlaneType = ObservationPlaneType.INTERIOR
    display: ObservationPlaneDisplay = ObservationPlaneDisplay.SPL
    interior_rendering: InteriorRenderingMode = InteriorRenderingMode.SMOOTH_FIELD
    invert_clip_side: bool = False
    response_id: str = "system"
    frequency_hz: float | None = None
    animation_speed_hz: float = 1.0
    pressure_color_limit_pa: float | None = None

    def validated(self) -> ObservationPlane:
        center = _finite_tuple(self.center_m, 3, label="center_m")
        quaternion = normalized_quaternion(self.orientation_wxyz)
        width = _positive_finite(self.width_m, label="width_m", minimum=MIN_PLANE_SIZE_M)
        height = _positive_finite(self.height_m, label="height_m", minimum=MIN_PLANE_SIZE_M)
        resolution = _positive_finite(
            self.resolution_m,
            label="resolution_m",
            minimum=MIN_PLANE_RESOLUTION_M,
        )
        name = str(self.name).strip() or "Observation Plane"
        identifier = str(self.id).strip()
        if not identifier:
            raise ValueError("Observation plane id must not be empty.")
        speed = _positive_finite(self.animation_speed_hz, label="animation_speed_hz", minimum=0.01)
        frequency = None
        if self.frequency_hz is not None:
            frequency = _positive_finite(self.frequency_hz, label="frequency_hz", minimum=1e-9)
        pressure_color_limit = None
        if self.pressure_color_limit_pa is not None:
            pressure_color_limit = _positive_finite(
                self.pressure_color_limit_pa,
                label="pressure_color_limit_pa",
                minimum=1e-12,
            )
        return replace(
            self,
            id=identifier,
            name=name,
            center_m=center,
            orientation_wxyz=quaternion,
            width_m=width,
            height_m=height,
            resolution_m=resolution,
            plane_type=ObservationPlaneType(self.plane_type),
            display=ObservationPlaneDisplay(self.display),
            interior_rendering=InteriorRenderingMode(self.interior_rendering),
            invert_clip_side=bool(self.invert_clip_side),
            response_id=str(self.response_id).strip() or "system",
            frequency_hz=frequency,
            animation_speed_hz=speed,
            pressure_color_limit_pa=pressure_color_limit,
        )

    @property
    def sample_shape(self) -> tuple[int, int]:
        """Number of sample vertices along local X and Y."""

        x_points = int(math.ceil(self.width_m / self.resolution_m)) + 1
        y_points = int(math.ceil(self.height_m / self.resolution_m)) + 1
        return max(x_points, 2), max(y_points, 2)

    @property
    def evaluation_point_count(self) -> int:
        x_points, y_points = self.sample_shape
        return x_points * y_points

    @property
    def warns_about_evaluation_points(self) -> bool:
        return self.evaluation_point_count > EVALUATION_POINT_WARNING_THRESHOLD

    @property
    def local_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        basis = quaternion_matrix(self.orientation_wxyz)
        return basis[:, 0], basis[:, 1], basis[:, 2]

    @property
    def corners_m(self) -> np.ndarray:
        center = np.asarray(self.center_m, dtype=float)
        axis_u, axis_v, _normal = self.local_axes
        half_u = axis_u * (self.width_m * 0.5)
        half_v = axis_v * (self.height_m * 0.5)
        return np.asarray(
            (
                center - half_u - half_v,
                center + half_u - half_v,
                center + half_u + half_v,
                center - half_u + half_v,
            ),
            dtype=float,
        )

    def to_payload(self) -> dict[str, Any]:
        plane = self.validated()
        return {
            "id": plane.id,
            "name": plane.name,
            "center_m": list(plane.center_m),
            "orientation_wxyz": list(plane.orientation_wxyz),
            "width_m": plane.width_m,
            "height_m": plane.height_m,
            "resolution_m": plane.resolution_m,
            "type": plane.plane_type.value,
            "display": plane.display.value,
            "interior_rendering": plane.interior_rendering.value,
            "invert_clip_side": plane.invert_clip_side,
            "response_id": plane.response_id,
            "frequency_hz": plane.frequency_hz,
            "animation_speed_hz": plane.animation_speed_hz,
            "pressure_color_limit_pa": plane.pressure_color_limit_pa,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ObservationPlane | None:
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                id=str(payload.get("id", "")).strip() or uuid.uuid4().hex[:12],
                name=str(payload.get("name", "Observation Plane")),
                center_m=tuple(payload.get("center_m", (0.0, 0.0, 0.0))),
                orientation_wxyz=tuple(payload.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))),
                width_m=float(payload.get("width_m", DEFAULT_PLANE_SIZE_M)),
                height_m=float(payload.get("height_m", DEFAULT_PLANE_SIZE_M)),
                resolution_m=float(payload.get("resolution_m", DEFAULT_PLANE_RESOLUTION_M)),
                plane_type=ObservationPlaneType(payload.get("type", ObservationPlaneType.INTERIOR.value)),
                display=ObservationPlaneDisplay(payload.get("display", ObservationPlaneDisplay.SPL.value)),
                interior_rendering=InteriorRenderingMode(
                    payload.get("interior_rendering", InteriorRenderingMode.SMOOTH_FIELD.value)
                ),
                invert_clip_side=bool(payload.get("invert_clip_side", False)),
                response_id=str(payload.get("response_id", "system")),
                frequency_hz=(None if payload.get("frequency_hz") is None else float(payload["frequency_hz"])),
                animation_speed_hz=float(payload.get("animation_speed_hz", 1.0)),
                pressure_color_limit_pa=(
                    None
                    if payload.get("pressure_color_limit_pa") is None
                    else float(payload["pressure_color_limit_pa"])
                ),
            ).validated()
        except (TypeError, ValueError):
            return None


def new_observation_plane(
    name: str,
    *,
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    width_m: float = DEFAULT_PLANE_SIZE_M,
    height_m: float | None = None,
) -> ObservationPlane:
    return ObservationPlane(
        id=uuid.uuid4().hex[:12],
        name=name,
        center_m=center_m,
        orientation_wxyz=orientation_wxyz,
        width_m=width_m,
        height_m=width_m if height_m is None else height_m,
    ).validated()


def observation_planes_from_payload(payload: object) -> tuple[ObservationPlane, ...]:
    if not isinstance(payload, list):
        return ()
    planes = tuple(plane for item in payload if (plane := ObservationPlane.from_payload(item)) is not None)
    seen: set[str] = set()
    unique = []
    for plane in planes:
        if plane.id in seen:
            continue
        seen.add(plane.id)
        unique.append(plane)
    return tuple(unique)


def normalized_quaternion(values: tuple[float, ...]) -> tuple[float, float, float, float]:
    quaternion = np.asarray(_finite_tuple(values, 4, label="orientation_wxyz"), dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("Observation plane orientation quaternion must not be zero.")
    quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def quaternion_matrix(values: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = normalized_quaternion(values)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def quaternion_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    u, _singular, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            values = (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            values = (
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            values = (
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            )
    return normalized_quaternion(values)


def quaternion_from_axes(
    axis_u: np.ndarray, axis_v: np.ndarray, normal: np.ndarray
) -> tuple[float, float, float, float]:
    return quaternion_from_matrix(np.column_stack((axis_u, axis_v, normal)))


def rotate_observation_plane(plane: ObservationPlane, local_axis: int, angle_deg: float) -> ObservationPlane:
    basis = quaternion_matrix(plane.orientation_wxyz)
    axis = basis[:, int(local_axis)]
    angle = math.radians(float(angle_deg))
    skew = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)),
        dtype=float,
    )
    rotation = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    return replace(plane, orientation_wxyz=quaternion_from_matrix(rotation @ basis))


def _finite_tuple(values: tuple[float, ...], length: int, *, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Observation plane {label} must contain {length} finite numbers.") from exc
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"Observation plane {label} must contain {length} finite numbers.")
    return result


def _positive_finite(value: float, *, label: str, minimum: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"Observation plane {label} must be finite and at least {minimum:g}.")
    return number
