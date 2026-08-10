"""Post-solve exterior-field evaluation using retained BEM boundary traces."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from blab.solvers.beat_engine_backend import DEFAULT_BEAT_ENGINE_CUDA_PROJECT, _get_julia_worker
from blab.solvers.coupled_backend import DEFAULT_COUPLED_CPU_PROJECT, DEFAULT_COUPLED_SOLVER_SCRIPT
from blab.solvers.registry import normalize_backend_id

_FIELD_ARRAYS_FILENAME = "field-arrays.bin"
_FIELD_RESULT_FILENAME = "field-result.bin"
_BINARY_ARRAY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BemFieldEvaluationRequest:
    domain_key: str
    points_m: np.ndarray
    boundary_points_m: np.ndarray
    boundary_triangles: np.ndarray
    boundary_pressure: np.ndarray
    boundary_normal_derivative: np.ndarray
    frequency_hz: float
    sound_speed_m_per_s: float
    symmetry: str = "off"


def evaluate_bem_field(
    request: BemFieldEvaluationRequest,
    *,
    backend_id: str,
    julia_executable: str = "julia",
) -> np.ndarray:
    """Evaluate one synthesized BEM response on arbitrary exterior points."""

    normalized_backend = normalize_backend_id(backend_id)
    if normalized_backend not in {"beat_cpu", "beat_cuda"}:
        raise ValueError("Retained coupled BEM fields require the BEAT Engine CPU or CUDA backend.")
    julia_project = (
        DEFAULT_BEAT_ENGINE_CUDA_PROJECT if normalized_backend == "beat_cuda" else DEFAULT_COUPLED_CPU_PROJECT
    )
    worker = _get_julia_worker(
        julia_executable=julia_executable,
        solver_script=DEFAULT_COUPLED_SOLVER_SCRIPT,
        julia_threads=4 if normalized_backend == "beat_cuda" else 8,
        julia_project=julia_project,
    )
    with tempfile.TemporaryDirectory(prefix="blab-bem-field-") as temp_dir:
        request_dir = Path(temp_dir)
        request_path = request_dir / "request.json"
        _write_evaluation_request(request_path, request, backend_id=normalized_backend)
        values = None
        for event in worker.submit(request_path, operation="bem_field"):
            event_type = str(event.get("type", ""))
            if event_type == "field_result":
                if "values_binary" in event:
                    values = _read_binary_array(request_dir, event["values_binary"])
                else:
                    values = _complex_array_from_wire(event["values"])
            elif event_type == "failed":
                raise RuntimeError(str(event.get("error", "Exterior field evaluation failed.")))
        if values is None:
            raise RuntimeError("The BEAT Engine field worker returned no exterior pressure values.")
        if values.dtype.kind != "c" or values.shape != (np.asarray(request.points_m).shape[0],):
            raise RuntimeError("The BEAT Engine field worker returned an invalid exterior pressure array.")
        return values


def _write_evaluation_request(
    request_path: Path,
    request: BemFieldEvaluationRequest,
    *,
    backend_id: str,
) -> None:
    payload, arrays = _validated_evaluation_request(request, backend_id=backend_id)
    array_path = request_path.with_name(_FIELD_ARRAYS_FILENAME)
    payload["binary_array_schema_version"] = _BINARY_ARRAY_SCHEMA_VERSION
    payload["binary_arrays"] = _write_binary_arrays(array_path, arrays)
    payload["binary_result_file"] = _FIELD_RESULT_FILENAME
    request_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _validated_evaluation_request(
    request: BemFieldEvaluationRequest,
    *,
    backend_id: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    points = np.asarray(request.points_m, dtype=np.float32)
    boundary_points = np.asarray(request.boundary_points_m, dtype=np.float32)
    triangles = np.asarray(request.boundary_triangles, dtype=np.int64)
    pressure = np.asarray(request.boundary_pressure)
    normal = np.asarray(request.boundary_normal_derivative)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Exterior evaluation points must have shape (point, 3).")
    if boundary_points.ndim != 2 or boundary_points.shape[1] != 3:
        raise ValueError("BEM boundary points must have shape (bem_node, 3).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("BEM boundary triangles must have shape (bem_face, 3).")
    if pressure.shape != (boundary_points.shape[0],):
        raise ValueError("BEM boundary pressure must contain one value per boundary node.")
    if normal.shape != (triangles.shape[0],):
        raise ValueError("BEM boundary normal derivative must contain one value per boundary face.")
    if pressure.dtype.kind != "c" or normal.dtype.kind != "c":
        raise ValueError("BEM boundary traces must be complex arrays.")
    if not np.isfinite(request.frequency_hz) or request.frequency_hz <= 0.0:
        raise ValueError("Exterior field frequency must be finite and greater than zero.")
    if not np.isfinite(request.sound_speed_m_per_s) or request.sound_speed_m_per_s <= 0.0:
        raise ValueError("Exterior-region sound speed must be finite and greater than zero.")
    symmetry = str(request.symmetry).strip().lower()
    if symmetry not in {"off", "x", "xy"}:
        raise ValueError("Exterior field symmetry must be off, x, or xy.")
    dtype = np.result_type(pressure.dtype, normal.dtype)
    if dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        dtype = np.dtype(np.complex64)
    domain_key = str(request.domain_key).strip()
    if not domain_key:
        raise ValueError("Exterior field requests require a non-empty domain cache key.")
    return (
        {
            "domain_key": domain_key,
            "frequency_hz": float(request.frequency_hz),
            "sound_speed_m_per_s": float(request.sound_speed_m_per_s),
            "symmetry": symmetry,
            "bem_backend": "cuda" if backend_id == "beat_cuda" else "cpu",
            "precision": "float64" if dtype == np.dtype(np.complex128) else "float32",
            "quadrature_order": 2,
        },
        {
            "points_m": np.ascontiguousarray(points, dtype=np.dtype("<f4")),
            "boundary_points_m": np.ascontiguousarray(boundary_points, dtype=np.dtype("<f4")),
            "boundary_triangles": np.ascontiguousarray(triangles, dtype=np.dtype("<i8")),
            "boundary_pressure": np.ascontiguousarray(pressure, dtype=dtype.newbyteorder("<")),
            "boundary_normal_derivative": np.ascontiguousarray(normal, dtype=dtype.newbyteorder("<")),
        },
    )


def _write_binary_arrays(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    descriptors: dict[str, dict[str, object]] = {}
    with path.open("wb") as handle:
        for name, values in arrays.items():
            array = np.ascontiguousarray(values)
            offset = handle.tell()
            _write_array_bytes(handle, array)
            descriptors[name] = {
                "file": path.name,
                "offset": offset,
                "nbytes": int(array.nbytes),
                "dtype": array.dtype.name,
                "shape": list(array.shape),
                "order": "C",
                "byte_order": "little",
            }
    return descriptors


def _write_array_bytes(handle: BinaryIO, array: np.ndarray) -> None:
    view = memoryview(array)
    if not view.contiguous:
        raise ValueError("Binary field arrays must be contiguous.")
    handle.write(view.cast("B"))


def _read_binary_array(directory: Path, raw: object) -> np.ndarray:
    if not isinstance(raw, dict):
        raise RuntimeError("The BEAT Engine field worker returned an invalid binary array descriptor.")
    filename = str(raw.get("file", ""))
    if not filename or Path(filename).name != filename:
        raise RuntimeError("The BEAT Engine field worker returned an invalid binary result path.")
    try:
        dtype = np.dtype(str(raw["dtype"])).newbyteorder("<")
        shape = tuple(int(value) for value in raw["shape"])
        offset = int(raw.get("offset", 0))
        nbytes = int(raw["nbytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("The BEAT Engine field worker returned an invalid binary array descriptor.") from exc
    count = int(np.prod(shape, dtype=np.int64))
    if count < 0 or offset < 0 or nbytes != count * dtype.itemsize:
        raise RuntimeError("The BEAT Engine field worker returned inconsistent binary array metadata.")
    path = directory / filename
    if not path.is_file() or path.stat().st_size < offset + nbytes:
        raise RuntimeError("The BEAT Engine field worker did not write the complete binary field result.")
    return np.fromfile(path, dtype=dtype, count=count, offset=offset).reshape(shape)


def _complex_array_from_wire(raw: dict) -> np.ndarray:
    dtype = np.dtype(str(raw["dtype"]))
    shape = tuple(int(value) for value in raw["shape"])
    real_dtype = np.empty((), dtype=dtype).real.dtype
    real = np.asarray(raw["real"], dtype=real_dtype)
    imag = np.asarray(raw["imag"], dtype=real_dtype)
    return (real + 1j * imag).astype(dtype, copy=False).reshape(shape)


__all__ = ["BemFieldEvaluationRequest", "evaluate_bem_field"]
