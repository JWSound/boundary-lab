"""Portable, display-only OBJ assets in the speaker package coordinate frame."""

import math
from pathlib import Path


def viewport_model_members(path: Path, scale: float) -> tuple[dict[str, bytes], dict]:
    """Validate an expanded source-frame OBJ and package its color materials."""
    path = Path(path)
    if path.suffix.lower() != ".obj" or not path.is_file():
        raise ValueError("Choose an existing .obj viewport model.")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Viewport model scale must be positive and finite.")
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        output = []
        counts = {"v": 0, "vt": 0, "vn": 0}
        faces = []
        libraries = []
        minimum = [math.inf] * 3
        maximum = [-math.inf] * 3
        for line in lines:
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue
            kind = parts[0]
            if kind == "mtllib":
                name = " ".join(parts[1:])
                candidate = path.parent / name
                # Prefer the complete filename (including spaces), then multiple filenames.
                libraries.extend([candidate] if candidate.is_file() else [path.parent / p for p in parts[1:]])
                continue
            if kind in counts:
                counts[kind] += 1
                values = [float(v) for v in parts[1:]]
                if not values or not all(math.isfinite(v) for v in values):
                    raise ValueError("Non-finite or empty vertex data.")
                if kind in ("v", "vn"):
                    if len(values) < 3:
                        raise ValueError("Vertices and normals require three coordinates.")
                    x, y, z = values[:3]
                    factor = scale if kind == "v" else 1.0
                    if kind == "v":
                        for axis, value in enumerate((x * factor, z * factor, -y * factor)):
                            minimum[axis] = min(minimum[axis], value)
                            maximum[axis] = max(maximum[axis], value)
                    line = f"{kind} {x * factor:.12g} {z * factor:.12g} {-y * factor:.12g}"
            elif kind == "f":
                if len(parts) < 4:
                    raise ValueError("Faces require at least three vertices.")
                faces.append((parts[1:], dict(counts)))
            output.append(line)
        if not counts["v"] or not faces:
            raise ValueError("The viewport model must contain vertices and faces.")
        for face, preceding in faces:
            for token in face:
                indices = token.split("/")
                if len(indices) > 3 or not indices[0]:
                    raise ValueError("Invalid face index.")
                for kind, value in zip(("v", "vt", "vn"), indices):
                    if value:
                        index = int(value)
                        if index == 0 or index > counts[kind] or -index > preceding[kind]:
                            raise ValueError("Face index is outside the vertex data.")
        if not any(p.is_file() for p in libraries) and path.with_suffix(".mtl").is_file():
            libraries = [path.with_suffix(".mtl")]
        warnings = []
        material_lines = []
        detected = []
        for library in dict.fromkeys(libraries):
            if library.resolve().parent != path.parent.resolve():
                raise ValueError("Viewport materials must be in the OBJ directory.")
            if not library.is_file():
                warnings.append(f"Missing material file: {library.name}; using default materials.")
                continue
            detected.append(library.name)
            for line in library.read_text(encoding="utf-8-sig").splitlines():
                parts = line.split()
                if parts and (parts[0].lower().startswith("map_") or parts[0].lower() in {"bump", "disp", "decal", "refl", "norm"}):
                    warnings.append("Texture maps are not supported; material colors will be used.")
                    continue
                material_lines.append(line)
        members = {}
        descriptor = {
            "path": "viewport/model.obj", "format": "obj", "unit": "m",
            "coordinate_frame": "package", "geometry_expansion": "full",
            "source_scale_to_m": scale, "source_name": path.name,
            "materials_detected": detected, "warnings": list(dict.fromkeys(warnings)),
            "bounds_min_m": minimum, "bounds_max_m": maximum,
        }
        if material_lines:
            output.insert(0, "mtllib materials.mtl")
            members["viewport/materials.mtl"] = ("\n".join(material_lines) + "\n").encode("utf-8")
            descriptor["material_path"] = "viewport/materials.mtl"
        members["viewport/model.obj"] = ("\n".join(output) + "\n").encode("utf-8")
        return members, descriptor
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid viewport model: {exc}") from exc
