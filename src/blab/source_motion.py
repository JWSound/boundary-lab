"""Qt-free prescribed-source motion validation shared by authoring and solvers."""

import math


def prescribed_source_parameters(parameters: dict, *, symmetry: str = "off", exterior: bool = True) -> dict:
    """Return canonical parameters; absent motion means legacy surface-normal motion."""
    unsupported = set(parameters) - {"motion_profile", "motion_axis", "boundary_motion_weights"}
    if unsupported:
        raise ValueError("Unsupported prescribed-source parameters: " + ", ".join(sorted(unsupported)))
    result = dict(parameters)
    profile = result.get("motion_profile", "uniform_normal")
    if not isinstance(profile, str):
        raise ValueError("Prescribed-source motion_profile must be a string.")
    if profile in {"uniform", "uniform_normal"}:
        if "motion_axis" in result:
            raise ValueError("motion_axis requires motion_profile=rigid_translation.")
        result.pop("motion_profile", None)
        return result
    if profile != "rigid_translation":
        raise ValueError("Prescribed-source motion_profile must be uniform_normal or rigid_translation.")
    if not exterior:
        raise ValueError("Axial prescribed-velocity sources require an exterior-only BEM system.")
    axis = result.get("motion_axis")
    if (
        not isinstance(axis, (list, tuple))
        or len(axis) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in axis
        )
    ):
        raise ValueError("motion_axis must contain three finite numbers.")
    scale = max(abs(value) for value in axis)
    if scale == 0:
        raise ValueError("motion_axis must have nonzero length.")
    scaled = [value / scale for value in axis]
    norm = math.sqrt(sum(value * value for value in scaled))
    axis = [value / norm for value in scaled]
    indices = {"off": (), "x": (0,), "xy": (0, 1)}[symmetry]
    if any(abs(axis[index]) > 1e-8 for index in indices):
        raise ValueError("Prescribed-source motion_axis must lie in the physical symmetry planes.")
    result["motion_axis"] = axis
    return result
