"""SI geometry and circular-arc NFR profiles; no Qt or mesher dependencies."""

import math

# key, label, unit, display multiplier, minimum, maximum, default, decimals
FIELDS = {
    "Enclosure": (
        ("width_m", "External width", "mm", 1000, 150, 1500, 320, 1),
        ("height_m", "External height", "mm", 1000, 200, 2000, 500, 1),
        ("depth_m", "External depth", "mm", 1000, 150, 1500, 360, 1),
        ("wall_m", "Wall / baffle thickness", "mm", 1000, 6, 50, 18, 1),
    ),
    "Transducer geometry": (
        ("driver_diameter_m", "Moving diameter (incl. surround)", "mm", 1000, 40, 600, 200, 1),
        ("cone_depth_m", "Cone depth", "mm", 1000, 5, 150, 40, 1),
        ("surround_width_m", "Surround width", "mm", 1000, 2, 50, 12, 1),
        ("cap_diameter_m", "Dust cap diameter", "mm", 1000, 10, 200, 50, 1),
        ("driver_y_m", "Driver centre Y", "mm", 1000, -800, 800, 100, 1),
    ),
    "Port": (
        ("port_length_m", "Total length (incl. lips)", "mm", 1000, 30, 1200, 220, 1),
        ("port_area_m2", "Throat cross-sectional area", "cm²", 10000, 2, 500, 28.27433388, 2),
        ("port_nfr", "Normalized flare rate", "", 1, 0, 0.8, 0.12, 3),
        ("port_roundover_m", "Lip roundover radius", "mm", 1000, 0, 50, 8, 1),
        ("port_wall_m", "Port wall thickness at lip", "mm", 1000, 2, 20, 3, 1),
        ("port_y_m", "Port centre Y", "mm", 1000, -800, 800, -130, 1),
    ),
    "Mesh": (
        ("mesh_size_m", "Cabinet / cavity element size", "mm", 1000, 5, 100, 40, 1),
        ("detail_size_m", "Driver / port element size", "mm", 1000, 2, 40, 12, 1),
    ),
}
DEFAULTS = {f[0]: f[6] / f[3] for fields in FIELDS.values() for f in fields}
DRIVER_DEFAULTS = {
    "re_ohm": 5.8,
    "le_h": 0.0005,
    "bl_n_per_a": 8.0,
    "mmd_kg": 0.035,
    "cms_m_per_n": 0.0004,
    "rms_n_s_per_m": 1.5,
    "motion_profile": "rigid_translation",
    "motion_axis": [0, 0, 1],
}


def port_profile(p):
    """Return (z, radius) points and arc centres for half of a symmetric port.

    NFR = L/(2R). Lip circles are internally tangent to the main flare circle
    and tangent to the mouth plane; L includes both lips. A zero NFR is a
    cylindrical throat. This is geometry, not the STV tuning/SPL optimizer.
    """
    length, nfr, lip = p["port_length_m"], p["port_nfr"], p["port_roundover_m"]
    throat = math.sqrt(p["port_area_m2"] / math.pi)
    half = length / 2
    if nfr == 0:
        join = (-lip, throat)
        mouth = throat + lip
        centre = None
    else:
        radius = half / nfr
        dz = half - lip
        dx = math.sqrt((radius - lip) ** 2 - dz**2)
        mouth = throat + radius - dx
        join = (-half + radius * dz / (radius - lip), throat + radius - radius * dx / (radius - lip))
        centre = (-half, throat + radius)
    return {
        "throat": (-half, throat),
        "join": join,
        "mouth": (0.0, mouth),
        "flare_centre": centre,
        "lip_centre": (-lip, mouth),
    }


def validate(source):
    if set(source) != set(DEFAULTS):
        raise ValueError("Unrecognized enclosure settings; restore the plugin's supported source fields.")
    p = {}
    for fields in FIELDS.values():
        for key, label, unit, factor, low, high, _, _ in fields:
            value = source[key]
            if type(value) not in (float, int) or not math.isfinite(value) or not low <= value * factor <= high:
                raise ValueError(f"{label} must be between {low} and {high} {unit}.")
            p[key] = float(value)
    w, h, d, t = (p[k] for k in ("width_m", "height_m", "depth_m", "wall_m"))
    length, lip = p["port_length_m"], p["port_roundover_m"]
    if 0 < p["port_nfr"] < 0.001:
        raise ValueError("NFR must be zero (straight) or at least 0.001 for curved CAD geometry.")
    if lip >= length / 4:
        raise ValueError("Lip roundover must be less than one quarter of the port length.")
    profile = port_profile(p)
    port_outer = profile["mouth"][1] + p["port_wall_m"]
    driver = p["driver_diameter_m"] / 2
    cap = p["cap_diameter_m"] / 2
    if cap >= driver - p["surround_width_m"]:
        raise ValueError("Dust cap must fit inside the cone, clear of the surround.")
    if cap / 2 >= p["cone_depth_m"]:
        raise ValueError("Cone depth must exceed one quarter of the dust cap diameter.")
    clearance = 0.003
    for label, radius, y in (("Driver", driver, p["driver_y_m"]), ("Port", port_outer, p["port_y_m"])):
        if radius + clearance >= w / 2 - t or abs(y) + radius + clearance >= h / 2 - t:
            raise ValueError(f"{label} must fit inside the enclosure walls with 3 mm clearance.")
    if abs(p["driver_y_m"] - p["port_y_m"]) <= driver + port_outer + clearance:
        raise ValueError("Driver and port must not overlap; leave at least 3 mm clearance.")
    if length <= t + lip + clearance or length + 2 * profile["mouth"][1] >= d - t:
        raise ValueError("Port must pass through the baffle and leave one mouth diameter clear of the rear wall.")
    if p["cone_depth_m"] + clearance >= d - t:
        raise ValueError("Cone intersects the rear wall.")
    if p["detail_size_m"] > p["mesh_size_m"]:
        raise ValueError("Driver / port element size must not exceed the cavity element size.")
    return p
