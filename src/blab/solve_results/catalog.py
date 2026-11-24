"""Reconstruct desktop plot datasets from committed frequency records."""

import numpy as np

from blab.balloon import BalloonPrepConfig, prepare_balloon_data
from blab.channel_synthesis import channel_configs_from_payload
from blab.live import ElectricalImpedanceDataset, LiveSolveDataset
from blab.phasor import convert_phasor
from blab.postprocess import PrepConfig
from blab.project.model import ProjectPreferencesState
from blab.solve_results.derived import phase_deg, pressure_spl_db
from blab.solve_results.model import (
    HORIZONTAL_POLAR_DOMAIN_ID,
    HORIZONTAL_POLAR_PRESSURE_ID,
    INTERFACE_RADIATION_ID,
    INTERFACE_VELOCITY_ID,
    SPHERE_DOMAIN_ID,
    SPHERE_PRESSURE_ID,
    VERTICAL_POLAR_DOMAIN_ID,
    VERTICAL_POLAR_PRESSURE_ID,
    VOICE_COIL_CURRENT_ID,
)
from blab.solvers.base import FrequencyResult
from blab.spinorama import compute_spinorama_from_planes

CATALOG_METRICS = (
    "on-axis-frequency-response",
    "group-delay",
    "horizontal-isobar",
    "vertical-isobar",
    "spinorama",
    "electrical-impedance",
    "real-input-power",
    "interface-velocity",
    "interface-radiation",
    "balloon",
)


def _traces(frequencies, names, **series):
    return {
        "frequencies_hz": np.asarray(frequencies).tolist(),
        "trace_names": np.asarray(names).tolist(),
        "axes": ["trace", "frequency"],
        "quantities": {name: {"unit": unit, "values": values} for name, (unit, values) in series.items()},
    }


def _electrical_dataset(run, ports, voltage_channels):
    drivers = [c for c in run.system["components"] if c["kind"] == "electrodynamic_transducer"]
    if not drivers or not voltage_channels:
        raise ValueError("Electrical reports require electrodynamic drivers on voltage-only channels.")
    return ElectricalImpedanceDataset(
        excitation_port_ids=tuple(p["id"] for p in ports),
        excitation_channel_names=np.array([run.channel_by_port[p["id"]] for p in ports]),
        excitation_component_ids=np.array([p["component_id"] for p in ports]),
        transducer_component_ids=np.array([c["id"] for c in drivers]),
        physical_driver_orbit_counts=np.array([c["parameters"].get("physical_driver_orbit_count", 1) for c in drivers]),
        channel_names=np.array(voltage_channels),
    )


def _grouped_pressure(run, hz, meta, quantity_id, channels, names):
    q, values = run.quantity(hz, meta, quantity_id)
    if q["axes"] != ["excitation", "observation"] or values.shape[0] != len(channels):
        raise ValueError(f"Unexpected pressure basis dimensions for {quantity_id}.")
    values = convert_phasor(values, run.manifest["phasor_convention"])
    return np.stack([values[channels == name].sum(axis=0) for name in names])


def _interface_report(run, frequencies, metric, uncorrected):
    radiation = metric == "interface-radiation"
    qid = INTERFACE_RADIATION_ID if radiation else INTERFACE_VELOCITY_ID
    rows = []
    domain_id = None
    for hz in frequencies:
        q, axes, values, _ = run.sample(hz, qid, response="system", uncorrected=uncorrected)
        expected = ["radiation_source", "observation"] if radiation else ["interface"]
        if axes != expected or (radiation and values.shape[-1] != 1):
            raise ValueError("Interface result dimensions differ from the desktop plot.")
        if domain_id is None:
            domain_id = q["target_id"]
            ids = run.domain_array(domain_id, "source_id" if radiation else "interface_id").tolist()
            names = run.domain_array(domain_id, "name").tolist()
            points = q["metadata"].get("points_m")
        elif domain_id != q["target_id"] or (radiation and points != q["metadata"].get("points_m")):
            raise ValueError("Interface geometry changed between frequency samples.")
        order = q["metadata"]["radiation_source_ids" if radiation else "interface_ids"]
        if set(order) != set(ids) or len(order) != len(ids) or values.shape[0] != len(ids):
            raise ValueError("Interface quantity identifiers differ from the recorded domain.")
        rows.append(np.asarray(values).reshape(-1)[[order.index(i) for i in ids]])
    data = np.stack(rows, axis=1)
    labels = [f"{n} [{i}]" if names.count(n) > 1 else n for i, n in zip(ids, names, strict=True)]
    hz = [run.completed[f]["freq_hz"] for f in frequencies]
    if radiation:
        return _traces(
            hz,
            labels,
            spl_db=("dB re 20 uPa", pressure_spl_db(data)),
            phase_deg=("deg", phase_deg(data)),
            real_pressure=("Pa", data.real),
            imaginary_pressure=("Pa", data.imag),
        ) | {"source_ids": ids, "points_m": points}
    return _traces(hz, labels, rms_velocity=("m/s", np.abs(data))) | {"interface_ids": ids}


def catalog_report(run, metric, frequencies, *, uncorrected=False, spherical=False):
    frequencies = sorted(set(frequencies))
    if metric.startswith("interface-"):
        return _interface_report(run, frequencies, metric, uncorrected)
    ports = run.system["excitation_ports"]
    if set(run.manifest["excitation_port_ids"]) != set(run.channel_by_port):
        raise ValueError("Desktop plot synthesis requires the complete saved excitation basis.")
    names = list(dict.fromkeys(run.channel_by_port[p["id"]] for p in ports))
    voltage = [n for n in names if all(p["kind"] == "voltage" for p in ports if run.channel_by_port[p["id"]] == n)]
    prefs = ProjectPreferencesState.from_payload(run.project.get("project_preferences")) or ProjectPreferencesState()
    electrical = metric in {"electrical-impedance", "real-input-power"}
    circuit = _electrical_dataset(run, ports, voltage) if electrical else None
    polar = not electrical or (metric == "real-input-power" and prefs.normalized_channel_correction and not uncorrected)
    if polar and HORIZONTAL_POLAR_DOMAIN_ID not in run.domains:
        hint = " Use --uncorrected to omit flat-target correction." if electrical else ""
        raise ValueError("This report requires retained horizontal polar observations." + hint)
    angles = run.domain_array(HORIZONTAL_POLAR_DOMAIN_ID, "angle_deg") if polar else np.array([])
    both = metric in {"spinorama", "horizontal-isobar", "vertical-isobar"}
    if both and not np.array_equal(angles, run.domain_array(VERTICAL_POLAR_DOMAIN_ID, "angle_deg")):
        raise ValueError("Horizontal and vertical polar angles must match.")
    exterior = next((r for r in run.system["regions"] if r["kind"] == "unbounded_air"), {})
    dataset = LiveSolveDataset(
        angles,
        radiator_names=np.array([], dtype=str),
        channel_configs=channel_configs_from_payload(run.project["channel_config_by_name"]),
        flat_target_normalization_enabled=prefs.normalized_channel_correction and not uncorrected,
        flat_target_reference_angle_deg=prefs.horizontal_normalization_angle,
        polar_observation_distance_m=prefs.polar_observation_distance_m,
        exterior_sound_speed_m_per_s=exterior.get("sound_speed_m_per_s", 343.0),
        voltage_channel_names=frozenset(voltage),
    )
    if spherical or metric == "balloon":
        if SPHERE_DOMAIN_ID not in run.domains:
            raise ValueError("This report requires project observations with spherical sampling enabled.")
        dataset.sphere_r_distance_m = run.domain_array(SPHERE_DOMAIN_ID, "r_distance_m")
        dataset.sphere_theta_polar_rad = run.domain_array(SPHERE_DOMAIN_ID, "theta_polar_rad")
        dataset.sphere_phi_azimuth_rad = run.domain_array(SPHERE_DOMAIN_ID, "phi_azimuth_rad")
    for hz in frequencies:
        if circuit is not None:
            circuit.add(run.frequency_result(hz, [VOICE_COIL_CURRENT_ID]))
            if metric == "electrical-impedance":
                continue
        meta = run.metadata(hz)
        channels = np.array([run.channel_by_port[i] for i in meta["excitation_port_ids"]])
        empty = np.empty((len(names), 0), dtype=np.complex64)
        horizontal = _grouped_pressure(run, hz, meta, HORIZONTAL_POLAR_PRESSURE_ID, channels, names) if polar else empty
        vertical = _grouped_pressure(run, hz, meta, VERTICAL_POLAR_PRESSURE_ID, channels, names) if both else empty
        sphere = (
            _grouped_pressure(run, hz, meta, SPHERE_PRESSURE_ID, channels, names)
            if spherical or metric == "balloon"
            else None
        )
        dataset.add(
            FrequencyResult(
                meta["freq_hz"],
                np.empty(0),
                np.empty(0),
                np.empty((0, 2)),
                channel_names=np.array(names),
                horizontal_pressure=horizontal,
                vertical_pressure=vertical,
                sphere_pressure=sphere,
            )
        )
    if metric == "electrical-impedance":
        f, names, magnitude, phase = circuit.as_impedance_arrays()
        return _traces(f, names, magnitude=("ohm", magnitude), phase_deg=("deg", phase))
    if metric == "real-input-power":
        f, names, power = circuit.as_power_arrays(dataset)
        return _traces(f, names, real_power=("W", power))
    return _acoustic_report(dataset, prefs, metric, spherical)


def _acoustic_report(dataset, prefs, metric, spherical):
    if metric == "group-delay":
        result = dataset.as_group_delay_arrays()
        if result is None:
            raise ValueError(
                "Group delay requires at least three distinct solved frequencies and horizontal polar data."
            )
        f, names, delay = result
        return _traces(f, names, group_delay=("ms", delay))
    if metric == "on-axis-frequency-response":
        f, names, spl, phase = dataset.as_channel_on_axis_export_arrays()
        total_spl, total_phase = dataset.as_summed_on_axis_export_arrays()
        return _traces(
            f,
            ["Sum", *names],
            spl_db=("dB re 20 uPa", np.vstack((total_spl, spl))),
            propagation_aligned_phase=("deg", np.vstack((total_phase, phase))),
        )
    if metric == "balloon":
        raw = dataset.as_balloon_raw_bundle()
        surface = prepare_balloon_data(raw, BalloonPrepConfig(prefs.spl_min_db, prefs.spl_max_db))
        return {
            "frequencies_hz": raw["freq_hz"].tolist(),
            "axes": ["frequency", "direction"],
            "directions_xyz": surface["directions_xyz"].tolist(),
            "triangle_indices": surface["triangle_indices"].tolist(),
            "r_distance_m": raw["r_distance_m"].tolist(),
            "clip_db": [prefs.spl_min_db, prefs.spl_max_db],
            "quantities": {"normalized_spl": {"unit": "dB", "values": raw["spl_norm"]}},
        }
    if metric == "spinorama":
        f, angles, h, v = dataset.as_raw_polar_arrays()
        curves = compute_spinorama_from_planes(
            f,
            angles,
            h,
            v,
            horizontal_reference_angle_deg=prefs.spin_horizontal_reference_angle,
            vertical_reference_angle_deg=prefs.spin_vertical_reference_angle,
            spherical_spl_relative_db=dataset.as_balloon_raw_bundle()["spl_norm"] if spherical else None,
            use_native_plane_resolution=spherical,
        )
        labels, values = zip(*(*curves.spl_curves(), *curves.di_curves()), strict=True)
        return _traces(f, labels, level=("dB", np.stack(values))) | {
            "sound_power_method": "sphere" if spherical else "planes"
        }
    arrays = dataset.as_visualization_dataset(
        PrepConfig(
            angle_samples=1000,
            freq_samples=500,
            octave_smoothing=prefs.polar_smoothing,
            hor_ref_angle=prefs.horizontal_normalization_angle,
            vert_ref_angle=prefs.vertical_normalization_angle,
            min_db=prefs.spl_min_db,
            max_db=prefs.spl_max_db,
            normalize_polar=True,
            auto_db_span=False,
        )
    )
    return {
        "frequencies_hz": arrays["isobar_freq_hz"].tolist(),
        "angles_deg": arrays["isobar_angle_deg"].tolist(),
        "axes": ["angle", "frequency"],
        "clip_db": [prefs.spl_min_db, prefs.spl_max_db],
        "contour_step_db": prefs.isobar_contour_step_db,
        "quantities": {"normalized_spl": {"unit": "dB", "values": arrays[metric.replace("-", "_") + "_db"]}},
    }
