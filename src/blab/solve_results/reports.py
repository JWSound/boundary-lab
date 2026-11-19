"""Desktop calculation adapters for persisted frequency results."""

import numpy as np

from blab.acoustic_impedance import normalization_records, normalize_generalized_impedance
from blab.channel_synthesis import complex_reference_pressure
from blab.config import DEFAULT_CHANNEL_VOLTAGE_V
from blab.live import AcousticLoadImpedanceDataset
from blab.max_spl import calculate_max_spl_curves, max_spl_limits_from_payload, transducer_rated_resistance_ohm
from blab.solve_results.model import (
    DIAPHRAGM_VELOCITY_ID,
    HORIZONTAL_POLAR_DOMAIN_ID,
    HORIZONTAL_POLAR_PRESSURE_ID,
    RADIATION_IMPEDANCE_ID,
    VOICE_COIL_CURRENT_ID,
)


def _ports(run, result):
    ports = {p["id"]: p for p in run.system["excitation_ports"]}
    return [ports[port_id] for port_id in result.excitation_port_ids]


def normalized_impedance(run, hz):
    system = run.system
    if run.manifest["solve_kind"] not in {"exterior_bem", "coupled_bem_fem"}:
        raise ValueError("Normalized acoustic impedance requires an exterior or coupled solve.")
    records = normalization_records(system.get("metadata"))
    region = system["regions"][0]
    density, speed = region["density_kg_per_m3"], region["sound_speed_m_per_s"]
    if run.manifest["solve_kind"] == "exterior_bem":
        result = run.frequency_result(hz, [RADIATION_IMPEDANCE_ID])
        ids = [c["id"] for c in system["components"] if c["kind"] == "ideal_velocity_source"]
        # Exterior solver impedance follows radiator/component ordering.
        ids = list(result.quantities[0].metadata.get("component_ids", ids))
        areas = np.array([records[i].effective_area_m2 for i in ids])
        values = normalize_generalized_impedance(result.quantities[0].values, areas, density, speed)
    else:
        result = run.frequency_result(hz, [DIAPHRAGM_VELOCITY_ID, VOICE_COIL_CURRENT_ID])
        ports = _ports(run, result)
        transducers = [c for c in system["components"] if c["kind"] == "electrodynamic_transducer"]
        ids = [c["id"] for c in transducers]
        areas = np.array([records[i].effective_area_m2 for i in ids])
        dataset = AcousticLoadImpedanceDataset(
            excitation_port_ids=result.excitation_port_ids,
            excitation_port_kinds=np.array([p["kind"] for p in ports]),
            excitation_component_ids=np.array([p["component_id"] for p in ports]),
            transducer_component_ids=np.array(ids),
            transducer_names=np.array([c["name"] for c in transducers]),
            effective_area_m2=areas,
            density_kg_per_m3=density,
            sound_speed_m_per_s=speed,
            **{
                key: np.array([c["parameters"][key] for c in transducers])
                for key in ("bl_n_per_a", "mmd_kg", "cms_m_per_n", "rms_n_s_per_m")
            },
        )
        dataset.add(result)
        arrays = dataset.as_impedance_arrays()
        if arrays is None:
            raise ValueError("Acoustic load recovery is unavailable.")
        values = arrays[2][:, 0] + 1j * arrays[3][:, 0]
    return {
        "component_ids": ids,
        "effective_area_m2": areas.tolist(),
        "density_kg_per_m3": density,
        "sound_speed_m_per_s": speed,
        "unit": "1",
    }, values


def maximum_spl(run, hz):
    result = run.frequency_result(hz, [HORIZONTAL_POLAR_PRESSURE_ID, DIAPHRAGM_VELOCITY_ID])
    pressure, velocity = result.quantities
    ports = _ports(run, result)
    if set(result.excitation_port_ids) != set(run.channel_by_port):
        raise ValueError("Maximum SPL needs the complete saved channel basis.")
    names = np.array([run.channel_by_port[p["id"]] for p in ports])
    eligible = [
        name
        for name in dict.fromkeys(names)
        if all(p["kind"] == "voltage" for p, n in zip(ports, names, strict=True) if n == name)
    ]
    limits = max_spl_limits_from_payload(run.project.get("max_spl_limits_by_channel"))
    channels = np.array([name for name in eligible if name in limits and limits[name].enabled])
    if not channels.size:
        raise ValueError("Maximum SPL requires saved Xmax and Pmax ratings on a voltage-only channel.")
    angles = run.domain_array(HORIZONTAL_POLAR_DOMAIN_ID, "angle_deg")
    on_axis = [complex_reference_pressure(pressure.values[names == n].sum(axis=0), angles, 0) for n in channels]
    components = {c["id"]: c for c in run.system["components"]}
    ids = velocity.metadata.get("component_ids") or run.domain_array(velocity.target_id, "component_id").tolist()
    mapping = run.project.get("component_channel_by_id", {})
    reference = result.diagnostics.get("transducer_reference_voltage_v", DEFAULT_CHANNEL_VOLTAGE_V)
    _, traces, spl = calculate_max_spl_curves(
        frequencies_hz=np.array([result.freq_hz]),
        channel_names=channels,
        on_axis_pressure_pa=np.array(on_axis)[:, None],
        excitation_channel_names=names,
        transducer_channel_names=np.array([mapping.get(i, "main") for i in ids]),
        transducer_resistance_ohm=np.array([transducer_rated_resistance_ohm(components[i]["parameters"]) for i in ids]),
        diaphragm_velocity_m_per_s=velocity.values[None],
        limits_by_channel=limits,
        reference_voltage_v=reference,
    )
    return {
        "channel_names": traces.tolist(),
        "unit": "dB re 20 uPa",
        "reference_voltage_v": reference,
        "ratings": {n: {"xmax_mm": limits[n].xmax_mm, "pmax_w": limits[n].pmax_w} for n in traces},
    }, spl[:, 0]
