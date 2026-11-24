"""Lossless result-session storage. Only explicitly supported data records can be constructed."""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from blab.config import ChannelConfig, CrossoverConfig
from blab.interface_radiation import InterfaceRadiationDataset
from blab.live import (
    AcousticLoadImpedanceDataset,
    ElectricalImpedanceDataset,
    InterfaceVelocityDataset,
    LiveSolveDataset,
    TransducerMotionDataset,
)
from blab.physical_model import CompiledPhysicalSystem
from blab.solve_results.model import ResultDomain, SolvedQuantity, SolvedSystem, SolveProvenance
from blab.solvers.base import FrequencyResult, FrequencySolveTimings, SolverDiagnostics
from blab.system_contract import compiled_system_from_dict, compiled_system_to_dict

_RECORDS = {
    cls.__name__: cls
    for cls in (
        ChannelConfig,
        CrossoverConfig,
        InterfaceRadiationDataset,
        LiveSolveDataset,
        ElectricalImpedanceDataset,
        InterfaceVelocityDataset,
        TransducerMotionDataset,
        AcousticLoadImpedanceDataset,
        ResultDomain,
        SolveProvenance,
        SolvedSystem,
        SolvedQuantity,
        FrequencyResult,
        FrequencySolveTimings,
        SolverDiagnostics,
    )
}
SESSION_FIELDS = (
    "solved_system",
    "live_dataset",
    "electrical_impedance",
    "acoustic_load_impedance",
    "interface_velocity",
    "interface_radiation",
    "transducer_motion",
    "voltage_channel_names",
    "acoustic_impedance_effective_areas_m2",
    "acoustic_impedance_density_kg_per_m3",
    "acoustic_impedance_sound_speed_m_per_s",
    "use_final_isobar_resolution",
)


def write_session(archive, session, pack):
    index = 0

    def encode(value):
        nonlocal index
        if isinstance(value, np.ndarray):
            member = f"arrays/{index}.npy"
            index += 1
            with archive.open(member, "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, value, allow_pickle=False)
            return {"array": member}
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, CompiledPhysicalSystem):
            raw = compiled_system_to_dict(value)
            for mesh in raw["meshes"]:
                if mesh.get("file") and Path(mesh["file"]).is_file():
                    mesh["file"] = pack(mesh["file"])
            return {"compiled": raw}
        if type(value).__name__ in _RECORDS:
            return {
                "record": type(value).__name__,
                "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)},
            }
        if isinstance(value, dict):
            return {"mapping": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, frozenset)):
            return {type(value).__name__: [encode(v) for v in value]}
        if isinstance(value, list):
            return [encode(v) for v in value]
        return value

    archive.writestr("results.json", json.dumps({k: encode(getattr(session, k)) for k in SESSION_FIELDS}))


def read_session(archive, unpack):
    def decode(value):
        if isinstance(value, list):
            return [decode(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "array" in value:
            with archive.open(value["array"]) as stream:
                return np.lib.format.read_array(stream, allow_pickle=False)
        if "compiled" in value:
            raw = value["compiled"]
            for mesh in raw["meshes"]:
                if mesh.get("file"):
                    mesh["file"] = unpack(mesh["file"])
            return compiled_system_from_dict(raw)
        if "mapping" in value:
            return {decode(k): decode(v) for k, v in value["mapping"]}
        if "tuple" in value:
            return tuple(map(decode, value["tuple"]))
        if "frozenset" in value:
            return frozenset(map(decode, value["frozenset"]))
        return _RECORDS[value["record"]](**{k: decode(v) for k, v in value["fields"].items()})

    raw = json.loads(archive.read("results.json"))
    return {k: decode(raw[k]) for k in SESSION_FIELDS}
