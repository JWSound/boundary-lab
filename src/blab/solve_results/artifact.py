"""Read-only access to schema-2 headless result directories."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np

from blab.channel_synthesis import channel_configs_from_payload, excitation_weights
from blab.config import DEFAULT_CHANNEL_VOLTAGE_V
from blab.phasor import SOLVER_PHASOR_CONVENTION, convert_phasor
from blab.project.io import migrate_project_payload
from blab.project.model import ProjectPreferencesState
from blab.solve_results.model import HORIZONTAL_POLAR_DOMAIN_ID, HORIZONTAL_POLAR_PRESSURE_ID
from blab.system_contract import QuantityResult, SystemFrequencyResult


class ResultRun:
    """Read a manifest snapshot and access only its committed frequency files."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest = self.read_json("manifest.json")
        m = self.manifest
        if (m.get("schema"), m.get("schema_version")) != ("boundary-lab-headless-result", 2):
            raise ValueError("Expected a schema-2 Boundary Lab headless result.")
        self.completed = {}
        for hz, done, entry in zip(m["frequencies_hz"], m["completion_mask"], m["results"], strict=True):
            if done:
                if not entry or not np.isclose(entry["freq_hz"], hz, rtol=2e-6, atol=1e-5):
                    raise ValueError(f"Inconsistent manifest entry at {hz} Hz.")
                self.completed[float(hz)] = entry

    def path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Artifact member lies outside the result directory.")
        return path

    def read_json(self, name: str) -> dict:
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    @cached_property
    def domains(self):
        return {d["id"]: d for d in self.read_json(self.manifest["domains_metadata_file"])["domains"]}

    def domain_array(self, domain_id: str, field: str, group: str | None = None):
        domain = self.domains[domain_id]
        if group is None and field in domain["coordinates"] and field in domain["topology"]:
            raise ValueError(f"{field} exists in coordinates and topology. Specify --group.")
        keys = domain[group] if group else domain["coordinates"] | domain["topology"]
        with np.load(self.path(self.manifest["domains_file"]), allow_pickle=False) as archive:
            return archive[keys[field]]

    def metadata(self, hz: float):
        if hz not in self.completed:
            raise ValueError(f"No committed sample at {hz:g} Hz.")
        entry = self.completed[hz]
        meta = self.read_json(entry["metadata_file"])
        if meta["freq_hz"] != entry["freq_hz"] or set(meta["excitation_port_ids"]) != set(
            self.manifest["excitation_port_ids"]
        ):
            raise ValueError(f"Inconsistent frequency metadata at {hz:g} Hz.")
        return meta

    def quantity(self, hz, meta, quantity_id):
        descriptor = next((q for q in meta["quantities"] if q["id"] == quantity_id), None)
        if descriptor is None:
            raise ValueError(f"{quantity_id!r} is unavailable at {hz:g} Hz.")
        with np.load(self.path(self.completed[hz]["arrays_file"]), allow_pickle=False) as archive:
            values = archive[descriptor["key"]]
        if list(values.shape) != descriptor["shape"] or len(descriptor["axes"]) != values.ndim:
            raise ValueError(f"Invalid array dimensions for {quantity_id!r} at {hz:g} Hz.")
        return descriptor, values

    def frequency_result(self, hz, quantity_ids):
        """Load selected canonical quantities in recorded excitation order."""
        meta = self.metadata(hz)
        quantities = []
        for quantity_id in quantity_ids:
            q, values = self.quantity(hz, meta, quantity_id)
            quantities.append(
                QuantityResult(
                    id=q["id"],
                    quantity=q["quantity"],
                    unit=q["unit"],
                    axes=tuple(q["axes"]),
                    metadata=q["metadata"],
                    target_id=q["target_id"],
                    values=convert_phasor(values, self.manifest["phasor_convention"], SOLVER_PHASOR_CONVENTION),
                )
            )
        return SystemFrequencyResult(
            meta["freq_hz"], tuple(quantities), tuple(meta["excitation_port_ids"]), meta["diagnostics"]
        )

    def inspect(self):
        inventory = {}
        for hz in self.completed:
            for q in self.metadata(hz)["quantities"]:
                item = inventory.setdefault(
                    q["id"], {k: q[k] for k in ("quantity", "unit", "axes", "target_id", "shape", "dtype")}
                )
                item.setdefault("frequencies_hz", []).append(hz)
        fields = (
            "status",
            "backend_id",
            "solve_kind",
            "phasor_convention",
            "project_sha256",
            "frequencies_hz",
            "completion_mask",
            "excitation_port_ids",
            "meshes",
            "schema_version",
        )
        return {k: self.manifest[k] for k in fields} | {"quantities": inventory, "domains": self.domains}

    @cached_property
    def project(self):
        return migrate_project_payload(self.read_json(self.manifest["project_file"]))

    @cached_property
    def system(self):
        return self.read_json(self.manifest["compiled_system_file"])

    @cached_property
    def channel_by_port(self):
        mapping = self.project.get("component_channel_by_id", {})
        return {p["id"]: mapping.get(p["component_id"], "main") for p in self.system["excitation_ports"]}

    def weights(self, hz, meta, response, uncorrected):
        ids = tuple(meta["excitation_port_ids"])
        selected = response.removeprefix("channel:")
        all_channels = self.channel_by_port
        if response != "system" and (not response.startswith("channel:") or selected not in all_channels.values()):
            raise ValueError(
                f"Unknown saved response {response!r}. Available channels: {sorted(set(all_channels.values()))}"
            )
        required = {port for port, name in all_channels.items() if response == "system" or name == selected}
        if not required.issubset(ids):
            raise ValueError("This run lacks excitation ports required for the selected response.")
        names = tuple(all_channels[port] for port in ids)
        ports = self.system["excitation_ports"]
        voltage = {all_channels[p["id"]] for p in ports if p["kind"] == "voltage"}
        velocity = {all_channels[p["id"]] for p in ports if p["kind"] != "voltage"}
        prefs = (
            ProjectPreferencesState.from_payload(self.project.get("project_preferences")) or ProjectPreferencesState()
        )
        correct = prefs.normalized_channel_correction and not uncorrected
        angles = horizontal = None
        if correct:
            if not any(q["id"] == HORIZONTAL_POLAR_PRESSURE_ID for q in meta["quantities"]):
                raise ValueError(
                    "Saved normalization requires horizontal polar pressure. Use --uncorrected to omit it."
                )
            _, horizontal = self.quantity(hz, meta, HORIZONTAL_POLAR_PRESSURE_ID)
            angles = self.domain_array(HORIZONTAL_POLAR_DOMAIN_ID, "angle_deg")
        return excitation_weights(
            solve_index=0,
            frequency_hz=meta["freq_hz"],
            response_id=response,
            excitation_ids=ids,
            channel_names_by_excitation=names,
            channel_configs=channel_configs_from_payload(self.project.get("channel_config_by_name")),
            voltage_channel_names=frozenset(voltage - velocity),
            horizontal_pressure_by_frequency=None if horizontal is None else horizontal[None],
            horizontal_angles_deg=angles,
            flat_target_enabled=correct,
            flat_target_reference_angle_deg=prefs.horizontal_normalization_angle,
        )

    def sample(self, hz, quantity_id, *, excitation=None, response=None, uncorrected=False):
        meta = self.metadata(hz)
        q, values = self.quantity(hz, meta, quantity_id)
        axes = list(q["axes"])
        ids = meta["excitation_port_ids"]
        if excitation is not None or response is not None:
            if "excitation" not in axes:
                raise ValueError("Quantity has no excitation axis.")
            axis = axes.index("excitation")
            if values.shape[axis] != len(ids):
                raise ValueError("Excitation axis does not match recorded ports.")
            axes.pop(axis)
            if excitation is not None:
                if excitation not in ids:
                    raise ValueError(f"Unknown excitation {excitation!r}. Available ports: {ids}")
                values = np.take(values, ids.index(excitation), axis=axis)
            else:
                if not np.iscomplexobj(values) or "impedance" in q["quantity"]:
                    raise ValueError("Channel synthesis requires a linear complex response quantity.")
                values = convert_phasor(values, self.manifest["phasor_convention"], SOLVER_PHASOR_CONVENTION)
                weights = self.weights(hz, meta, response, uncorrected)
                if q["quantity"] in {"interface_average_normal_velocity", "interface_radiated_pressure"}:
                    # Desktop interface projections rescale voltage rows to the recorded solve reference.
                    voltage_ids = {p["id"] for p in self.system["excitation_ports"] if p["kind"] == "voltage"}
                    mask = np.array([port in voltage_ids for port in ids])
                    if np.any(mask):
                        reference = meta["diagnostics"].get("transducer_reference_voltage_v")
                        if reference is None or not np.isfinite(reference) or reference <= 0:
                            raise ValueError("Interface synthesis requires the solve reference voltage.")
                        weights[mask] *= DEFAULT_CHANNEL_VOLTAGE_V / reference
                values = np.tensordot(weights, values, axes=(0, axis))
        return q, axes, values, meta
