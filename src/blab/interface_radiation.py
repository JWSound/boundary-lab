"""Channel synthesis of full-distribution exterior radiation contributions."""

from dataclasses import dataclass, field

import numpy as np

from blab.live import DEFAULT_CHANNEL_VOLTAGE_V, LiveSolveDataset, excitation_basis_weights
from blab.solve_results.derived import pressure_spl_db
from blab.solve_results.model import INTERFACE_RADIATION_ID
from blab.system_contract import SystemFrequencyResult


@dataclass
class InterfaceRadiationDataset:
    excitation_port_ids: tuple[str, ...]
    excitation_channel_names: np.ndarray
    voltage_excitation_mask: np.ndarray
    source_ids: tuple[str, ...]
    source_names: np.ndarray
    points_m: np.ndarray
    results: dict[float, np.ndarray] = field(default_factory=dict)
    reference_voltages_v: dict[float, float] = field(default_factory=dict)

    def add(self, result: SystemFrequencyResult) -> None:
        quantity = next((q for q in result.quantities if q.id == INTERFACE_RADIATION_ID), None)
        if quantity is None:  # Older results do not invent missing contributions.
            return
        if quantity.axes != ("excitation", "radiation_source", "observation") or quantity.unit != "Pa":
            raise ValueError("Interface radiation requires complex pressure on excitation/source/observation axes.")
        ports = tuple(result.excitation_port_ids)
        sources = tuple(quantity.metadata.get("radiation_source_ids", ()))
        if (
            len(set(ports)) != len(ports)
            or set(ports) != set(self.excitation_port_ids)
            or len(set(sources)) != len(sources)
            or set(sources) != set(self.source_ids)
        ):
            raise ValueError("Interface radiation IDs do not match the prepared solve.")
        points = np.asarray(quantity.metadata.get("points_m", ()), dtype=float)
        if points.shape != self.points_m.shape or not np.allclose(points, self.points_m, rtol=0, atol=1e-7):
            raise ValueError("Interface radiation observation points changed during the solve.")
        values = np.asarray(quantity.values)
        if (
            values.dtype.kind != "c"
            or values.shape != (len(ports), len(sources), len(points))
            or not np.isfinite(values).all()
        ):
            raise ValueError("Interface radiation requires finite complex values matching its IDs.")
        self.results[float(result.freq_hz)] = values[
            np.ix_(
                [ports.index(port) for port in self.excitation_port_ids],
                [sources.index(source) for source in self.source_ids],
                range(len(points)),
            )
        ].astype(np.complex128, copy=True)
        self.reference_voltages_v[float(result.freq_hz)] = float(
            result.diagnostics.get("transducer_reference_voltage_v", np.nan)
        )

    def as_pressure_arrays(self, acoustic: LiveSolveDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        frequencies = sorted(set(self.results).intersection(acoustic.results))
        if not frequencies:
            return None
        rows = []
        for frequency in frequencies:
            weights = excitation_basis_weights(acoustic, frequency, self.excitation_channel_names)
            if np.any(self.voltage_excitation_mask):
                reference = self.reference_voltages_v[frequency]
                if not np.isfinite(reference) or reference <= 0:
                    rows.append(np.full(len(self.source_ids), complex(np.nan, np.nan)))
                    continue
                weights[self.voltage_excitation_mask] *= DEFAULT_CHANNEL_VOLTAGE_V / reference
            rows.append(weights @ self.results[frequency][:, :, 0])
        names = self.source_names.astype(str)
        labels = np.asarray(
            [
                f"{name} [{source}]" if np.count_nonzero(names == name) > 1 else name
                for name, source in zip(names, self.source_ids, strict=True)
            ]
        )
        return np.asarray(frequencies), labels, np.asarray(rows).T

    def as_spl_arrays(self, acoustic: LiveSolveDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        data = self.as_pressure_arrays(acoustic)
        return None if data is None else (data[0], data[1], pressure_spl_db(data[2]))
