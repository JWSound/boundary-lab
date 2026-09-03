"""Compatibility adapter from canonical exterior solves to legacy BEM backends."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterator

import numpy as np

from blab.config import SimulationConfig
from blab.solve_results.legacy import legacy_excitation_ids, legacy_result_to_system_result
from blab.solvers.base import FrequencyResult, SolveMetadata, SolveRequest
from blab.solvers.registry import create_backend, normalize_backend_id
from blab.system_contract import SystemFrequencyResult, SystemSolveRequest


@dataclass(frozen=True)
class ExteriorCompatibilityOptions:
    """Runtime-only inputs needed by a legacy exterior backend."""

    config: SimulationConfig
    backend_id: str
    excitation_port_id_by_channel: tuple[tuple[str, str], ...]
    server_url: str = "http://127.0.0.1:8765"
    server_access_token: str = field(default="", compare=False, repr=False)


@dataclass(frozen=True)
class AdaptedExteriorFrequencyResult:
    """Canonical persistence result paired with its legacy live projection."""

    canonical: SystemFrequencyResult
    live: FrequencyResult


class ExteriorCompatibilitySession:
    """Expose a legacy exterior backend through the physical-system contract."""

    def __init__(
        self,
        request: SystemSolveRequest,
        options: ExteriorCompatibilityOptions,
    ) -> None:
        self.request = request
        self.options = options
        backend = create_backend(
            normalize_backend_id(options.backend_id),
            server_url=options.server_url,
            server_access_token=options.server_access_token,
        )
        self._session = backend.create_session(
            SolveRequest(
                options.config,
                np.asarray(request.frequencies_hz, dtype=np.float32),
                status_callback=request.status_callback,
            )
        )

    @property
    def metadata(self) -> SolveMetadata:
        return self._session.metadata

    def solve_stream(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[AdaptedExteriorFrequencyResult]:
        port_id_by_channel = dict(self.options.excitation_port_id_by_channel)
        for live_result in self._session.solve_stream(stop_requested=stop_requested):
            channel_names = legacy_excitation_ids(live_result)
            unknown = [name for name in channel_names if name not in port_id_by_channel]
            if unknown:
                raise ValueError(
                    "Legacy exterior backend returned unplanned excitation channels: " + ", ".join(unknown)
                )
            excitation_port_ids = tuple(port_id_by_channel[name] for name in channel_names)
            if excitation_port_ids != self.request.excitation_port_ids:
                raise ValueError(
                    "Legacy exterior backend changed the planned channel excitation order; "
                    f"expected {self.request.excitation_port_ids}, received {excitation_port_ids}."
                )
            canonical = replace(
                legacy_result_to_system_result(live_result),
                excitation_port_ids=excitation_port_ids,
            )
            yield AdaptedExteriorFrequencyResult(canonical=canonical, live=live_result)

    def stop(self) -> None:
        self._session.stop()


__all__ = [
    "AdaptedExteriorFrequencyResult",
    "ExteriorCompatibilityOptions",
    "ExteriorCompatibilitySession",
]
