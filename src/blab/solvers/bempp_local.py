"""Local in-process bempp-cl solver backend."""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from blab.solver import HornBEMSolver
from blab.solvers.base import (
    FrequencyResult,
    SolveMetadata,
    SolveRequest,
    SolverCapabilities,
)


class BemppLocalSession:
    def __init__(self, request: SolveRequest):
        self.request = request
        self.solver = HornBEMSolver(request.config)
        self._stop = False

    @property
    def metadata(self) -> SolveMetadata:
        return SolveMetadata(
            polar_angle_deg=self.solver.polar_angles_deg,
            radiator_names=np.asarray(self.solver.radiator_names),
            sphere_metadata=self.solver.sphere_metadata,
        )

    def solve_stream(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[FrequencyResult]:
        def should_stop() -> bool:
            return self._stop or (stop_requested is not None and stop_requested())

        for (
            freq,
            horizontal,
            vertical,
            impedance,
            raw_horizontal,
            raw_vertical,
            sphere_spl,
            channel_names,
            horizontal_pressure,
            vertical_pressure,
            sphere_pressure,
            timings,
        ) in self.solver.solve_frequencies_stream(
            self.request.frequencies_hz,
            stop_requested=should_stop,
        ):
            yield FrequencyResult(
                freq_hz=freq,
                horizontal_spl_norm_db=horizontal,
                vertical_spl_norm_db=vertical,
                impedance=impedance,
                horizontal_spl_db=raw_horizontal,
                vertical_spl_db=raw_vertical,
                sphere_spl_norm_db=sphere_spl,
                channel_names=channel_names,
                horizontal_pressure=horizontal_pressure,
                vertical_pressure=vertical_pressure,
                sphere_pressure=sphere_pressure,
                timings=timings,
            )

    def stop(self) -> None:
        self._stop = True


class BemppLocalBackend:
    backend_id = "local"
    label = "Local process"
    capabilities = SolverCapabilities(
        supports_remote_assets=False,
        supports_parallel_workers=True,
        supports_channel_resynthesis=True,
        is_remote=False,
    )

    def create_session(self, request: SolveRequest) -> BemppLocalSession:
        if request.config.symmetry != "off":
            raise RuntimeError("The Bempp OpenCL CPU backend does not support symmetry acceleration.")
        return BemppLocalSession(request)
