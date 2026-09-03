"""Live solve results and the plot-resolution flags derived from them.

Seven modules read and write this state — the solve workflow that produces it,
the plot presenter that draws it, and channels, exports, dialogs, panels and
state sync that interrogate it. Holding it on the window made that sharing
invisible and left the solve workflow unextractable; holding it here makes the
sharing explicit and lets the derivations be tested without a Qt application.

Companion to :mod:`blab.ui.main_window.geometry_store`, which does the same job
for generated geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

from blab.live import (
    AcousticLoadImpedanceDataset,
    ElectricalImpedanceDataset,
    LiveSolveDataset,
    TransducerMotionDataset,
)
from blab.solve_results import SolvedSystem, SolvedSystemBuilder
from blab.ui.result_projection import VisualizationProjection


@dataclass
class SolveSession:
    """Results of the current solve, plus the previous one kept for comparison."""

    #: Results accumulated by the running (or most recent) solve.
    live_dataset: LiveSolveDataset | None = None

    #: Canonical raw quantities accumulated by the active solve.
    result_builder: SolvedSystemBuilder | None = None

    #: Lightweight live transducer motion rows used by the excursion plot.
    transducer_motion: TransducerMotionDataset | None = None

    #: Voltage-basis currents projected into parallel per-channel loads.
    electrical_impedance: ElectricalImpedanceDataset | None = None

    #: Intrinsic coupled acoustic load recovered from the voltage basis.
    acoustic_load_impedance: AcousticLoadImpedanceDataset | None = None

    #: Derived effective driven areas aligned with exterior solver radiator order.
    acoustic_impedance_effective_areas_m2: tuple[float, ...] | None = None

    #: Fluid properties used by the dimensionless impedance projection.
    acoustic_impedance_density_kg_per_m3: float = 1.21
    acoustic_impedance_sound_speed_m_per_s: float = 343.0

    #: Physical-system channels whose grouped basis contains voltage ports only.
    voltage_channel_names: frozenset[str] = frozenset()

    #: Whether the user has requested a maximum-SPL projection for this solve.
    max_spl_requested: bool = False

    #: Immutable canonical snapshot of the most recent complete or partial run.
    solved_system: SolvedSystem | None = None

    #: Whether isobars should be drawn at final rather than live resolution.
    #: Only a completed solve earns the expensive resolution.
    use_final_isobar_resolution: bool = False

    #: Whether the final-resolution isobars have actually been drawn. Contours
    #: may not be captured until they have, or they would freeze mismatched
    #: against the finished solve.
    final_isobar_plots_rendered: bool = False

    #: Snapshot of the last completed solve, overlaid on the next one.
    #: Deliberately outlives :meth:`begin`.
    last_completed_visualization: VisualizationProjection | None = None

    # -- derivations -------------------------------------------------------

    @property
    def solved_count(self) -> int:
        """How many frequencies have results, zero when nothing has run."""
        return 0 if self.live_dataset is None else self.live_dataset.solved_count

    def has_solved_data(self) -> bool:
        """Whether there is anything a user would be upset to lose."""
        return self.solved_count > 0

    # -- mutation ----------------------------------------------------------

    def begin(self) -> None:
        """Discard the previous run's results and drop back to live resolution.

        Leaves :attr:`last_completed_visualization` alone: the point of the
        comparison overlay is to survive into the next solve.
        """
        self.live_dataset = None
        self.result_builder = None
        self.transducer_motion = None
        self.electrical_impedance = None
        self.acoustic_load_impedance = None
        self.acoustic_impedance_effective_areas_m2 = None
        self.acoustic_impedance_density_kg_per_m3 = 1.21
        self.acoustic_impedance_sound_speed_m_per_s = 343.0
        self.voltage_channel_names = frozenset()
        self.max_spl_requested = False
        self.solved_system = None
        self.use_final_isobar_resolution = False
        self.final_isobar_plots_rendered = False

    def forget_comparison(self) -> None:
        """Drop the stored previous solve, so nothing is overlaid."""
        self.last_completed_visualization = None

    def finalize_results(self, *, status: str) -> SolvedSystem | None:
        builder = self.result_builder
        if builder is None or builder.solved_count == 0:
            self.solved_system = None
            return None
        self.solved_system = builder.finalize(status=status)
        self.result_builder = None
        return self.solved_system
