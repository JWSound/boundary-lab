"""Typed presentation projections for live solve results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blab.config import ChannelConfig
from blab.live import (
    AcousticLoadImpedanceDataset,
    ElectricalImpedanceDataset,
    LiveSolveDataset,
    TransducerMotionDataset,
)
from blab.max_spl import MaxSplLimit
from blab.postprocess import PrepConfig
from blab.spinorama import SpinoramaCurves, compute_spinorama_from_planes


@dataclass(frozen=True)
class ProjectionOptions:
    angle_samples: int
    freq_samples: int
    octave_smoothing: int | float | None
    horizontal_reference_angle_deg: float
    vertical_reference_angle_deg: float
    spin_horizontal_reference_angle_deg: float
    spin_vertical_reference_angle_deg: float
    min_db: float
    max_db: float


@dataclass(frozen=True)
class IsobarProjection:
    freq_hz: np.ndarray
    angle_deg: np.ndarray
    horizontal_db: np.ndarray
    vertical_db: np.ndarray
    clip_min_db: float
    clip_max_db: float

    def snapshot(self) -> IsobarProjection:
        return IsobarProjection(
            freq_hz=np.asarray(self.freq_hz).copy(),
            angle_deg=np.asarray(self.angle_deg).copy(),
            horizontal_db=np.asarray(self.horizontal_db).copy(),
            vertical_db=np.asarray(self.vertical_db).copy(),
            clip_min_db=self.clip_min_db,
            clip_max_db=self.clip_max_db,
        )


@dataclass(frozen=True)
class ImpedanceProjection:
    freq_hz: np.ndarray
    radiator_names: np.ndarray
    real: np.ndarray
    imaginary: np.ndarray

    def snapshot(self) -> ImpedanceProjection:
        return ImpedanceProjection(
            freq_hz=np.asarray(self.freq_hz).copy(),
            radiator_names=np.asarray(self.radiator_names).copy(),
            real=np.asarray(self.real).copy(),
            imaginary=np.asarray(self.imaginary).copy(),
        )


@dataclass(frozen=True)
class PolarResponseProjection:
    freq_hz: np.ndarray
    angle_deg: np.ndarray
    horizontal_spl_db: np.ndarray
    vertical_spl_db: np.ndarray
    channel_on_axis_names: np.ndarray | None
    channel_on_axis_spl_db: np.ndarray | None
    on_axis_phase_deg: np.ndarray | None
    channel_on_axis_phase_deg: np.ndarray | None
    spin_horizontal_reference_angle_deg: float
    spin_vertical_reference_angle_deg: float

    def snapshot(self) -> PolarResponseProjection:
        return PolarResponseProjection(
            freq_hz=np.asarray(self.freq_hz).copy(),
            angle_deg=np.asarray(self.angle_deg).copy(),
            horizontal_spl_db=np.asarray(self.horizontal_spl_db).copy(),
            vertical_spl_db=np.asarray(self.vertical_spl_db).copy(),
            channel_on_axis_names=(
                None if self.channel_on_axis_names is None else np.asarray(self.channel_on_axis_names).copy()
            ),
            channel_on_axis_spl_db=(
                None if self.channel_on_axis_spl_db is None else np.asarray(self.channel_on_axis_spl_db).copy()
            ),
            on_axis_phase_deg=(None if self.on_axis_phase_deg is None else np.asarray(self.on_axis_phase_deg).copy()),
            channel_on_axis_phase_deg=(
                None if self.channel_on_axis_phase_deg is None else np.asarray(self.channel_on_axis_phase_deg).copy()
            ),
            spin_horizontal_reference_angle_deg=self.spin_horizontal_reference_angle_deg,
            spin_vertical_reference_angle_deg=self.spin_vertical_reference_angle_deg,
        )


@dataclass(frozen=True)
class ExcursionProjection:
    freq_hz: np.ndarray
    transducer_names: np.ndarray
    excursion_mm: np.ndarray

    def snapshot(self) -> ExcursionProjection:
        return ExcursionProjection(
            freq_hz=np.asarray(self.freq_hz).copy(),
            transducer_names=np.asarray(self.transducer_names).copy(),
            excursion_mm=np.asarray(self.excursion_mm).copy(),
        )


@dataclass(frozen=True)
class ElectricalImpedanceProjection:
    freq_hz: np.ndarray
    channel_names: np.ndarray
    magnitude_ohm: np.ndarray
    phase_deg: np.ndarray

    def snapshot(self) -> ElectricalImpedanceProjection:
        return ElectricalImpedanceProjection(
            freq_hz=np.asarray(self.freq_hz).copy(),
            channel_names=np.asarray(self.channel_names).copy(),
            magnitude_ohm=np.asarray(self.magnitude_ohm).copy(),
            phase_deg=np.asarray(self.phase_deg).copy(),
        )


@dataclass(frozen=True)
class GroupDelayProjection:
    freq_hz: np.ndarray
    trace_names: np.ndarray
    group_delay_ms: np.ndarray

    def snapshot(self) -> GroupDelayProjection:
        return GroupDelayProjection(
            freq_hz=np.asarray(self.freq_hz).copy(),
            trace_names=np.asarray(self.trace_names).copy(),
            group_delay_ms=np.asarray(self.group_delay_ms).copy(),
        )


@dataclass(frozen=True)
class MaxSplProjection:
    freq_hz: np.ndarray
    channel_names: np.ndarray
    spl_db: np.ndarray

    def snapshot(self) -> MaxSplProjection:
        return MaxSplProjection(
            freq_hz=np.asarray(self.freq_hz).copy(),
            channel_names=np.asarray(self.channel_names).copy(),
            spl_db=np.asarray(self.spl_db).copy(),
        )


@dataclass(frozen=True)
class VisualizationProjection:
    isobar: IsobarProjection
    impedance: ImpedanceProjection
    response: PolarResponseProjection
    excursion: ExcursionProjection | None = None
    electrical_impedance: ElectricalImpedanceProjection | None = None
    group_delay: GroupDelayProjection | None = None
    max_spl: MaxSplProjection | None = None
    spinorama_planes: SpinoramaCurves | None = None
    spinorama_spherical: SpinoramaCurves | None = None

    def snapshot(self) -> VisualizationProjection:
        return VisualizationProjection(
            isobar=self.isobar.snapshot(),
            impedance=self.impedance.snapshot(),
            response=self.response.snapshot(),
            excursion=None if self.excursion is None else self.excursion.snapshot(),
            electrical_impedance=(
                None
                if self.electrical_impedance is None
                else self.electrical_impedance.snapshot()
            ),
            group_delay=None if self.group_delay is None else self.group_delay.snapshot(),
            max_spl=None if self.max_spl is None else self.max_spl.snapshot(),
            spinorama_planes=_snapshot_spinorama_curves(self.spinorama_planes),
            spinorama_spherical=_snapshot_spinorama_curves(self.spinorama_spherical),
        )


class ResultProjectionService:
    """Project solve-domain results into view-specific, typed array models."""

    def prepare(
        self,
        dataset: LiveSolveDataset,
        channels: tuple[ChannelConfig, ...],
        options: ProjectionOptions,
        transducer_motion: TransducerMotionDataset | None = None,
        electrical_impedance: ElectricalImpedanceDataset | None = None,
        acoustic_load_impedance: AcousticLoadImpedanceDataset | None = None,
        max_spl_limits: dict[str, MaxSplLimit] | None = None,
        voltage_channel_names: frozenset[str] = frozenset(),
    ) -> VisualizationProjection | None:
        dataset.set_channel_synthesis(
            channels,
            flat_target_reference_angle_deg=options.horizontal_reference_angle_deg,
        )
        arrays = dataset.as_visualization_dataset(
            PrepConfig(
                angle_samples=options.angle_samples,
                freq_samples=options.freq_samples,
                octave_smoothing=options.octave_smoothing,
                hor_ref_angle=options.horizontal_reference_angle_deg,
                vert_ref_angle=options.vertical_reference_angle_deg,
                spin_hor_ref_angle=options.spin_horizontal_reference_angle_deg,
                spin_vert_ref_angle=options.spin_vertical_reference_angle_deg,
                min_db=options.min_db,
                max_db=options.max_db,
                normalize_polar=True,
                auto_db_span=False,
            )
        )
        if arrays is None:
            return None
        excursion = None
        if transducer_motion is not None:
            excursion_arrays = transducer_motion.as_excursion_arrays(dataset)
            if excursion_arrays is not None:
                excursion = ExcursionProjection(*excursion_arrays)
        electrical_projection = None
        if electrical_impedance is not None:
            electrical_arrays = electrical_impedance.as_impedance_arrays()
            if electrical_arrays is not None:
                electrical_projection = ElectricalImpedanceProjection(*electrical_arrays)
        group_delay_projection = None
        group_delay_arrays = dataset.as_group_delay_arrays()
        if group_delay_arrays is not None:
            group_delay_projection = GroupDelayProjection(*group_delay_arrays)
        max_spl_projection = None
        if transducer_motion is not None and max_spl_limits is not None:
            max_spl_arrays = transducer_motion.as_max_spl_arrays(
                dataset,
                max_spl_limits,
                voltage_channel_names,
            )
            if max_spl_arrays is not None:
                max_spl_projection = MaxSplProjection(*max_spl_arrays)
        impedance_projection = ImpedanceProjection(
            freq_hz=arrays["impedance_freq_hz"],
            radiator_names=arrays["impedance_radiator_names"],
            real=arrays["impedance_real"],
            imaginary=arrays["impedance_imag"],
        )
        if acoustic_load_impedance is not None:
            acoustic_load_arrays = acoustic_load_impedance.as_impedance_arrays()
            if acoustic_load_arrays is not None:
                impedance_projection = ImpedanceProjection(*acoustic_load_arrays)
        spinorama_options = {
            "horizontal_reference_angle_deg": options.spin_horizontal_reference_angle_deg,
            "vertical_reference_angle_deg": options.spin_vertical_reference_angle_deg,
        }
        spinorama_planes = compute_spinorama_from_planes(
            arrays["freq_hz"],
            arrays["polar_angle_deg"],
            arrays["horizontal_spl_db"],
            arrays["vertical_spl_db"],
            **spinorama_options,
        )
        spinorama_spherical = None
        sphere = dataset.as_balloon_raw_bundle()
        if sphere is not None:
            sphere_freqs = np.asarray(sphere.get("freq_hz"), dtype=np.float32)
            sphere_spl = np.asarray(sphere.get("spl_norm"), dtype=np.float32)
            if sphere_freqs.shape == arrays["freq_hz"].shape and np.allclose(
                sphere_freqs, arrays["freq_hz"]
            ):
                spinorama_spherical = compute_spinorama_from_planes(
                    arrays["freq_hz"],
                    arrays["polar_angle_deg"],
                    arrays["horizontal_spl_db"],
                    arrays["vertical_spl_db"],
                    spherical_spl_relative_db=sphere_spl,
                    **spinorama_options,
                )
        return VisualizationProjection(
            isobar=IsobarProjection(
                freq_hz=arrays["isobar_freq_hz"],
                angle_deg=arrays["isobar_angle_deg"],
                horizontal_db=arrays["horizontal_isobar_db"],
                vertical_db=arrays["vertical_isobar_db"],
                clip_min_db=float(arrays["clip_min_db"]),
                clip_max_db=float(arrays["clip_max_db"]),
            ),
            impedance=impedance_projection,
            response=PolarResponseProjection(
                freq_hz=arrays["freq_hz"],
                angle_deg=arrays["polar_angle_deg"],
                horizontal_spl_db=arrays["horizontal_spl_db"],
                vertical_spl_db=arrays["vertical_spl_db"],
                channel_on_axis_names=arrays.get("channel_on_axis_names"),
                channel_on_axis_spl_db=arrays.get("channel_on_axis_spl_db"),
                on_axis_phase_deg=arrays.get("on_axis_phase_deg"),
                channel_on_axis_phase_deg=arrays.get("channel_on_axis_phase_deg"),
                spin_horizontal_reference_angle_deg=float(arrays["spin_horizontal_reference_angle_deg"]),
                spin_vertical_reference_angle_deg=float(arrays["spin_vertical_reference_angle_deg"]),
            ),
            excursion=excursion,
            electrical_impedance=electrical_projection,
            group_delay=group_delay_projection,
            max_spl=max_spl_projection,
            spinorama_planes=spinorama_planes,
            spinorama_spherical=spinorama_spherical,
        )


def _snapshot_spinorama_curves(curves: SpinoramaCurves | None) -> SpinoramaCurves | None:
    if curves is None:
        return None
    return SpinoramaCurves(
        freq_hz=np.asarray(curves.freq_hz).copy(),
        on_axis_db=np.asarray(curves.on_axis_db).copy(),
        listening_window_db=np.asarray(curves.listening_window_db).copy(),
        early_reflections_db=np.asarray(curves.early_reflections_db).copy(),
        sound_power_db=np.asarray(curves.sound_power_db).copy(),
        estimated_in_room_db=np.asarray(curves.estimated_in_room_db).copy(),
        early_reflections_di_db=np.asarray(curves.early_reflections_di_db).copy(),
        sound_power_di_db=np.asarray(curves.sound_power_di_db).copy(),
        sound_power_di_label=curves.sound_power_di_label,
    )
