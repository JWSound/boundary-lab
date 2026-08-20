"""Typed presentation projections for live solve results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blab.config import ChannelConfig
from blab.live import ElectricalImpedanceDataset, LiveSolveDataset, TransducerMotionDataset
from blab.postprocess import PrepConfig


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
class VisualizationProjection:
    isobar: IsobarProjection
    impedance: ImpedanceProjection
    response: PolarResponseProjection
    excursion: ExcursionProjection | None = None
    electrical_impedance: ElectricalImpedanceProjection | None = None

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
        return VisualizationProjection(
            isobar=IsobarProjection(
                freq_hz=arrays["isobar_freq_hz"],
                angle_deg=arrays["isobar_angle_deg"],
                horizontal_db=arrays["horizontal_isobar_db"],
                vertical_db=arrays["vertical_isobar_db"],
                clip_min_db=float(arrays["clip_min_db"]),
                clip_max_db=float(arrays["clip_max_db"]),
            ),
            impedance=ImpedanceProjection(
                freq_hz=arrays["impedance_freq_hz"],
                radiator_names=arrays["impedance_radiator_names"],
                real=arrays["impedance_real"],
                imaginary=arrays["impedance_imag"],
            ),
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
        )
