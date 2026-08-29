"""Plot image, numerical data, and speaker-package export orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QDialog,
    QMessageBox,
)

from blab.exporting import (
    TraceQuantity,
    export_frequency_trace_table,
    export_on_axis_text_files,
    export_plot_png,
    export_polar_text_files,
)
from blab.max_spl import max_spl_limits_from_payload
from blab.plotting import VisualizerConfig
from blab.spinorama import compute_spinorama_from_planes
from blab.ui.plots import FINAL_ISOBAR_ANGLE_SAMPLES, FINAL_ISOBAR_FREQ_SAMPLES
from blab.ui.speaker_package_dialog import SpeakerPackageDialog


class ExportsMixin:
    """Plot image and numerical data export.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    @Slot(str)
    def export_plot(self, plot_id: str) -> None:
        dataset = self.prepared_live_dataset(
            angle_samples=(
                FINAL_ISOBAR_ANGLE_SAMPLES
                if plot_id in {"horizontal_isobar", "vertical_isobar"}
                else None
            ),
            freq_samples=(
                FINAL_ISOBAR_FREQ_SAMPLES
                if plot_id in {"horizontal_isobar", "vertical_isobar"}
                else None
            ),
        )
        if dataset is None or not self.plot_data_is_available(plot_id):
            QMessageBox.warning(self, "No plot data", "Run a solve before exporting a plot.")
            return

        entry = next((item for item in self.plot_entries if item.plot_id == plot_id), None)
        if entry is None:
            return

        output_path = self.file_dialogs.save_file(
            self,
            f"Export {entry.title}",
            "PNG images (*.png);;All files (*)",
            entry.default_filename,
        )
        if output_path is None:
            return

        if output_path.suffix == "":
            output_path = output_path.with_suffix(".png")
        try:
            entry.update(dataset)
            figure = getattr(entry.widget, "figure")
            output_path = export_plot_png(figure, output_path, dpi=VisualizerConfig.figure_dpi)
            self.status_label.setText(f"Exported {entry.title} to {output_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export plot failed", str(exc))

    @Slot(str)
    def export_plot_data(self, plot_id: str) -> None:
        entry = next((item for item in self.plot_entries if item.plot_id == plot_id), None)
        if entry is None:
            return
        if not self.plot_data_is_available(plot_id):
            QMessageBox.warning(self, "No plot data", "This plot has no data to export.")
            return

        spec = entry.data_export
        if spec.target_kind == "directory":
            output_target = self.file_dialogs.select_directory(self, f"Export {entry.title} data")
        else:
            output_target = self.file_dialogs.save_file(
                self,
                f"Export {entry.title} data",
                spec.file_filter,
                spec.default_filename,
            )
        if output_target is None:
            return

        try:
            written = self._write_plot_data(plot_id, output_target)
            if len(written) == 1:
                self.status_label.setText(f"Exported {entry.title} data to {written[0]}")
            else:
                self.status_label.setText(
                    f"Exported {len(written)} {entry.title} data files to {output_target}"
                )
        except Exception as exc:
            QMessageBox.critical(self, f"Export {entry.title} data failed", str(exc))

    def plot_data_is_available(self, plot_id: str) -> bool:
        dataset = self.live_dataset
        if dataset is None or dataset.solved_count == 0:
            return False
        session = self._solve_session()
        try:
            if plot_id == "electrical_impedance":
                return (
                    session.electrical_impedance is not None
                    and session.electrical_impedance.as_impedance_arrays() is not None
                )
            if plot_id == "group_delay":
                return dataset.as_group_delay_arrays() is not None
            if plot_id == "transducer_excursion":
                return (
                    session.transducer_motion is not None
                    and session.transducer_motion.as_excursion_arrays(dataset) is not None
                )
            if plot_id == "max_spl":
                if session.transducer_motion is None or not session.max_spl_requested:
                    return False
                return (
                    session.transducer_motion.as_max_spl_arrays(
                        dataset,
                        max_spl_limits_from_payload(self.project.max_spl_limits_by_channel),
                        session.voltage_channel_names,
                    )
                    is not None
                )
            return plot_id in {
                "horizontal_isobar",
                "vertical_isobar",
                "acoustic_impedance",
                "on_axis_frequency_response",
                "spinorama",
            }
        except (KeyError, TypeError, ValueError):
            return False

    def _write_plot_data(self, plot_id: str, output_target: str | Path) -> list[Path]:
        dataset = self.live_dataset
        if dataset is None:
            raise ValueError("No solved plot data is available.")
        dataset.set_channel_synthesis(
            self.channel_configs(),
            flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
        )
        target = Path(output_target)
        if plot_id in {"horizontal_isobar", "vertical_isobar"}:
            plane = "H" if plot_id == "horizontal_isobar" else "V"
            return export_polar_text_files(
                dataset,
                target,
                planes=(plane,),
                reference_angles_deg={
                    "H": self.preferences.horizontal_normalization_angle,
                    "V": self.preferences.vertical_normalization_angle,
                },
            )
        if plot_id == "on_axis_frequency_response":
            if dataset.supports_channel_resynthesis:
                return export_on_axis_text_files(dataset, target, include_sum=True)
            projection = self.prepared_live_dataset()
            if projection is None:
                raise ValueError("No prepared on-axis data is available.")
            response = projection.response
            on_axis = np.asarray(
                [
                    np.interp(0.0, response.angle_deg.astype(float), row.astype(float))
                    for row in response.horizontal_spl_db
                ],
                dtype=np.float32,
            )
            output_path = target / "on_axis_Sum.txt"
            return [
                export_frequency_trace_table(
                    output_path,
                    title="On-Axis Frequency Response",
                    frequency_hz=response.freq_hz,
                    trace_names=np.asarray(["Sum"]),
                    quantities=(TraceQuantity("SPL", "dB", on_axis[np.newaxis, :]),),
                )
            ]

        projection = self.prepared_live_dataset()
        if projection is None:
            raise ValueError("No prepared plot data is available.")
        if plot_id == "acoustic_impedance":
            data = projection.impedance
            return [
                export_frequency_trace_table(
                    target,
                    title="Normalized Acoustic Impedance (Z / rho*c*Sd)",
                    frequency_hz=data.freq_hz,
                    trace_names=data.radiator_names,
                    quantities=(
                        TraceQuantity("Real", "1", data.real),
                        TraceQuantity("Imaginary", "1", data.imaginary),
                    ),
                )
            ]
        if plot_id == "electrical_impedance" and projection.electrical_impedance is not None:
            data = projection.electrical_impedance
            return [
                export_frequency_trace_table(
                    target,
                    title="Electrical Impedance",
                    frequency_hz=data.freq_hz,
                    trace_names=data.channel_names,
                    quantities=(
                        TraceQuantity("Magnitude", "ohm", data.magnitude_ohm),
                        TraceQuantity("Phase", "deg", data.phase_deg),
                    ),
                )
            ]
        if plot_id == "group_delay" and projection.group_delay is not None:
            data = projection.group_delay
            return [
                export_frequency_trace_table(
                    target,
                    title="Group Delay",
                    frequency_hz=data.freq_hz,
                    trace_names=data.trace_names,
                    quantities=(TraceQuantity("Group Delay", "ms", data.group_delay_ms),),
                )
            ]
        if plot_id == "transducer_excursion" and projection.excursion is not None:
            data = projection.excursion
            return [
                export_frequency_trace_table(
                    target,
                    title="Transducer Excursion",
                    frequency_hz=data.freq_hz,
                    trace_names=data.transducer_names,
                    quantities=(TraceQuantity("Excursion", "mm", data.excursion_mm),),
                )
            ]
        if plot_id == "max_spl" and projection.max_spl is not None:
            data = projection.max_spl
            return [
                export_frequency_trace_table(
                    target,
                    title="Maximum SPL",
                    frequency_hz=data.freq_hz,
                    trace_names=data.channel_names,
                    quantities=(TraceQuantity("SPL", "dB", data.spl_db),),
                )
            ]
        if plot_id == "spinorama":
            response = projection.response
            curves = compute_spinorama_from_planes(
                response.freq_hz,
                response.angle_deg,
                response.horizontal_spl_db,
                response.vertical_spl_db,
                horizontal_reference_angle_deg=response.spin_horizontal_reference_angle_deg,
                vertical_reference_angle_deg=response.spin_vertical_reference_angle_deg,
            )
            named_curves = (*curves.spl_curves(), *curves.di_curves())
            return [
                export_frequency_trace_table(
                    target,
                    title="Spinorama",
                    frequency_hz=curves.freq_hz,
                    trace_names=np.asarray([name for name, _values in named_curves]),
                    quantities=(
                        TraceQuantity(
                            "Level",
                            "dB",
                            np.vstack([values for _name, values in named_curves]),
                        ),
                    ),
                )
            ]
        raise ValueError(f"No data exporter is registered for plot {plot_id!r}.")

    @Slot()
    def export_speaker_package(self) -> None:
        system = self._project_document().physical_system
        default_name = "Speaker" if system is None else system.name
        dialog = SpeakerPackageDialog(
            self,
            default_name=default_name,
            file_dialogs=self.file_dialogs,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self.solve_workflow.start_speaker_package_solve(dialog.config())
