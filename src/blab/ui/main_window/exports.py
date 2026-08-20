"""Plot image and polar data export."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QDialog,
    QMessageBox,
)

from blab.exporting import (
    default_on_axis_filename,
    export_on_axis_text_files,
    export_plot_png,
    export_polar_text_files,
)
from blab.plotting import VisualizerConfig
from blab.ui.plots import (
    FINAL_ISOBAR_ANGLE_SAMPLES,
    FINAL_ISOBAR_FREQ_SAMPLES,
)
from blab.ui.speaker_package_dialog import SpeakerPackageDialog


class ExportsMixin:
    """Plot image and polar data export.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    @Slot(str)
    def export_plot(self, plot_id: str) -> None:
        dataset = self.prepared_live_dataset(
            angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES if plot_id in {"horizontal_isobar", "vertical_isobar"} else None,
            freq_samples=FINAL_ISOBAR_FREQ_SAMPLES if plot_id in {"horizontal_isobar", "vertical_isobar"} else None,
        )
        if dataset is None:
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

    @Slot()
    def export_polar_data(self) -> None:
        if self.live_dataset is None or self.live_dataset.solved_count == 0:
            QMessageBox.warning(self, "No polar data", "Run a solve before exporting polar data.")
            return

        output_dir = self.file_dialogs.select_directory(
            self,
            "Export polar data",
        )
        if output_dir is None:
            return

        try:
            self.live_dataset.set_channel_synthesis(
                self.channel_configs(),
                flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
            )
            written = export_polar_text_files(self.live_dataset, output_dir)
            self.status_label.setText(f"Exported {len(written)} polar files to {output_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Export polar data failed", str(exc))

    @Slot()
    def export_on_axis_data(self) -> None:
        if self.live_dataset is None or self.live_dataset.solved_count == 0:
            QMessageBox.warning(self, "No on-axis data", "Run a solve before exporting on-axis data.")
            return

        try:
            self.live_dataset.set_channel_synthesis(
                self.channel_configs(),
                flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
            )
            _freqs, channel_names, _spl_db, _phase_deg = self.live_dataset.as_channel_on_axis_export_arrays()
        except Exception as exc:
            QMessageBox.critical(self, "Export on-axis data failed", str(exc))
            return

        if channel_names.size == 1:
            output_target = self.file_dialogs.save_file(
                self,
                "Export on-axis data",
                "Text files (*.txt);;All files (*)",
                default_on_axis_filename(str(channel_names[0])),
            )
        else:
            output_target = self.file_dialogs.select_directory(
                self,
                "Export on-axis channel data",
            )
        if output_target is None:
            return

        try:
            written = export_on_axis_text_files(self.live_dataset, output_target)
            if len(written) == 1:
                self.status_label.setText(f"Exported on-axis data to {written[0]}")
            else:
                self.status_label.setText(f"Exported {len(written)} on-axis channel files to {output_target}")
        except Exception as exc:
            QMessageBox.critical(self, "Export on-axis data failed", str(exc))

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
