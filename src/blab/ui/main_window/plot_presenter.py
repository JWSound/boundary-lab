"""Plot canvas presentation: DPI handling, coalesced live refresh, contours, and updates."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Slot

from blab.ui.main_window.plot_controls import contour_controls
from blab.ui.main_window_widgets import (
    PlotEntry,
)
from blab.ui.plots import (
    FINAL_ISOBAR_ANGLE_SAMPLES,
    FINAL_ISOBAR_FREQ_SAMPLES,
    FINAL_ISOBAR_SHADING,
    LIVE_ISOBAR_SHADING,
    IsobarCanvas,
)
from blab.ui.result_projection import (
    ProjectionOptions,
    VisualizationProjection,
)
from blab.ui.settings import (
    live_plot_angle_samples,
    live_plot_freq_samples,
)


class PlotPresenterMixin:
    """Plot canvas presentation: DPI handling, coalesced live refresh, contours, and updates.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    def connect_dpi_signals(self) -> None:
        window = self.windowHandle()
        if window is None:
            QTimer.singleShot(0, self.connect_dpi_signals)
            return
        if self._plot_dpi_window_handle is not window:
            if self._plot_dpi_window_handle is not None:
                try:
                    self._plot_dpi_window_handle.screenChanged.disconnect(self._on_plot_screen_changed)
                except (RuntimeError, TypeError):
                    pass
            window.screenChanged.connect(self._on_plot_screen_changed)
            self._plot_dpi_window_handle = window
        self._on_plot_screen_changed(window.screen())

    def _on_plot_screen_changed(self, screen) -> None:
        if screen is self._plot_dpi_screen:
            return
        self._disconnect_plot_dpi_screen()
        self._plot_dpi_screen = screen
        if screen is not None:
            screen.logicalDotsPerInchChanged.connect(self._schedule_plot_canvas_dpi_refresh)
            screen.physicalDotsPerInchChanged.connect(self._schedule_plot_canvas_dpi_refresh)
            screen.geometryChanged.connect(self._schedule_plot_canvas_dpi_refresh)
        self._schedule_plot_canvas_dpi_refresh()

    def _disconnect_plot_dpi_screen(self) -> None:
        screen = self._plot_dpi_screen
        self._plot_dpi_screen = None
        if screen is None:
            return
        for signal in (
            screen.logicalDotsPerInchChanged,
            screen.physicalDotsPerInchChanged,
            screen.geometryChanged,
        ):
            try:
                signal.disconnect(self._schedule_plot_canvas_dpi_refresh)
            except (RuntimeError, TypeError):
                pass

    def _schedule_plot_canvas_dpi_refresh(self, *_args) -> None:
        if self._plot_dpi_refresh_pending:
            return
        self._plot_dpi_refresh_pending = True
        QTimer.singleShot(0, self._refresh_plot_canvas_dpi)

    def _refresh_plot_canvas_dpi(self) -> None:
        self._plot_dpi_refresh_pending = False
        window = self.windowHandle()
        screen = None if window is None else window.screen()
        for entry in self.plot_entries:
            if not self._plot_entry_is_actively_visible(entry):
                continue
            canvas = entry.widget
            if screen is not None and hasattr(canvas, "_update_screen"):
                canvas._update_screen(screen)
            if hasattr(canvas, "_update_pixel_ratio"):
                canvas._update_pixel_ratio()
            canvas.draw_idle()

    def request_live_refresh(self) -> None:
        self._live_plot_refresh_dirty = True
        if not self._live_plot_refresh_timer.isActive():
            self._live_plot_refresh_timer.start()

    @Slot()
    def flush_live_refresh(self) -> None:
        if not self._live_plot_refresh_dirty:
            return
        self._live_plot_refresh_dirty = False
        self.refresh_plots(active_only=True)

    def cancel_live_refresh(self) -> None:
        self._live_plot_refresh_dirty = False
        self._live_plot_refresh_timer.stop()

    def clear_plots(self) -> None:
        self.cancel_live_refresh()
        self._solve_session().begin()
        for entry in self.plot_entries:
            entry.widget._draw_empty()
        self.set_plot_exports_available(False)
        self.set_polar_export_available(False)
        self.set_on_axis_export_available(False)
        self.set_balloon_plot_available(False)
        self.refresh_contour_controls()

    def apply_last_completed_comparison(self) -> None:
        dataset = self._last_completed_visualization_dataset
        if dataset is None:
            for entry in self.plot_entries:
                entry.widget.clear_comparison_plot()
            return
        isobar = dataset.isobar
        options = {
            "shading": FINAL_ISOBAR_SHADING,
            "contour_step_db": self.preferences.isobar_contour_step_db,
        }
        self.horizontal_plot.set_comparison_plot(
            isobar.freq_hz,
            isobar.angle_deg,
            isobar.horizontal_db,
            isobar.clip_min_db,
            isobar.clip_max_db,
            **options,
        )
        self.vertical_plot.set_comparison_plot(
            isobar.freq_hz,
            isobar.angle_deg,
            isobar.vertical_db,
            isobar.clip_min_db,
            isobar.clip_max_db,
            **options,
        )
        impedance = dataset.impedance
        self.impedance_plot.set_comparison_plot(
            impedance.freq_hz,
            impedance.radiator_names,
            impedance.real,
            impedance.imaginary,
        )
        response = dataset.response
        self.on_axis_plot.set_comparison_plot(
            response.freq_hz,
            response.angle_deg,
            response.horizontal_spl_db,
            response.channel_on_axis_names,
            response.channel_on_axis_spl_db,
        )
        self.spinorama_plot.set_comparison_plot(
            response.freq_hz,
            response.angle_deg,
            response.horizontal_spl_db,
            response.vertical_spl_db,
            horizontal_reference_angle_deg=response.spin_horizontal_reference_angle_deg,
            vertical_reference_angle_deg=response.spin_vertical_reference_angle_deg,
        )

    def clear_comparison_history(self) -> None:
        self._solve_session().forget_comparison()
        for entry in self.plot_entries:
            entry.widget.clear_comparison_plot()

    def visible_isobar_plots(self) -> tuple[IsobarCanvas, ...]:
        plots: list[IsobarCanvas] = []
        horizontal_dock = self.plot_docks.get("horizontal_isobar")
        vertical_dock = self.plot_docks.get("vertical_isobar")
        if horizontal_dock is not None and not horizontal_dock.isHidden():
            plots.append(self.horizontal_plot)
        if vertical_dock is not None and not vertical_dock.isHidden():
            plots.append(self.vertical_plot)
        return tuple(plots)

    def refresh_contour_controls(self) -> None:
        for plot_id, plot in (
            ("horizontal_isobar", self.horizontal_plot),
            ("vertical_isobar", self.vertical_plot),
        ):
            dock = self.plot_docks.get(plot_id)
            controls = contour_controls(
                has_live_data=self.live_dataset is not None,
                final_resolution_active=self._use_final_isobar_resolution,
                final_plots_rendered=self._final_isobar_plots_rendered,
                plot_visible=dock is not None and not dock.isHidden(),
                has_captured_contours=plot.has_captured_contours,
            )
            capture_action = self.capture_contour_actions.get(plot_id)
            clear_action = self.clear_contour_actions.get(plot_id)
            if capture_action is not None:
                capture_action.setEnabled(controls.capture)
            if clear_action is not None:
                clear_action.setEnabled(controls.clear)

    @Slot(str)
    def capture_isobar_contours(self, plot_id: str) -> None:
        plot = self._isobar_plot_for_id(plot_id)
        if plot is None:
            return
        if plot.capture_contours():
            entry = next((item for item in self.plot_entries if item.plot_id == plot_id), None)
            self.status_label.setText(f"Captured contours for {entry.title if entry is not None else 'isobar plot'}")
        self.refresh_contour_controls()

    @Slot(str)
    def clear_isobar_contours(self, plot_id: str) -> None:
        plot = self._isobar_plot_for_id(plot_id)
        if plot is None:
            return
        plot.clear_contours()
        entry = next((item for item in self.plot_entries if item.plot_id == plot_id), None)
        self.status_label.setText(f"Cleared contours for {entry.title if entry is not None else 'isobar plot'}")
        self.refresh_contour_controls()

    def _isobar_plot_for_id(self, plot_id: str) -> IsobarCanvas | None:
        if plot_id == "horizontal_isobar":
            return self.horizontal_plot
        if plot_id == "vertical_isobar":
            return self.vertical_plot
        return None

    def prepared_live_dataset(
        self,
        *,
        angle_samples: int | None = None,
        freq_samples: int | None = None,
    ) -> VisualizationProjection | None:
        if self.live_dataset is None:
            return None
        if angle_samples is None:
            angle_samples = live_plot_angle_samples(self.preferences.live_plot_quality)
        if freq_samples is None:
            freq_samples = live_plot_freq_samples(self.preferences.live_plot_quality)
        return self.result_projection_service.prepare(
            self.live_dataset,
            self.channel_configs(),
            ProjectionOptions(
                angle_samples=angle_samples,
                freq_samples=freq_samples,
                octave_smoothing=self.preferences.polar_smoothing,
                horizontal_reference_angle_deg=self.preferences.horizontal_normalization_angle,
                vertical_reference_angle_deg=self.preferences.vertical_normalization_angle,
                spin_horizontal_reference_angle_deg=self.preferences.spin_horizontal_reference_angle,
                spin_vertical_reference_angle_deg=self.preferences.spin_vertical_reference_angle,
                min_db=self.preferences.spl_min_db,
                max_db=self.preferences.spl_max_db,
            ),
        )

    def _plot_entry_is_actively_visible(self, entry: PlotEntry) -> bool:
        dock = self.plot_docks.get(entry.plot_id)
        if dock is None or dock.isHidden():
            return False
        return not entry.widget.visibleRegion().isEmpty()

    def refresh_plots(self, *, active_only: bool = False) -> VisualizationProjection | None:
        # A tab-covered QDockWidget isVisible() and isHidden() == False even
        # though its canvas has no visible region and may still be only a few
        # pixels high. Drawing Matplotlib text at that transient geometry can
        # fail in FreeType, and leaves the tab with a partially drawn figure.
        # All redraws therefore target canvases that Qt is actually exposing;
        # covered tabs are refreshed when their dock becomes active.
        visible_entries = [
            entry
            for entry in self.plot_entries
            if (dock := self.plot_docks.get(entry.plot_id)) is not None
            and not dock.isHidden()
            and self._plot_entry_is_actively_visible(entry)
        ]
        if not visible_entries:
            return None

        dataset = self.prepared_live_dataset(
            angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES
            if self._use_final_isobar_resolution
            else live_plot_angle_samples(self.preferences.live_plot_quality),
            freq_samples=FINAL_ISOBAR_FREQ_SAMPLES
            if self._use_final_isobar_resolution
            else live_plot_freq_samples(self.preferences.live_plot_quality),
        )
        if dataset is None:
            return None

        for entry in visible_entries:
            entry.update(dataset)
        return dataset

    def _update_horizontal_plot(self, dataset: VisualizationProjection) -> None:
        isobar = dataset.isobar
        self.horizontal_plot.update_plot(
            isobar.freq_hz,
            isobar.angle_deg,
            isobar.horizontal_db,
            isobar.clip_min_db,
            isobar.clip_max_db,
            shading=FINAL_ISOBAR_SHADING if self._use_final_isobar_resolution else LIVE_ISOBAR_SHADING,
            contour_step_db=self.preferences.isobar_contour_step_db,
        )

    def _update_vertical_plot(self, dataset: VisualizationProjection) -> None:
        isobar = dataset.isobar
        self.vertical_plot.update_plot(
            isobar.freq_hz,
            isobar.angle_deg,
            isobar.vertical_db,
            isobar.clip_min_db,
            isobar.clip_max_db,
            shading=FINAL_ISOBAR_SHADING if self._use_final_isobar_resolution else LIVE_ISOBAR_SHADING,
            contour_step_db=self.preferences.isobar_contour_step_db,
        )

    def _update_impedance_plot(self, dataset: VisualizationProjection) -> None:
        impedance = dataset.impedance
        self.impedance_plot.update_plot(
            impedance.freq_hz,
            impedance.radiator_names,
            impedance.real,
            impedance.imaginary,
        )

    def _update_on_axis_plot(self, dataset: VisualizationProjection) -> None:
        response = dataset.response
        self.on_axis_plot.update_plot(
            response.freq_hz,
            response.angle_deg,
            response.horizontal_spl_db,
            response.channel_on_axis_names,
            response.channel_on_axis_spl_db,
        )

    def _update_spinorama_plot(self, dataset: VisualizationProjection) -> None:
        response = dataset.response
        self.spinorama_plot.update_plot(
            response.freq_hz,
            response.angle_deg,
            response.horizontal_spl_db,
            response.vertical_spl_db,
            horizontal_reference_angle_deg=response.spin_horizontal_reference_angle_deg,
            vertical_reference_angle_deg=response.spin_vertical_reference_angle_deg,
        )
