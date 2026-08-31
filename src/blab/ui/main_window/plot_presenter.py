"""Plot canvas presentation: DPI handling, coalesced live refresh, contours, and updates."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Slot

from blab.max_spl import max_spl_limits_from_payload, max_spl_limits_payload
from blab.spinorama import compute_spinorama_from_planes
from blab.ui.main_window.plot_controls import contour_controls
from blab.ui.main_window_widgets import (
    PlotEntry,
)
from blab.ui.max_spl_dialog import MaxSplLimitsDialog
from blab.ui.plot_limits_dialog import PlotLimitsDialog
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
        self.set_spherical_spin_available(False)
        observation_planes = getattr(self, "observation_plane_controller", None)
        if observation_planes is not None:
            observation_planes.sync_view()
        for entry in self.plot_entries:
            entry.widget._draw_empty()
        self.set_plot_exports_available(False)
        self.set_balloon_plot_available(False)
        self.set_max_spl_available(False)
        self.refresh_contour_controls()

    @Slot(str)
    def open_plot_limits(self, plot_id: str) -> None:
        entry = next((item for item in self.plot_entries if item.plot_id == plot_id), None)
        if entry is None:
            return
        canvas = entry.widget
        dialog = PlotLimitsDialog(
            entry.title,
            canvas.displayed_axis_limits(),
            automatic=canvas.automatic_axis_limits,
            parent=self,
        )
        if not dialog.exec():
            return
        canvas.set_axis_limits(dialog.limits())

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
        electrical_impedance = dataset.electrical_impedance
        if electrical_impedance is None:
            self.electrical_impedance_plot.clear_comparison_plot()
        else:
            self.electrical_impedance_plot.set_comparison_plot(
                electrical_impedance.freq_hz,
                electrical_impedance.channel_names,
                electrical_impedance.magnitude_ohm,
                electrical_impedance.phase_deg,
            )
        response = dataset.response
        self.on_axis_plot.set_comparison_plot(
            response.freq_hz,
            response.angle_deg,
            response.horizontal_spl_db,
            response.channel_on_axis_names,
            response.channel_on_axis_spl_db,
            response.on_axis_phase_deg,
            response.channel_on_axis_phase_deg,
        )
        group_delay = dataset.group_delay
        if group_delay is None:
            self.group_delay_plot.clear_comparison_plot()
        else:
            self.group_delay_plot.set_comparison_plot(
                group_delay.freq_hz,
                group_delay.trace_names,
                group_delay.group_delay_ms,
            )
        if dataset.excursion is None:
            self.excursion_plot.clear_comparison_plot()
        else:
            self.excursion_plot.set_comparison_plot(
                dataset.excursion.freq_hz,
                dataset.excursion.transducer_names,
                dataset.excursion.excursion_mm,
            )
        if dataset.max_spl is None:
            self.max_spl_plot.clear_comparison_plot()
        else:
            self.max_spl_plot.set_comparison_plot(
                dataset.max_spl.freq_hz,
                dataset.max_spl.channel_names,
                dataset.max_spl.spl_db,
            )
        if dataset.spinorama_planes is None:
            self.spinorama_plot.set_comparison_plot(
                response.freq_hz,
                response.angle_deg,
                response.horizontal_spl_db,
                response.vertical_spl_db,
                horizontal_reference_angle_deg=response.spin_horizontal_reference_angle_deg,
                vertical_reference_angle_deg=response.spin_vertical_reference_angle_deg,
            )
        else:
            self.spinorama_plot.set_comparison_curves(
                self._spinorama_curves_for_projection(dataset)
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
            transducer_motion=self._solve_session().transducer_motion,
            electrical_impedance=self._solve_session().electrical_impedance,
            acoustic_load_impedance=self._solve_session().acoustic_load_impedance,
            max_spl_limits=(
                max_spl_limits_from_payload(self.project.max_spl_limits_by_channel)
                if self._solve_session().max_spl_requested
                else None
            ),
            voltage_channel_names=self._solve_session().voltage_channel_names,
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
            response.on_axis_phase_deg,
            response.channel_on_axis_phase_deg,
        )

    def _update_electrical_impedance_plot(self, dataset: VisualizationProjection) -> None:
        impedance = dataset.electrical_impedance
        if impedance is None:
            if self.electrical_impedance_plot._plot_state is not None:
                self.electrical_impedance_plot._draw_empty()
            return
        self.electrical_impedance_plot.update_plot(
            impedance.freq_hz,
            impedance.channel_names,
            impedance.magnitude_ohm,
            impedance.phase_deg,
        )

    def _update_group_delay_plot(self, dataset: VisualizationProjection) -> None:
        group_delay = dataset.group_delay
        if group_delay is None:
            if self.group_delay_plot._plot_state is not None:
                self.group_delay_plot._draw_empty()
            return
        self.group_delay_plot.update_plot(
            group_delay.freq_hz,
            group_delay.trace_names,
            group_delay.group_delay_ms,
        )

    def _update_excursion_plot(self, dataset: VisualizationProjection) -> None:
        excursion = dataset.excursion
        if excursion is None:
            if self.excursion_plot._plot_state is not None:
                self.excursion_plot._draw_empty()
            return
        self.excursion_plot.update_plot(
            excursion.freq_hz,
            excursion.transducer_names,
            excursion.excursion_mm,
        )

    def _update_max_spl_plot(self, dataset: VisualizationProjection) -> None:
        maximum = dataset.max_spl
        if maximum is None:
            if self.max_spl_plot._plot_state is not None:
                self.max_spl_plot._draw_empty()
            return
        self.max_spl_plot.update_plot(
            maximum.freq_hz,
            maximum.channel_names,
            maximum.spl_db,
        )

    @Slot()
    def calculate_max_spl(self) -> None:
        session = self._solve_session()
        motion = session.transducer_motion
        channel_names = self.max_spl_channel_names()
        if not channel_names and motion is not None:
            channel_names = motion.eligible_max_spl_channel_names(session.voltage_channel_names)
        if not channel_names:
            self.warn("Maximum SPL", "No voltage-only electrodynamic channels are available.")
            return
        dialog = MaxSplLimitsDialog(
            channel_names,
            max_spl_limits_from_payload(self.project.max_spl_limits_by_channel),
            self,
        )
        if not dialog.exec():
            return
        limits = dialog.limits()
        self.project.max_spl_limits_by_channel = max_spl_limits_payload(limits)
        session.max_spl_requested = any(limit.enabled for limit in limits.values())
        if motion is None or self.live_dataset is None:
            self.show_status("Maximum SPL configuration updated")
            return
        if not session.max_spl_requested:
            self.max_spl_plot._draw_empty()
            self.set_max_spl_export_available(False)
            self.show_status("Maximum SPL configuration updated; all channels disabled")
            return
        dataset = self.prepared_live_dataset()
        if dataset is None or dataset.max_spl is None:
            self.warn("Maximum SPL", "Maximum SPL could not be calculated from the current solve.")
            return
        self._update_max_spl_plot(dataset)
        self.set_max_spl_export_available(True)
        enabled_count = sum(limit.enabled for limit in limits.values())
        self.show_status(f"Maximum SPL configuration updated: {enabled_count} channel(s) enabled")

    def _update_spinorama_plot(self, dataset: VisualizationProjection) -> None:
        self.set_spherical_spin_available(dataset.spinorama_spherical is not None)
        self.spinorama_plot.update_curves(self._spinorama_curves_for_projection(dataset))

    def _spinorama_curves_for_projection(self, dataset: VisualizationProjection):
        if self.spherical_spin_action.isChecked() and dataset.spinorama_spherical is not None:
            return dataset.spinorama_spherical
        if dataset.spinorama_planes is not None:
            return dataset.spinorama_planes
        response = dataset.response
        return compute_spinorama_from_planes(
            response.freq_hz,
            response.angle_deg,
            response.horizontal_spl_db,
            response.vertical_spl_db,
            horizontal_reference_angle_deg=response.spin_horizontal_reference_angle_deg,
            vertical_reference_angle_deg=response.spin_vertical_reference_angle_deg,
        )

    def set_spherical_spin_available(self, available: bool) -> None:
        action = getattr(self, "spherical_spin_action", None)
        if action is not None:
            action.setEnabled(bool(available))

    @Slot(bool)
    def set_spherical_spin_enabled(self, _enabled: bool) -> None:
        dataset = self.prepared_live_dataset()
        if dataset is not None:
            self._update_spinorama_plot(dataset)
        comparison = self._last_completed_visualization_dataset
        if comparison is None:
            self.spinorama_plot.clear_comparison_plot()
        else:
            self.spinorama_plot.set_comparison_curves(
                self._spinorama_curves_for_projection(comparison)
            )
