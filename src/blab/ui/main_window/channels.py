"""Source and channel configuration, including the channel config dialog."""

from __future__ import annotations

from PySide6.QtCore import Slot

from blab.config import ChannelConfig, RadiatorConfig
from blab.generators.base import GeneratedGeometry
from blab.ui.dialogs import (
    ChannelConfigDialog,
)
from blab.ui.project_state import (
    generator_mesh_name,
)
from blab.ui.source_channel_config import (
    apply_saved_imported_source_config,
    apply_saved_source_config_to_result,
    channel_config_payload,
    channel_configs_from_payload,
    channels_for_solver_radiators,
)


class ChannelsMixin:
    """Source and channel configuration, including the channel config dialog.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    def source_config_by_name(self) -> dict[str, dict]:
        return self.project.source_config_by_name

    def channel_config_by_name(self) -> dict[str, dict]:
        return self.project.channel_config_by_name

    def _save_channel_config(self, channels: tuple[ChannelConfig, ...]) -> None:
        self.project.channel_config_by_name = channel_config_payload(channels)

    def channel_configs(self) -> tuple[ChannelConfig, ...]:
        return channel_configs_from_payload(self.project.channel_config_by_name)

    def solver_channel_configs(
        self,
        radiators: tuple[RadiatorConfig, ...],
    ) -> tuple[ChannelConfig, ...]:
        return channels_for_solver_radiators(self.channel_configs(), radiators)

    def _channel_configs_for_current_radiators(self) -> tuple[ChannelConfig, ...]:
        return self.solver_channel_configs(self.all_radiators())

    def discard_channel_config_dialog(self) -> None:
        dialog = self.channel_config_dialog
        self.channel_config_dialog = None
        if dialog is not None:
            dialog.deleteLater()

    def apply_saved_source_config_to_result(
        self,
        result: GeneratedGeometry | None,
        mesh_name: str,
    ) -> GeneratedGeometry | None:
        return apply_saved_source_config_to_result(result, mesh_name, self.source_config_by_name())

    def apply_saved_imported_source_config(self, surface_tags: dict[str, tuple[str, int]]) -> None:
        generated_mesh_names = {generator_mesh_name(document) for document in self.generator_documents}
        self.imported_radiators = apply_saved_imported_source_config(
            surface_tags=surface_tags,
            generated_mesh_names=generated_mesh_names,
            existing_radiators=self.imported_radiators,
            config_by_name=self.source_config_by_name(),
        )

    @Slot()
    def open_channel_config(self) -> None:
        if self.channel_config_dialog is not None:
            self.channel_config_dialog.show()
            self.channel_config_dialog.raise_()
            self.channel_config_dialog.activateWindow()
            return

        dialog = ChannelConfigDialog(self._channel_configs_for_current_radiators(), self)
        self.channel_config_dialog = dialog
        dialog.channelsApplied.connect(self._apply_channel_config)
        dialog.destroyed.connect(lambda *_args: setattr(self, "channel_config_dialog", None))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @Slot(object)
    def _apply_channel_config(self, channels: tuple[ChannelConfig, ...]) -> None:
        channels = tuple(channels)
        channel_config_changed = channels != self.channel_configs()
        valid_names = {channel.name for channel in channels}
        fallback = channels[0].name
        radiator_assignments_changed = any(radiator.channel not in valid_names for radiator in self.all_radiators())
        can_resynthesize = (
            not radiator_assignments_changed
            and self.live_dataset is not None
            and self.live_dataset.supports_channel_resynthesis
        )
        if not channel_config_changed and not radiator_assignments_changed:
            self.show_status("Channel config unchanged")
            return
        if not can_resynthesize and not self._confirm_clear_solved_data():
            return

        self._save_channel_config(channels)
        self._geometry_store().reassign_channels(valid_names, fallback)
        if can_resynthesize:
            self.live_dataset.set_channel_synthesis(
                channels,
                flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
            )
            self.refresh_plots()
            self.set_balloon_plot_available(self.live_dataset.has_balloon_data)
            balloon_window = self.balloon_window
            if balloon_window is not None and balloon_window.isVisible():
                refresh_balloon = getattr(balloon_window, "refresh_from_latest_results", None)
                if callable(refresh_balloon):
                    refresh_balloon()
            self.show_status(f"Channel config updated: {len(channels)} channels; plots resynthesized")
        else:
            self.solve_results_invalidated.emit("channel_config_changed")
            self.show_status(f"Channel config updated: {len(channels)} channels")
