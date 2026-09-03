"""The narrow seam between workflow logic and the widget tree.

Workflow controllers must not reach for ``QPushButton``/``QLabel`` instances:
that is exactly the coupling that makes a UI revamp expensive. They depend on
:class:`WorkflowView` instead, and decide *what* state the controls should be
in via the pure :func:`workflow_controls` mapping, which is unit-testable
without a Qt application.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from blab.ui.application_state import OperationPhase

BUSY_PHASES = frozenset({OperationPhase.RUNNING, OperationPhase.CANCELLING})


class UnsavedChoice(Enum):
    """What the user chose when warned that a project has unsaved work.

    The view reports the choice and nothing more; deciding what to do with it
    — save first, throw the work away, or abandon the action — is the
    workflow's call, which is what keeps that rule testable without a dialog.
    """

    SAVE = "save"
    DISCARD = "discard"
    CANCEL = "cancel"


@dataclass(frozen=True)
class WorkflowControls:
    """Desired enabled-state of the primary workflow controls."""

    generate: bool
    solve: bool
    cancel: bool
    mesh_config: bool
    system_config: bool
    channel_config: bool
    speaker_package: bool


def workflow_controls(
    phase: OperationPhase,
    *,
    has_solver_meshes: bool,
    cancel_available: bool = True,
) -> WorkflowControls:
    """Map an operation phase to control availability.

    While an operation runs, every entry point that would mutate its inputs is
    disabled and only Cancel stays live; once cancellation has been requested
    Cancel is disabled too, so it cannot be pressed twice. When idle, System
    Config is additionally gated on there being meshes to configure.

    ``cancel_available`` covers the case of an operation that is running but not
    yet cancellable — geometry generation withholds Stop for a few seconds so a
    short run cannot be interrupted mid-write. It only ever withholds Cancel; it
    never re-enables the inputs.
    """
    if phase in BUSY_PHASES:
        return WorkflowControls(
            generate=False,
            solve=False,
            cancel=cancel_available and phase is OperationPhase.RUNNING,
            mesh_config=False,
            system_config=False,
            channel_config=False,
            speaker_package=False,
        )
    return WorkflowControls(
        generate=True,
        solve=True,
        cancel=False,
        mesh_config=True,
        system_config=has_solver_meshes,
        channel_config=True,
        speaker_package=has_solver_meshes,
    )


@dataclass(frozen=True)
class FrequencyRange:
    """The solve frequency range as chosen in the UI.

    Carried as a value rather than read off three spin boxes, so workflow code
    never needs to know that spin boxes are how the user expresses it.
    """

    min_hz: int
    max_hz: int
    count: int

    def normalized(self) -> "FrequencyRange":
        """Swap the endpoints if the user dragged min above max."""
        return FrequencyRange(
            min_hz=min(self.min_hz, self.max_hz),
            max_hz=max(self.min_hz, self.max_hz),
            count=self.count,
        )

    @property
    def span_hz(self) -> int:
        normalized = self.normalized()
        return normalized.max_hz - normalized.min_hz


@runtime_checkable
class SolveInputs(Protocol):
    """The domain side of a solve: what to solve, and with what settings.

    Everything here survives a UI revamp untouched — it describes meshes,
    radiators and channels, never widgets. Paired with :class:`WorkflowView`
    and :class:`PlotPresenter`, it is the complete set of collaborators the
    solve workflow needs, which is what makes the workflow extractable from
    the window.
    """

    def has_solver_meshes(self) -> bool:
        """Whether there is anything to solve."""

    def mesh_entries_for_symmetry(self, symmetry: str):
        """Mesh entries reduced for the given symmetry setting."""

    def mesh_service(self):
        """The geometry preparation service."""

    def ensure_seeded_exterior_system(self, *, required: bool = False) -> bool:
        """Represent legacy exterior sources as a physical system."""

    def channel_configs(self) -> tuple:
        """The configured channels."""

    def solver_channel_configs(self, radiators: tuple) -> tuple:
        """Channels narrowed to those driving the given radiators."""

    def reconcile_symmetry_with_backend(self) -> bool:
        """Downgrade project symmetry if the backend cannot honour it."""


@runtime_checkable
class GeometryInputs(Protocol):
    """The domain side of geometry generation.

    Like :class:`SolveInputs`, everything here describes designs, geometry and
    meshes rather than widgets, so it survives a UI revamp untouched.
    """

    def active_generator_document(self):
        """The design document currently selected for editing, if any."""

    def apply_saved_source_config_to_result(self, result, mesh_name: str):
        """Re-apply the project's stored source settings to fresh geometry."""

    def record_generated_geometry(self, document_id: str, result) -> None:
        """Store generated geometry against its design and update the document."""

    def ensure_seeded_exterior_system(self, *, required: bool = False) -> bool:
        """Create a default exterior physical system if none is configured."""


@runtime_checkable
class ProjectInputs(Protocol):
    """The domain side of opening, saving and applying a project.

    Wider than :class:`SolveInputs` because a project genuinely touches more:
    it is the one operation that rewrites every other kind of state at once.
    Everything here still describes designs, meshes and saved configuration
    rather than widgets, so it survives a UI revamp untouched.
    """

    def source_config_by_name(self) -> dict:
        """Saved per-surface source settings, keyed by surface name."""

    def channel_config_by_name(self) -> dict:
        """Saved channel settings, keyed by channel name."""

    def discard_channel_config_dialog(self) -> None:
        """Drop any open channel editor, whose contents a load invalidates."""

    def rebuild_generator_document_tabs(self) -> None:
        """Re-derive the design editors from the current documents."""

    def result_from_generator_document(self, document):
        """Recover generated geometry recorded against a saved design."""

    def apply_saved_source_config_to_result(self, result, mesh_name: str):
        """Re-apply the project's stored source settings to geometry."""

    def apply_saved_imported_source_config(self, surface_tags: dict) -> None:
        """Rebuild imported-mesh radiators from saved source settings."""

    def surface_tags_for_meshes(self) -> dict:
        """Physical surface names and tags across the imported meshes."""

    def reconcile_symmetry_with_backend(self) -> bool:
        """Downgrade project symmetry if the backend cannot honour it."""

    def ensure_seeded_exterior_system(self, *, required: bool = False) -> bool:
        """Create a default exterior physical system if none is configured."""

    def active_generator_document(self):
        """The design document currently selected for editing, if any."""


@runtime_checkable
class PlotPresenter(Protocol):
    """Everything workflow code may ask of the plotting layer.

    Deliberately narrow and public: a UI revamp replaces the implementation
    wholesale, and this is the contract the replacement has to honour. Anything
    still private on the presenter is an implementation detail and must not be
    called from workflow code.
    """

    def refresh_plots(self, *, active_only: bool = False):
        """Recompute and redraw canvases that are actually exposed on screen."""

    def clear_plots(self) -> None:
        """Drop all plotted data, leaving empty axes."""

    def request_live_refresh(self) -> None:
        """Ask for a redraw, coalesced with any already pending."""

    def flush_live_refresh(self) -> None:
        """Run a pending coalesced redraw now."""

    def cancel_live_refresh(self) -> None:
        """Discard any pending coalesced redraw."""

    def prepared_live_dataset(self, **options):
        """Project the current solve results for plotting, without drawing."""

    def apply_last_completed_comparison(self) -> None:
        """Overlay the previous completed solve for comparison."""

    def clear_comparison_history(self) -> None:
        """Forget the stored previous solve."""

    def set_spherical_spin_available(self, available: bool) -> None:
        """Enable spherical spin calculations when a solve contains sphere samples."""

    def refresh_contour_controls(self) -> None:
        """Re-evaluate whether contour capture/clear are currently offered."""

    def visible_isobar_plots(self) -> tuple:
        """The isobar canvases currently on screen."""

    def capture_isobar_contours(self, plot_id: str) -> None:
        """Freeze the drawn contours of one isobar plot."""

    def clear_isobar_contours(self, plot_id: str) -> None:
        """Remove captured contours from one isobar plot."""


@runtime_checkable
class WorkflowView(Protocol):
    """What a workflow controller is allowed to ask of the UI."""

    def show_status(self, message: str) -> None:
        """Display a short progress/status message to the user."""

    def warn(self, title: str, message: str) -> None:
        """Tell the user why an action could not proceed.

        Workflow code must not construct dialogs itself: a ``QMessageBox``
        needs a parent widget, which would put the window back in its
        constructor signature.
        """

    def show_error(self, title: str, message: str) -> None:
        """Report a failure that interrupted work already under way."""

    def confirm(self, title: str, message: str) -> bool:
        """Ask a yes/no question, returning True for yes."""

    def confirm_mesh_topology_warning(self, report) -> bool:
        """Offer a cancel-safe override for an exterior mesh topology warning."""

    def show_mesh_topology_issues(self, report) -> None:
        """Highlight invalid exterior mesh edges, or clear an earlier highlight."""

    def ask_unsaved_changes(self, *, closing: bool) -> UnsavedChoice:
        """Warn that the project has unsaved work and report the choice.

        ``closing`` only changes the wording: closing the window discards the
        work outright, whereas every other action can still save first.
        """

    def choose_open_file(self, title: str, file_filter: str) -> Path | None:
        """Ask the user to pick an existing file. None if they cancelled.

        Workflow code must not reach a file dialog directly: like a
        ``QMessageBox`` it needs a parent widget, and choosing a file is a
        view concern that a different frontend answers its own way.
        """

    def choose_save_file(self, title: str, file_filter: str, suggested_name: str) -> Path | None:
        """Ask the user where to write a file. None if they cancelled."""

    def show_stitch_or_generic_error(self, title: str, exc: Exception) -> None:
        """Report a mesh preparation failure, expanding the stitch case.

        A view concern rather than a domain one: the difference is whether the
        cause is worth a details pane, not what went wrong.
        """

    def show_mesh_quality_warning(self, result) -> None:
        """Surface any mesh quality warnings carried by generated geometry."""

    def set_busy_cursor(self, busy: bool) -> None:
        """Show or clear the wait cursor for a long-running operation."""

    def apply_workflow_controls(self, controls: WorkflowControls) -> None:
        """Reflect the given control availability in the UI."""

    def set_workflow_phase(self, phase: OperationPhase, *, cancel_available: bool = True) -> None:
        """Map an operation phase through :func:`workflow_controls` and apply it.

        ``cancel_available`` withholds Cancel for an operation that is running
        but not yet interruptible; it never re-enables the inputs.
        """

    def frequency_range(self) -> FrequencyRange:
        """The frequency range currently chosen by the user."""

    def set_frequency_range(self, value: FrequencyRange) -> None:
        """Show the given range without emitting change signals."""

    def set_plot_exports_available(self, available: bool) -> None:
        """Offer or withdraw the per-plot export actions."""

    def set_balloon_plot_available(self, available: bool) -> None:
        """Offer or withdraw the balloon plot entry point."""

    def set_max_spl_available(self, available: bool) -> None:
        """Offer or withdraw maximum-SPL configuration."""

    def set_max_spl_export_available(self, available: bool) -> None:
        """Offer or withdraw image export for a calculated maximum-SPL plot."""

    def set_system_config_available(self, available: bool) -> None:
        """Offer or withdraw the System Config entry point.

        Narrower than apply_workflow_controls: mesh availability can change
        while an operation is running, and must not re-enable the rest.
        """

    def clear_mesh_preview(self) -> None:
        """Empty the 3D preview."""

    def show_mesh_preview(self, meshes, **options) -> None:
        """Render the given solver meshes in the 3D preview."""
