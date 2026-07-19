"""Typed application-operation state and solve invalidation policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OperationPhase(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class OperationState:
    phase: OperationPhase = OperationPhase.IDLE
    message: str = ""

    @property
    def active(self) -> bool:
        return self.phase in {OperationPhase.RUNNING, OperationPhase.CANCELLING}


@dataclass(frozen=True)
class SolveCompletion:
    phase: OperationPhase
    solved_count: int
    expected_count: int
    elapsed_s: float

    @property
    def completed(self) -> bool:
        return (
            self.phase == OperationPhase.COMPLETED
            and self.expected_count > 0
            and self.solved_count >= self.expected_count
        )


class SolveInvalidationReason(StrEnum):
    ATH_SCRIPT_RENAMED = "ath_script_renamed"
    ATH_SCRIPT_REMOVED = "ath_script_removed"
    IMPORTED_MESH_FILES_RELOADED = "imported_mesh_files_reloaded"
    NEW_PROJECT = "new_project"
    PROJECT_LOADED = "project_loaded"
    PREFERENCES_CHANGED = "preferences_changed"
    MESH_CONFIG_CHANGED = "mesh_config_changed"
    CHANNEL_CONFIG_CHANGED = "channel_config_changed"
    SOURCE_CONFIG_CHANGED = "source_config_changed"
    GEOMETRY_GENERATION_STARTED = "geometry_generation_started"


@dataclass(frozen=True)
class SolveInvalidationPolicy:
    clear_solve_results: bool = True
    clear_comparison_history: bool = False


_POLICIES = {
    SolveInvalidationReason.NEW_PROJECT: SolveInvalidationPolicy(clear_comparison_history=True),
    SolveInvalidationReason.PROJECT_LOADED: SolveInvalidationPolicy(clear_comparison_history=True),
}


def solve_invalidation_policy(reason: str | SolveInvalidationReason) -> SolveInvalidationPolicy:
    """Return the intentional destructive invalidation behavior for a change.

    Unknown reasons remain conservative and clear current solve results. This
    preserves Boundary Lab's rapid tweak/regenerate/solve workflow while making
    the policy explicit and independently testable.
    """
    try:
        normalized = SolveInvalidationReason(str(reason))
    except ValueError:
        return SolveInvalidationPolicy()
    return _POLICIES.get(normalized, SolveInvalidationPolicy())
