"""Concise status text for completed solve frequencies."""

from blab.solvers.base import FrequencyResult


def format_frequency_solve_timings(result: FrequencyResult) -> str:
    timings = result.timings
    return f"Assembly {timings.assembly_s:.2f}s | Solve {timings.solve_s:.2f}s | Field {timings.field_s:.2f}s"


def format_frequency_completion(result: FrequencyResult, completed: int, total: int) -> str:
    return f"Solved {completed}/{total} ({result.freq_hz:.1f} Hz) | {format_frequency_solve_timings(result)}"
