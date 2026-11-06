"""Concise status text for completed solve frequencies."""

from blab.solve_results.live_projection import frequency_result_timings
from blab.system_contract import SystemFrequencyResult


def format_frequency_solve_timings(result: SystemFrequencyResult) -> str:
    timings = frequency_result_timings(result)
    return f"Assembly {timings.assembly_s:.2f}s | Solve {timings.solve_s:.2f}s | Field {timings.field_s:.2f}s"
