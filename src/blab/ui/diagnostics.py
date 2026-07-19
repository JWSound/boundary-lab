"""Diagnostics collection and display for the Boundary Lab GUI."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import datetime
from typing import Any

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from blab import __version__
from blab.startup_checks import DEPENDENCY_NAMES
from blab.ui.settings import GuiPreferences

SENSITIVE_DIAGNOSTIC_KEYS = {"server_url", "solve_server_url"}


def collect_diagnostics(
    preferences: GuiPreferences,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preference_values = asdict(preferences)
    solve_server_url = str(preference_values.pop("solve_server_url", "")).strip()
    context_values = _without_sensitive_fields(context or {})
    if solve_server_url:
        context_values = _redact_sensitive_values(context_values, (solve_server_url,))
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "boundary_lab_version": __version__,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "system_ram": _system_ram_text(),
        "dependencies": _dependency_versions(),
        "opencl": _opencl_devices(),
        "preferences": preference_values,
        "context": context_values,
    }


def format_diagnostics(diagnostics: dict[str, Any]) -> str:
    lines = [
        f"Generated: {diagnostics.get('generated_at', 'unknown')}",
        f"Boundary Lab: {diagnostics.get('boundary_lab_version', 'unknown')}",
        f"Python: {diagnostics.get('python', 'unknown')}",
        f"OS: {diagnostics.get('platform', 'unknown')}",
        f"CPU: {diagnostics.get('processor', 'unknown')}",
        f"System RAM: {diagnostics.get('system_ram', 'unknown')}",
        "",
        "Dependencies:",
    ]
    dependencies = diagnostics.get("dependencies", {})
    for name in DEPENDENCY_NAMES:
        lines.append(f"  {name}: {dependencies.get(name, 'not installed')}")

    lines.extend(["", "OpenCL:"])
    opencl = diagnostics.get("opencl", [])
    if opencl:
        for item in opencl:
            lines.append(f"  {item}")
    else:
        lines.append("  unavailable")

    lines.extend(["", "Preferences:"])
    preferences = diagnostics.get("preferences", {})
    for key in sorted(preferences):
        lines.append(f"  {key}: {preferences[key]}")

    context = diagnostics.get("context", {})
    if isinstance(context, Mapping):
        for section, values in context.items():
            lines.extend(["", f"{str(section).replace('_', ' ').title()}:"])
            if isinstance(values, Mapping):
                _append_mapping(lines, values)
            else:
                lines.append(f"  {values}")

    return "\n".join(lines)


class DiagnosticsDialog(QDialog):
    def __init__(
        self,
        preferences: GuiPreferences,
        parent: QWidget | None = None,
        *,
        context_provider: Callable[[], Mapping[str, Any]] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Diagnostic Info")
        self.resize(760, 560)
        self._preferences = preferences
        self._context_provider = context_provider

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.run_button = QPushButton("Run Diagnostics")
        self.run_button.clicked.connect(self.run_diagnostics)
        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        action_row = QHBoxLayout()
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.copy_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        layout.addWidget(self.output, 1)
        layout.addWidget(buttons)

    def run_diagnostics(self) -> None:
        try:
            context = self._context_provider() if self._context_provider is not None else {}
        except Exception as exc:
            context = {"diagnostic context": {"error": f"{type(exc).__name__}: {exc}"}}
        self.output.setPlainText(format_diagnostics(collect_diagnostics(self._preferences, context)))

    def copy_to_clipboard(self) -> None:
        if not self.output.toPlainText().strip():
            self.run_diagnostics()
        QApplication.clipboard().setText(self.output.toPlainText())


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for name in DEPENDENCY_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _opencl_devices() -> list[str]:
    try:
        import pyopencl as cl

        devices = []
        for platform_info in cl.get_platforms():
            for device in platform_info.get_devices():
                devices.append(f"{platform_info.name}: {device.name}")
        return devices
    except Exception:
        return []


def _system_ram_text() -> str:
    if sys.platform != "win32":
        return "unknown"

    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return f"{status.ullTotalPhys / (1024**3):.1f} GiB"
    except Exception:
        return "unknown"

    return "unknown"


def _without_sensitive_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_sensitive_fields(item)
            for key, item in value.items()
            if str(key).strip().lower() not in SENSITIVE_DIAGNOSTIC_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_without_sensitive_fields(item) for item in value]
    return value


def _redact_sensitive_values(value: Any, sensitive_values: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redact_sensitive_values(item, sensitive_values)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_sensitive_values(item, sensitive_values) for item in value]
    if isinstance(value, str):
        for sensitive_value in sensitive_values:
            value = value.replace(sensitive_value, "[hidden]")
    return value


def _append_mapping(lines: list[str], values: Mapping[str, Any], *, indent: int = 2) -> None:
    prefix = " " * indent
    for key, value in values.items():
        label = str(key).replace("_", " ")
        if isinstance(value, Mapping):
            lines.append(f"{prefix}{label}:")
            _append_mapping(lines, value, indent=indent + 2)
        elif isinstance(value, (list, tuple)):
            lines.append(f"{prefix}{label}: {', '.join(str(item) for item in value) or 'none'}")
        else:
            lines.append(f"{prefix}{label}: {value}")
