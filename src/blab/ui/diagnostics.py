"""Diagnostics collection and display for the Boundary Lab GUI."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import asdict
from typing import Any

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPushButton, QTextEdit, QVBoxLayout, QWidget

from blab import __version__
from blab.startup_checks import DEPENDENCY_NAMES
from blab.ui.settings import GuiPreferences


def collect_diagnostics(preferences: GuiPreferences) -> dict[str, Any]:
    return {
        "boundary_lab_version": __version__,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "system_ram": _system_ram_text(),
        "dependencies": _dependency_versions(),
        "opencl": _opencl_devices(),
        "preferences": asdict(preferences),
    }


def format_diagnostics(diagnostics: dict[str, Any]) -> str:
    lines = [
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

    return "\n".join(lines)


class DiagnosticsDialog(QDialog):
    def __init__(self, preferences: GuiPreferences, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Diagnostic Info")
        self.resize(760, 560)
        self._preferences = preferences

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.run_button = QPushButton("Run Diagnostics")
        self.run_button.clicked.connect(self.run_diagnostics)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.run_button)
        layout.addWidget(self.output, 1)
        layout.addWidget(buttons)

    def run_diagnostics(self) -> None:
        self.output.setPlainText(format_diagnostics(collect_diagnostics(self._preferences)))


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
            return f"{status.ullTotalPhys / (1024 ** 3):.1f} GiB"
    except Exception:
        return "unknown"

    return "unknown"
