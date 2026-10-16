"""Nonblocking setup for optional, user-installed macOS generator tools."""

from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from blab.ath_setup import ATH_PATH_KEY, TEST_CONFIG, WINE_PATH_KEY, discover_wine, validate_paths, wine_prefix


class AthSetupDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Ath Setup")
        self.resize(650, 320)
        self._temporary = None
        self._stage = ""
        self._output = bytearray()
        self._errors = bytearray()
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Ath generation is optional. Mesh import and solving do not require Ath or Wine.\n"
            'Download <a href="https://at-horns.eu/">Ath</a> separately under its own license, and install '
            '<a href="https://github.com/Gcenx/macOS_Wine_builds">Wine for macOS</a>. '
            "An Intel Wine build may also require Apple's Rosetta on Apple Silicon."
        )
        intro.setWordWrap(True)
        intro.setOpenExternalLinks(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.ath = QLineEdit(str(settings.value(ATH_PATH_KEY, "")))
        self.wine = QLineEdit(str(settings.value(WINE_PATH_KEY, "")))
        self._controls = [self.ath, self.wine]
        for label, field, file_filter in (
            ("Ath executable", self.ath, "Ath executable (*.exe)"),
            ("Wine executable", self.wine, "All files (*)"),
        ):
            row = QWidget()
            horizontal = QHBoxLayout(row)
            horizontal.setContentsMargins(0, 0, 0, 0)
            horizontal.addWidget(field)
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _checked=False, edit=field, pattern=file_filter: self._browse(edit, pattern))
            horizontal.addWidget(browse)
            self._controls.append(browse)
            form.addRow(label, row)
        layout.addLayout(form)
        hint = QLabel(
            "Choose the Wine executable inside its app bundle, usually Contents/Resources/wine/bin/wine. You can also paste its full path."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        actions = QHBoxLayout()
        detect = QPushButton("Find Wine")
        detect.clicked.connect(self._detect)
        actions.addWidget(detect)
        self.test_button = QPushButton("Test configuration")
        self.test_button.clicked.connect(self._test)
        actions.addWidget(self.test_button)
        self._controls.extend([detect, self.test_button])
        layout.addLayout(actions)
        self.status = QLabel(
            "Setup required"
            if not self.ath.text() or not self.wine.text()
            else "Saved paths — test to verify generation."
        )
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.readyReadStandardError.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._timeout)
        for field in (self.ath, self.wine):
            field.textChanged.connect(lambda: self.status.setText("Paths changed — test to verify generation."))

    def _browse(self, field, pattern):
        path, _ = QFileDialog.getOpenFileName(self, "Select executable", field.text(), pattern)
        if path:
            field.setText(path)

    def _detect(self):
        path = discover_wine()
        if path:
            self.wine.setText(path)
        else:
            self.status.setText("Wine was not found. Install it separately, then browse or paste its executable path.")

    def accept(self):
        if self._stage:
            return
        try:
            if self.ath.text().strip() or self.wine.text().strip():
                ath, wine = validate_paths(self.ath.text(), self.wine.text())
                values = str(ath), str(wine)
            else:
                values = "", ""
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        for key, value in zip((ATH_PATH_KEY, WINE_PATH_KEY), values):
            self.settings.setValue(key, value)
        self.settings.sync()
        super().accept()

    def reject(self):
        if self._stage:
            self.status.setText("Stopping configuration test…")
            self._stage = "cancelled"
            self.process.kill()
            return
        super().reject()

    def closeEvent(self, event):
        if self._stage:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)

    def _test(self):
        try:
            self._ath_path, self._wine_path = validate_paths(self.ath.text(), self.wine.text())
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._temporary = TemporaryDirectory(prefix="boundary-ath-check-")
        root = Path(self._temporary.name)
        self._config = root / "check.cfg"
        self._config.write_text(TEST_CONFIG, encoding="utf-8")
        environment = QProcessEnvironment.systemEnvironment()
        prefix = wine_prefix()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        environment.insert("WINEPREFIX", str(prefix))
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(root))
        for control in self._controls:
            control.setEnabled(False)
        self.buttons.button(QDialogButtonBox.Save).setEnabled(False)
        self.status.setText("Checking Wine, then generating a small Ath geometry. First use may take several minutes…")
        self._start("version", ["--version"])

    def _start(self, stage, arguments):
        self._stage = stage
        self._output.clear()
        self._errors.clear()
        self.timer.start(180000)
        self.process.start(str(self._wine_path), arguments)

    def _read(self):
        self._output.extend(bytes(self.process.readAllStandardOutput()))
        self._errors.extend(bytes(self.process.readAllStandardError()))
        # Bound diagnostics while continuing to drain both pipes.
        del self._output[:-262144]
        del self._errors[:-16384]

    def _finished(self, code, exit_status):
        self._read()
        self.timer.stop()
        if self._stage in ("cancelled", "timeout"):
            self._done(
                "Test cancelled."
                if self._stage == "cancelled"
                else "Test timed out. Check Wine and Rosetta, then try again."
            )
        elif code != 0 or exit_status != QProcess.NormalExit:
            self._done("Test failed:\n" + self._errors.decode(errors="replace")[-4000:])
        elif self._stage == "version":
            self._version = self._output.decode(errors="replace").strip()
            self._start("geometry", [str(self._ath_path), str(self._config), "-b"])
        else:
            output = self._output.decode(errors="replace")
            if "Point(" in output or "Point (" in output:
                self._done(
                    f"Configuration test passed ({self._version}). Ath produced geometry. Click Save to use these paths."
                )
            else:
                self._done(
                    "Ath returned no recognizable geometry. Check that your Ath version supports batch mode (-b).\n"
                    + self._errors.decode(errors="replace")[-2000:]
                )

    def _error(self, error):
        if error == QProcess.FailedToStart:
            self._done("Wine could not start: " + self.process.errorString())

    def _timeout(self):
        self._stage = "timeout"
        self.process.kill()

    def _done(self, message):
        self.timer.stop()
        self._stage = ""
        if self._temporary:
            self._temporary.cleanup()
            self._temporary = None
        for control in self._controls:
            control.setEnabled(True)
        self.buttons.button(QDialogButtonBox.Save).setEnabled(True)
        self.status.setText(message)
