"""Server connection settings with a nonblocking capability check."""

import queue
import secrets
import threading

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QCheckBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from blab.remote import RemoteBackend


class ServerPreferences(QWidget):
    def __init__(self, preferences, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.url = QLineEdit(preferences.solve_server_url)
        self.access_key = QLineEdit(preferences.solve_server_access_key)
        self.access_key.setEchoMode(QLineEdit.Password)
        self.access_key.setPlaceholderText("Optional on private networks")
        self.access_key.setToolTip(
            "Save this key locally in a safe place so you can paste it again next time. "
            "Boundary Lab keeps it only for this session and does not save it. "
            "For a hosted server, copy the same key into the container's BLAB_SERVER_TOKEN secret."
        )
        self.generate = QPushButton("Generate")
        self.copy = QPushButton("Copy")
        self.show_key = QCheckBox("Show")
        key_actions = QWidget()
        actions_layout = QHBoxLayout(key_actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addWidget(self.generate)
        actions_layout.addWidget(self.copy)
        actions_layout.addWidget(self.show_key)
        self.generate.clicked.connect(lambda: self.access_key.setText(secrets.token_urlsafe(32)))
        self.copy.clicked.connect(lambda: QApplication.clipboard().setText(self.access_key.text()))
        self.show_key.toggled.connect(
            lambda shown: self.access_key.setEchoMode(QLineEdit.Normal if shown else QLineEdit.Password)
        )
        self.check = QPushButton("Check connection")
        self.status = QLabel("")
        self.status.hide()
        self.status.setWordWrap(True)
        form.addRow("Address", self.url)
        form.addRow("Access key", self.access_key)
        form.addRow(key_actions)
        form.addRow(self.check)
        form.addRow(self.status)
        self.check.clicked.connect(self.check_connection)
        self._queue = queue.Queue()
        self._revision = 0
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        for edit in (self.url, self.access_key):
            edit.textChanged.connect(self.invalidate)

    def invalidate(self, *_args):
        self._revision += 1
        self.status.show()
        self.status.setText("Connection settings changed. Check connection again.")

    def check_connection(self):
        url, token = self.url.text().strip(), self.access_key.text().strip()
        revision = self._revision
        results = self._queue
        self.check.setEnabled(False)
        self.status.show()
        self.status.setText("Connecting…")

        def work():
            try:
                info = RemoteBackend(url, token=token or None).check_capabilities()
                state = info.get("state", "ready")
                message = {
                    "ready": "Connected. Ready to solve.",
                    "starting": "Connected. Server is starting.",
                    "unavailable": "Connected. Server is not ready to solve.",
                }.get(state, "Connected.")
            except Exception as exc:
                message = f"Connection failed: {exc}"
            results.put((revision, message))

        threading.Thread(target=work, daemon=True).start()
        self.timer.start()

    def poll(self):
        try:
            revision, message = self._queue.get_nowait()
        except queue.Empty:
            return
        self.timer.stop()
        self.check.setEnabled(True)
        if revision == self._revision:
            self.status.setText(message)
