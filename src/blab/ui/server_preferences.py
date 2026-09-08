"""Server connection settings with a nonblocking capability check."""

import os
import queue
import threading

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFormLayout, QLabel, QLineEdit, QPushButton, QWidget

from blab.remote import RemoteBackend


class ServerPreferences(QWidget):
    def __init__(self, preferences, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.url = QLineEdit(preferences.solve_server_url)
        self.token_env = QLineEdit(preferences.solve_server_token_env)
        self.token_env.setToolTip(
            "Name of the environment variable containing the token. The token itself is not saved in preferences."
        )
        self.ca = QLineEdit(preferences.solve_server_ca)
        self.ca.setPlaceholderText("Optional PEM CA/certificate path")
        self.check = QPushButton("Check connection")
        self.status = QLabel("")
        self.status.hide()
        self.status.setWordWrap(True)
        form.addRow("Address", self.url)
        form.addRow("Token environment variable", self.token_env)
        form.addRow("Trusted certificate", self.ca)
        form.addRow(self.check)
        form.addRow(self.status)
        self.check.clicked.connect(self.check_connection)
        self._queue = queue.Queue()
        self._revision = 0
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        for edit in (self.url, self.token_env, self.ca):
            edit.textChanged.connect(self.invalidate)

    def invalidate(self, *_args):
        self._revision += 1
        self.status.show()
        self.status.setText("Connection settings changed. Check connection again.")

    def check_connection(self):
        url, token_env, ca = self.url.text().strip(), self.token_env.text().strip(), self.ca.text().strip()
        revision = self._revision
        results = self._queue
        self.check.setEnabled(False)
        self.status.show()
        self.status.setText("Connecting…")

        def work():
            try:
                info = RemoteBackend(url, token=os.environ.get(token_env), ca_file=ca or None).check_capabilities()
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
