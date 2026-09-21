"""Custom QWidget editor; the application owns the surrounding dock and buttons."""

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QLabel, QWidget


class BoxEditor:
    def __init__(self, parent, host):
        self.host = host
        self.widget = QWidget(parent)
        form = QFormLayout(self.widget)
        self.spins = {}
        self.source = {}
        self.revision = ""
        for key, label in (("width_m", "Width"), ("height_m", "Height"), ("depth_m", "Depth")):
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(0.001, 10.0)
            spin.setSuffix(" m")
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self.edited)
            self.spins[key] = spin
            form.addRow(label, spin)
        self.status = QLabel("Use Generate to create the mesh.")
        self.status.setWordWrap(True)
        form.addRow(self.status)
        # Host automatically releases subscriptions when this editor is closed.
        self.host.services.subscribe(self.event)

    def apply_source(self, source, revision):
        self.source, self.revision = dict(source), revision
        for key, spin in self.spins.items():
            blocked = spin.blockSignals(True)
            spin.setValue(float(source[key]))
            spin.blockSignals(blocked)

    def edited(self):
        source = self.source | {key: spin.value() for key, spin in self.spins.items()}
        snapshot = self.host.update_source(source, expected_revision=self.revision)
        self.source, self.revision = snapshot.source, snapshot.revision

    def event(self, event):
        if event.kind == "geometry_accepted":
            self.status.setText("Mesh accepted. Ready to solve with the host frequency settings.")
        elif event.job is not None:
            self.status.setText(event.job.message)

    def set_operation_state(self, state):
        for spin in self.spins.values():
            spin.setEnabled(not state.active)

    def dispose(self):
        for spin in self.spins.values():
            spin.valueChanged.disconnect(self.edited)


def create_editor(parent, document_host):
    return BoxEditor(parent, document_host)
