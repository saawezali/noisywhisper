from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt


class SettingsDialog(QDialog):
    """Phase-2 settings dialog shell."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Settings")
        self.resize(420, 320)

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.noise_toggle = QCheckBox("Enable noise reduction")
        self.noise_toggle.setChecked(True)
        form.addRow("Noise Reduction", self.noise_toggle)

        self.beam_slider = QSlider(Qt.Orientation.Horizontal)
        self.beam_slider.setMinimum(1)
        self.beam_slider.setMaximum(10)
        self.beam_slider.setValue(5)
        form.addRow("Beam Size", self.beam_slider)

        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["auto", "int8", "float16", "float32"])
        form.addRow("Compute Type", self.compute_combo)

        self.vad_slider = QSlider(Qt.Orientation.Horizontal)
        self.vad_slider.setMinimum(1)
        self.vad_slider.setMaximum(99)
        self.vad_slider.setValue(50)
        form.addRow("VAD Threshold", self.vad_slider)

        self.default_formats = QLabel("TXT")
        form.addRow("Default Export", self.default_formats)

        root.addLayout(form)

        buttons = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.cancel_button)
        root.addLayout(buttons)
