from __future__ import annotations

import configparser

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt


class SettingsDialog(QDialog):
    """Config-backed settings dialog."""

    def __init__(self, config: configparser.ConfigParser, parent=None) -> None:
        super().__init__()
        self._config = config
        self.setWindowTitle("Settings")
        self.resize(520, 380)

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.noise_toggle = QCheckBox("Enable noise reduction")
        self.noise_toggle.setChecked(
            config.getboolean("transcription", "noise_reduction", fallback=True)
        )
        form.addRow("Noise Reduction", self.noise_toggle)

        self.beam_slider = QSlider(Qt.Orientation.Horizontal)
        self.beam_slider.setMinimum(1)
        self.beam_slider.setMaximum(10)
        self.beam_slider.setValue(config.getint("transcription", "beam_size", fallback=5))
        form.addRow("Beam Size", self.beam_slider)

        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["int8", "float16", "float32"])
        current_compute = config.get("transcription", "compute_type", fallback="int8")
        idx = self.compute_combo.findText(current_compute)
        self.compute_combo.setCurrentIndex(max(0, idx))
        form.addRow("Compute Type", self.compute_combo)

        self.vad_slider = QSlider(Qt.Orientation.Horizontal)
        self.vad_slider.setMinimum(1)
        self.vad_slider.setMaximum(99)
        threshold = config.getfloat("transcription", "vad_threshold", fallback=0.5)
        self.vad_slider.setValue(max(1, min(99, int(round(threshold * 100)))))
        form.addRow("VAD Threshold", self.vad_slider)

        self.default_formats = QLineEdit(
            config.get("export", "default_formats", fallback="txt")
        )
        self.default_formats.setPlaceholderText("txt,srt,json,docx,pdf")
        form.addRow("Default Export", self.default_formats)

        self.model_path = QLineEdit(
            config.get("transcription", "model_path", fallback="turkish_whisper_for_noisy_datas")
        )
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path)
        self.model_browse = QPushButton("Browse")
        self.model_browse.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_browse)
        form.addRow("Model Path", model_row)

        self.auto_download_toggle = QCheckBox("Allow model download if local model is missing")
        self.auto_download_toggle.setChecked(
            config.getboolean("transcription", "auto_download_model", fallback=False)
        )
        form.addRow("Online Fallback", self.auto_download_toggle)

        self.export_dir = QLineEdit(config.get("export", "export_dir", fallback="outputs"))
        export_row = QHBoxLayout()
        export_row.addWidget(self.export_dir)
        self.export_browse = QPushButton("Browse")
        self.export_browse.clicked.connect(self._browse_export_dir)
        export_row.addWidget(self.export_browse)
        form.addRow("Export Folder", export_row)

        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _browse_model(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if selected:
            self.model_path.setText(selected)

    def _browse_export_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if selected:
            self.export_dir.setText(selected)

    def apply_to_config(self, config: configparser.ConfigParser) -> None:
        config.set(
            "transcription",
            "noise_reduction",
            "true" if self.noise_toggle.isChecked() else "false",
        )
        config.set("transcription", "beam_size", str(self.beam_slider.value()))
        config.set("transcription", "compute_type", self.compute_combo.currentText())
        config.set("transcription", "vad_threshold", f"{self.vad_slider.value() / 100.0:.2f}")
        config.set("transcription", "model_path", self.model_path.text().strip())
        config.set(
            "transcription",
            "auto_download_model",
            "true" if self.auto_download_toggle.isChecked() else "false",
        )
        config.set("export", "default_formats", self.default_formats.text().strip() or "txt")
        config.set("export", "export_dir", self.export_dir.text().strip() or "outputs")
