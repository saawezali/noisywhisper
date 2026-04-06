from __future__ import annotations

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """Phase-2 GUI shell. CLI pipeline is implemented in v0.1."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NoisyWhisper")
        self.resize(980, 680)

        root = QWidget()
        layout = QVBoxLayout(root)

        title = QLabel("NoisyWhisper")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)

        self.drop_area = QLabel("Drag and drop audio file here or click Browse")
        self.drop_area.setStyleSheet(
            "border: 2px dashed #909090; border-radius: 8px; padding: 30px;"
        )
        self.drop_area.setMinimumHeight(120)
        layout.addWidget(self.drop_area)

        button_row = QHBoxLayout()
        self.browse_button = QPushButton("Browse File")
        self.transcribe_button = QPushButton("Transcribe")
        self.settings_button = QPushButton("Settings")
        button_row.addWidget(self.browse_button)
        button_row.addWidget(self.transcribe_button)
        button_row.addWidget(self.settings_button)
        layout.addLayout(button_row)

        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setPlaceholderText("Transcript preview appears here...")
        layout.addWidget(self.transcript_view)

        self.setCentralWidget(root)
