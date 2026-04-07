from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.worker import PipelineSettings, TranscriptionPipeline
from output.docx_writer import write_docx
from output.json_writer import write_json
from output.pdf_writer import write_pdf
from output.srt_writer import write_srt
from output.txt_writer import write_txt
from ui.settings_dialog import SettingsDialog
from utils.config import load_config, save_config
from utils.logger import setup_logger
from utils.model_manager import ensure_model_available


class _TranscriptionTask(QObject):
    progress = pyqtSignal(str, int, str)
    completed = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(
        self,
        audio_file: str,
        config_path: str,
        fallback_model_path: str,
    ) -> None:
        super().__init__()
        self.audio_file = audio_file
        self.config_path = config_path
        self.fallback_model_path = fallback_model_path

    def run(self) -> None:
        try:
            config = load_config(self.config_path)
            logger = setup_logger(
                log_file=config.get("logging", "file", fallback="noisywhisper.log"),
                level=config.get("logging", "level", fallback="INFO"),
                max_mb=config.getint("logging", "max_mb", fallback=5),
                backup_count=config.getint("logging", "backup_count", fallback=3),
            )

            model_path = config.get(
                "transcription", "model_path", fallback="turkish_whisper_for_noisy_datas"
            )
            model_repo = config.get(
                "transcription",
                "model_repo_id",
                fallback="Cosmobillian/turkish_whisper_for_noisy_datas",
            )
            fallback_repo = config.get(
                "transcription",
                "fallback_model_repo_id",
                fallback="Systran/faster-whisper-small",
            )
            auto_download = config.getboolean(
                "transcription", "auto_download_model", fallback=True
            )

            resolved_model = ensure_model_available(
                model_dir=model_path,
                primary_repo_id=model_repo,
                fallback_repo_id=fallback_repo,
                auto_download=auto_download,
                logger=logger,
                quantization="int8",
            )

            fallback_path = Path(self.fallback_model_path).expanduser().resolve()
            resolved_fallback = None
            if fallback_path.exists():
                resolved_fallback = fallback_path
            elif auto_download and fallback_repo:
                resolved_fallback = ensure_model_available(
                    model_dir=str(fallback_path),
                    primary_repo_id=fallback_repo,
                    fallback_repo_id=None,
                    auto_download=True,
                    logger=logger,
                    quantization="int8",
                )

            settings = PipelineSettings(
                model_path=str(resolved_model),
                fallback_model_path=(str(resolved_fallback) if resolved_fallback else None),
                beam_size=config.getint("transcription", "beam_size", fallback=5),
                compute_type=config.get("transcription", "compute_type", fallback="int8"),
                denoise_enabled=config.getboolean(
                    "transcription", "noise_reduction", fallback=True
                ),
                vad_threshold=config.getfloat("transcription", "vad_threshold", fallback=0.5),
            )
            pipeline = TranscriptionPipeline(settings=settings, logger=logger)
            segments = pipeline.run(self.audio_file, progress_callback=self.progress.emit)
            self.completed.emit(segments)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    """NoisyWhisper main window for offline file transcription."""

    def __init__(self) -> None:
        super().__init__()
        self.config_path = "config.ini"
        self.config = load_config(self.config_path)
        self.selected_file: str | None = None
        self.segments = []
        self._thread: QThread | None = None
        self._task: _TranscriptionTask | None = None

        self.setWindowTitle("NoisyWhisper")
        self.resize(980, 680)
        self.setAcceptDrops(True)

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
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.drop_area)

        button_row = QHBoxLayout()
        self.browse_button = QPushButton("Browse File")
        self.transcribe_button = QPushButton("Transcribe")
        self.settings_button = QPushButton("Settings")
        button_row.addWidget(self.browse_button)
        button_row.addWidget(self.transcribe_button)
        button_row.addWidget(self.settings_button)
        layout.addLayout(button_row)

        self.status_label = QLabel("Stage: idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)

        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setPlaceholderText("Transcript preview appears here...")
        layout.addWidget(self.transcript_view)

        export_grid = QGridLayout()
        self.export_txt = QCheckBox("TXT")
        self.export_srt = QCheckBox("SRT")
        self.export_json = QCheckBox("JSON")
        self.export_docx = QCheckBox("DOCX")
        self.export_pdf = QCheckBox("PDF")
        self.export_txt.setChecked(True)

        export_grid.addWidget(QLabel("Export formats:"), 0, 0)
        export_grid.addWidget(self.export_txt, 0, 1)
        export_grid.addWidget(self.export_srt, 0, 2)
        export_grid.addWidget(self.export_json, 0, 3)
        export_grid.addWidget(self.export_docx, 0, 4)
        export_grid.addWidget(self.export_pdf, 0, 5)

        self.export_dir_label = QLabel(
            f"Export folder: {self.config.get('export', 'export_dir', fallback='outputs')}"
        )
        self.export_dir_button = QPushButton("Change Export Folder")
        self.export_button = QPushButton("Export Selected")

        export_grid.addWidget(self.export_dir_label, 1, 0, 1, 4)
        export_grid.addWidget(self.export_dir_button, 1, 4)
        export_grid.addWidget(self.export_button, 1, 5)

        layout.addLayout(export_grid)

        self.setCentralWidget(root)

        self.browse_button.clicked.connect(self._browse_file)
        self.transcribe_button.clicked.connect(self._start_transcription)
        self.settings_button.clicked.connect(self._open_settings)
        self.export_dir_button.clicked.connect(self._select_export_dir)
        self.export_button.clicked.connect(self._export_selected)
        self._apply_default_formats()

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        urls = event.mimeData().urls()
        if not urls:
            return
        local_file = urls[0].toLocalFile()
        if local_file:
            self._set_selected_file(local_file)

    def _browse_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.aac *.wma *.aiff *.amr *.3gp *.mkv *.webm);;All Files (*.*)",
        )
        if selected:
            self._set_selected_file(selected)

    def _set_selected_file(self, file_path: str) -> None:
        self.selected_file = file_path
        self.drop_area.setText(f"Selected: {file_path}")

    def _start_transcription(self) -> None:
        if not self.selected_file:
            QMessageBox.warning(self, "No file", "Please choose an audio file first.")
            return
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "Transcription is already running.")
            return

        self.transcript_view.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Stage: starting")
        self.transcribe_button.setEnabled(False)

        self._thread = QThread(self)
        self._task = _TranscriptionTask(
            audio_file=self.selected_file,
            config_path=self.config_path,
            fallback_model_path="models/faster-whisper-small",
        )
        self._task.moveToThread(self._thread)

        self._thread.started.connect(self._task.run)
        self._task.progress.connect(self._on_progress)
        self._task.completed.connect(self._on_completed)
        self._task.failed.connect(self._on_failed)

        self._task.completed.connect(self._thread.quit)
        self._task.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _on_progress(self, stage: str, pct: int, preview: str) -> None:
        self.status_label.setText(f"Stage: {stage}")
        self.progress_bar.setValue(max(0, min(100, int(pct))))
        if preview:
            self.transcript_view.append(preview)

    def _on_completed(self, segments: list) -> None:
        self.segments = segments
        self.transcribe_button.setEnabled(True)
        self.status_label.setText("Stage: completed")
        self.progress_bar.setValue(100)

        lines: list[str] = []
        for seg in segments:
            start_ms = int(getattr(seg, "start_ms", 0))
            hh = start_ms // 3_600_000
            mm = (start_ms % 3_600_000) // 60_000
            ss = (start_ms % 60_000) // 1000
            text = str(getattr(seg, "text", "")).strip()
            if text:
                lines.append(f"[{hh:02d}:{mm:02d}:{ss:02d}] {text}")

        self.transcript_view.setPlainText("\n".join(lines))

    def _on_failed(self, message: str) -> None:
        self.transcribe_button.setEnabled(True)
        self.status_label.setText("Stage: failed")
        QMessageBox.critical(self, "Transcription error", message)

    def _cleanup_thread(self) -> None:
        if self._task is not None:
            self._task.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._task = None
        self._thread = None

    def _apply_default_formats(self) -> None:
        defaults = {
            item.strip().lower()
            for item in self.config.get("export", "default_formats", fallback="txt").split(",")
            if item.strip()
        }
        self.export_txt.setChecked("txt" in defaults)
        self.export_srt.setChecked("srt" in defaults)
        self.export_json.setChecked("json" in defaults)
        self.export_docx.setChecked("docx" in defaults)
        self.export_pdf.setChecked("pdf" in defaults)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            dialog.apply_to_config(self.config)
            save_config(self.config, self.config_path)
            self._apply_default_formats()
            self.export_dir_label.setText(
                f"Export folder: {self.config.get('export', 'export_dir', fallback='outputs')}"
            )

    def _selected_export_formats(self) -> set[str]:
        selected = set()
        if self.export_txt.isChecked():
            selected.add("txt")
        if self.export_srt.isChecked():
            selected.add("srt")
        if self.export_json.isChecked():
            selected.add("json")
        if self.export_docx.isChecked():
            selected.add("docx")
        if self.export_pdf.isChecked():
            selected.add("pdf")
        return selected

    def _select_export_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not selected:
            return
        self.config.set("export", "export_dir", selected)
        save_config(self.config, self.config_path)
        self.export_dir_label.setText(f"Export folder: {selected}")

    def _export_selected(self) -> None:
        if not self.segments:
            QMessageBox.warning(self, "No transcript", "Run transcription first.")
            return

        formats = self._selected_export_formats()
        if not formats:
            QMessageBox.warning(self, "No format", "Select at least one export format.")
            return

        export_dir = Path(
            self.config.get("export", "export_dir", fallback="outputs")
        ).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(self.selected_file or "transcript").stem
        if "txt" in formats:
            write_txt(self.segments, export_dir / f"{stem}.txt")
        if "srt" in formats:
            write_srt(self.segments, export_dir / f"{stem}.srt")
        if "json" in formats:
            write_json(self.segments, export_dir / f"{stem}.json")
        if "docx" in formats:
            write_docx(self.segments, str(export_dir / f"{stem}.docx"))
        if "pdf" in formats:
            write_pdf(self.segments, str(export_dir / f"{stem}.pdf"))

        QMessageBox.information(self, "Export completed", f"Saved to: {export_dir}")
