from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from ctypes import windll
from pathlib import Path

from core.worker import PipelineSettings, TranscriptionPipeline
from output.docx_writer import write_docx
from output.json_writer import write_json
from output.pdf_writer import write_pdf
from output.srt_writer import write_srt
from output.txt_writer import write_txt
from utils.config import load_config
from utils.logger import setup_logger
from utils.model_manager import ensure_model_available


def _show_startup_error(message: str) -> None:
    try:
        # 0x10 = MB_ICONERROR
        windll.user32.MessageBoxW(0, message, "NoisyWhisper Startup Error", 0x10)
    except Exception:
        print(message)


def _prepare_frozen_dll_paths() -> list[str]:
    added: list[str] = []
    if not getattr(sys, "frozen", False):
        return added

    base = Path(getattr(sys, "_MEIPASS", "")).resolve()
    if not base.exists():
        return added

    candidates = [
        base,
        base / "PyQt6",
        base / "PyQt6" / "Qt6" / "bin",
        base / "PyQt6" / "Qt6" / "plugins",
        base / "PyQt6" / "Qt6" / "plugins" / "platforms",
    ]

    for path in candidates:
        if not path.exists() or not path.is_dir():
            continue
        os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(path))
        except Exception:
            pass
        added.append(str(path))

    return added


def _write_startup_diagnostics(exc: Exception, added_dll_paths: list[str]) -> str | None:
    try:
        diag_file = Path(tempfile.gettempdir()) / "noisywhisper_startup_diag.txt"
        lines = [
            "NoisyWhisper startup diagnostics",
            "=" * 36,
            f"python={sys.version}",
            f"frozen={getattr(sys, 'frozen', False)}",
            f"_MEIPASS={getattr(sys, '_MEIPASS', '')}",
            f"cwd={Path.cwd()}",
            "",
            "added_dll_paths:",
            *[f"- {p}" for p in added_dll_paths],
            "",
            f"error={exc}",
            traceback.format_exc(),
        ]
        diag_file.write_text("\n".join(lines), encoding="utf-8")
        return str(diag_file)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NoisyWhisper CLI")
    parser.add_argument("--gui", action="store_true", help="Launch desktop GUI")
    parser.add_argument("--file", required=False, help="Input audio file path")
    parser.add_argument(
        "--format",
        nargs="+",
        default=["txt"],
        choices=["txt", "srt", "json", "docx", "pdf"],
        help="One or more export formats",
    )
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local model directory path",
    )
    parser.add_argument(
        "--model-repo",
        default=None,
        help="Primary Hugging Face repo for auto-download",
    )
    parser.add_argument(
        "--fallback-model-repo",
        default=None,
        help="Fallback Hugging Face repo for auto-download",
    )
    parser.add_argument(
        "--fallback-model-path",
        default="models/faster-whisper-small",
        help="Fallback local CTranslate2 model directory",
    )
    parser.add_argument("--no-model-download", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--ffmpeg-path", default=None)
    parser.add_argument("--config", default="config.ini")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    launch_gui = args.gui or not args.file

    if launch_gui:
        added_dll_paths = _prepare_frozen_dll_paths()
        try:
            from PyQt6.QtWidgets import QApplication

            from ui.mainwindow import MainWindow
        except Exception as exc:
            diag_path = _write_startup_diagnostics(exc, added_dll_paths)
            diag_hint = f"\n\nDiagnostics saved to:\n{diag_path}" if diag_path else ""
            hint = (
                "NoisyWhisper could not load GUI runtime libraries.\n\n"
                f"Details: {exc}\n\n"
                "If this is a shared build, either:\n"
                "1) Keep NoisyWhisper.exe with its full dist folder (_internal), or\n"
                "2) Use the one-file build artifact.\n\n"
                "Also ensure Microsoft Visual C++ Redistributable 2015-2022 (x64) is installed."
                f"{diag_hint}"
            )
            _show_startup_error(hint)
            raise SystemExit(1)

        app = QApplication([])
        window = MainWindow()
        window.show()
        app.exec()
        return

    config = load_config(args.config)

    logger = setup_logger(
        log_file=config.get("logging", "file", fallback="noisywhisper.log"),
        level=config.get("logging", "level", fallback="INFO"),
        max_mb=config.getint("logging", "max_mb", fallback=5),
        backup_count=config.getint("logging", "backup_count", fallback=3),
    )

    audio_path = Path(args.file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[NoisyWhisper] Audio: {audio_path}")

    model_path = args.model_path or config.get(
        "transcription",
        "model_path",
        fallback="turkish_whisper_for_noisy_datas",
    )
    model_repo = args.model_repo or config.get(
        "transcription",
        "model_repo_id",
        fallback="Cosmobillian/turkish_whisper_for_noisy_datas",
    )
    fallback_repo = args.fallback_model_repo or config.get(
        "transcription",
        "fallback_model_repo_id",
        fallback="Systran/faster-whisper-small",
    )
    fallback_model_path = Path(args.fallback_model_path).expanduser().resolve()
    auto_download = not args.no_model_download

    resolved_model_path = ensure_model_available(
        model_dir=model_path,
        primary_repo_id=model_repo,
        fallback_repo_id=fallback_repo,
        auto_download=auto_download,
        logger=logger,
        quantization="int8",
    )
    print(f"[NoisyWhisper] Model: {resolved_model_path}")

    resolved_fallback_model_path: Path | None = None
    if fallback_model_path.exists():
        resolved_fallback_model_path = fallback_model_path
    elif auto_download and fallback_repo:
        resolved_fallback_model_path = ensure_model_available(
            model_dir=str(fallback_model_path),
            primary_repo_id=fallback_repo,
            fallback_repo_id=None,
            auto_download=True,
            logger=logger,
            quantization="int8",
        )

    if resolved_fallback_model_path:
        print(f"[NoisyWhisper] Fallback model: {resolved_fallback_model_path}")

    beam_size = args.beam_size or config.getint("transcription", "beam_size", fallback=5)
    compute_type = args.compute_type or config.get("transcription", "compute_type", fallback="int8")
    denoise_enabled = not args.no_denoise and config.getboolean(
        "transcription", "noise_reduction", fallback=True
    )
    vad_threshold = args.vad_threshold
    if vad_threshold is None:
        vad_threshold = config.getfloat("transcription", "vad_threshold", fallback=0.5)

    settings = PipelineSettings(
        model_path=str(resolved_model_path),
        fallback_model_path=(
            str(resolved_fallback_model_path) if resolved_fallback_model_path else None
        ),
        beam_size=beam_size,
        compute_type=compute_type,
        denoise_enabled=denoise_enabled,
        vad_threshold=vad_threshold,
        ffmpeg_path=args.ffmpeg_path,
    )
    pipeline = TranscriptionPipeline(settings=settings, logger=logger)
    segments = pipeline.run(
        str(audio_path),
        progress_callback=lambda stage, pct, preview: print(
            f"[progress] {stage} {pct}% {preview[:60]}".strip()
        ),
    )

    requested = {fmt.strip().lower() for fmt in args.format if fmt.strip()}
    stem = audio_path.stem

    if "txt" in requested:
        txt_path = write_txt(segments, out_dir / f"{stem}.txt")
        print(f"[export] TXT -> {txt_path}")

    if "srt" in requested:
        srt_path = write_srt(segments, out_dir / f"{stem}.srt")
        print(f"[export] SRT -> {srt_path}")

    if "json" in requested:
        json_path = write_json(segments, out_dir / f"{stem}.json")
        print(f"[export] JSON -> {json_path}")

    if "docx" in requested:
        docx_path = write_docx(segments, str(out_dir / f"{stem}.docx"))
        print(f"[export] DOCX -> {docx_path}")

    if "pdf" in requested:
        pdf_path = write_pdf(segments, str(out_dir / f"{stem}.pdf"))
        print(f"[export] PDF -> {pdf_path}")

    print("[NoisyWhisper] Done")


if __name__ == "__main__":
    main()
