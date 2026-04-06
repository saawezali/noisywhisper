from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.config import load_config
from utils.logger import setup_logger

SUPPORTED_FORMATS = ("txt", "docx", "pdf", "srt", "json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="noisywhisper",
        description="Offline Turkish noisy-audio transcription pipeline",
    )

    parser.add_argument("--file", required=True, help="Path to input audio file")
    parser.add_argument(
        "--format",
        nargs="+",
        choices=SUPPORTED_FORMATS,
        default=None,
        help="Output formats (one or more)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default from config.ini)",
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help="Local faster-whisper CTranslate2 model directory",
    )
    parser.add_argument(
        "--model-repo",
        default=None,
        help="Hugging Face repo id used for auto model download",
    )
    parser.add_argument(
        "--fallback-model-repo",
        default=None,
        help="Fallback Hugging Face repo id if primary download fails",
    )
    parser.add_argument(
        "--no-model-download",
        action="store_true",
        help="Disable automatic model download when local model is missing",
    )
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument(
        "--compute-type",
        default=None,
        choices=("auto", "int8", "float16", "float32"),
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Silero VAD speech threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable DeepFilterNet denoising",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="Explicit ffmpeg binary path",
    )

    parser.add_argument(
        "--config",
        default="config.ini",
        help="Config file path",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )

    return parser.parse_args()


def _parse_format_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    return [fmt for fmt in values if fmt in SUPPORTED_FORMATS]


def _progress(stage: str, pct: int, text_preview: str) -> None:
    preview = f" | {text_preview[:90]}" if text_preview else ""
    print(f"[{pct:3d}%] {stage}{preview}", flush=True)


def _write_outputs(
    formats: list[str],
    out_dir: Path,
    source_file: Path,
    segments,
) -> list[Path]:
    from output.docx_writer import write_docx
    from output.json_writer import write_json
    from output.pdf_writer import write_pdf
    from output.srt_writer import write_srt
    from output.txt_writer import write_txt

    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = source_file.stem
    generated: list[Path] = []

    for fmt in formats:
        out_path = out_dir / f"{base_name}.{fmt}"
        if fmt == "txt":
            generated.append(write_txt(segments, str(out_path)))
        elif fmt == "srt":
            generated.append(write_srt(segments, str(out_path)))
        elif fmt == "json":
            generated.append(write_json(segments, str(out_path), source_file=str(source_file)))
        elif fmt == "docx":
            generated.append(write_docx(segments, str(out_path)))
        elif fmt == "pdf":
            generated.append(write_pdf(segments, str(out_path)))

    return generated


def main() -> int:
    args = parse_args()

    # Lazy import so --help works even when dependencies are not installed yet.
    try:
        from core.worker import PipelineSettings, TranscriptionPipeline
        from utils.model_manager import ensure_model_available
    except Exception as exc:
        print(
            "Runtime dependencies are missing. Install with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Detail: {exc}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)

    log_level = args.log_level or cfg.get("logging", "level", fallback="INFO")
    log_file = cfg.get("logging", "file", fallback="noisywhisper.log")
    max_mb = cfg.getint("logging", "max_mb", fallback=5)
    backup_count = cfg.getint("logging", "backup_count", fallback=3)
    logger = setup_logger(log_file, log_level, max_mb=max_mb, backup_count=backup_count)

    source = Path(args.file).expanduser().resolve()
    if not source.exists() or not source.is_file():
        print(f"Input file not found: {source}", file=sys.stderr)
        return 2

    model_path = args.model_path or cfg.get(
        "transcription", "model_path", fallback="models/turkish-noisy-v1"
    )
    model_repo_id = args.model_repo or cfg.get(
        "transcription",
        "model_repo_id",
        fallback="Cosmobillian/turkish_whisper_for_noisy_datas",
    )
    fallback_model_repo_id = args.fallback_model_repo or cfg.get(
        "transcription",
        "fallback_model_repo_id",
        fallback="Systran/faster-whisper-large-v3",
    )
    auto_download_model_cfg = cfg.getboolean(
        "transcription", "auto_download_model", fallback=True
    )
    auto_download_model = False if args.no_model_download else auto_download_model_cfg
    compute_type = args.compute_type or cfg.get(
        "transcription", "compute_type", fallback="int8"
    )

    try:
        resolved_model_dir = ensure_model_available(
            model_dir=model_path,
            primary_repo_id=model_repo_id,
            auto_download=auto_download_model,
            logger=logger,
            fallback_repo_id=fallback_model_repo_id,
            quantization=compute_type if compute_type != "auto" else "int8",
        )
    except Exception as exc:
        logger.error("Model preparation failed: %s", exc)
        print("Error: unable to prepare ASR model.", file=sys.stderr)
        print(f"Detail: {exc}", file=sys.stderr)
        print(
            "Tip: place a converted CTranslate2 model in the local model directory.",
            file=sys.stderr,
        )
        print(
            f"Expected directory: {Path(model_path).expanduser().resolve()}",
            file=sys.stderr,
        )
        return 1

    beam_size = args.beam_size or cfg.getint("transcription", "beam_size", fallback=5)
    vad_threshold = args.vad_threshold
    if vad_threshold is None:
        vad_threshold = cfg.getfloat("transcription", "vad_threshold", fallback=0.5)

    denoise_default = cfg.getboolean("transcription", "noise_reduction", fallback=True)
    denoise_enabled = False if args.no_denoise else denoise_default

    out_dir_raw = args.out_dir or cfg.get("export", "export_dir", fallback="outputs")
    out_dir = Path(out_dir_raw).expanduser().resolve()

    formats = args.format
    if not formats:
        formats = _parse_format_list(cfg.get("export", "default_formats", fallback="txt"))
    if not formats:
        formats = ["txt"]

    # Deduplicate while preserving order.
    seen = set()
    normalized_formats = []
    for fmt in formats:
        if fmt not in seen:
            seen.add(fmt)
            normalized_formats.append(fmt)

    settings = PipelineSettings(
        model_path=str(resolved_model_dir),
        beam_size=beam_size,
        compute_type=compute_type,
        denoise_enabled=denoise_enabled,
        vad_threshold=vad_threshold,
        ffmpeg_path=args.ffmpeg,
    )

    pipeline = TranscriptionPipeline(settings, logger=logger)

    try:
        logger.info("Starting transcription for %s", source)
        segments = pipeline.run(str(source), progress_callback=_progress)

        if not segments:
            logger.warning("Transcription returned no segments")
            print("Transcription completed but no speech segments were produced.")
            return 1

        generated = _write_outputs(normalized_formats, out_dir, source, segments)

        print("\nExport complete:")
        for path in generated:
            print(f" - {path}")

        logger.info("Finished transcription. files=%s", [str(p) for p in generated])
        return 0
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
