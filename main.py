from __future__ import annotations

import argparse
from pathlib import Path

from core.transcribe import TurkishWhisperTranscriber
from output.json_writer import write_json
from output.srt_writer import write_srt
from output.txt_writer import write_txt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NoisyWhisper CLI MVP")
    parser.add_argument("--file", required=True, help="Input audio file path")
    parser.add_argument(
        "--model-dir",
        default="turkish_whisper_for_noisy_datas",
        help="Local Whisper model directory",
    )
    parser.add_argument(
        "--formats",
        default="txt,srt,json",
        help="Comma-separated export formats (txt,srt,json)",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Output directory",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument(
        "--fallback-model-dir",
        default="models/faster-whisper-small",
        help="Fallback CTranslate2 model dir used when primary model output is invalid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.file).resolve()
    model_dir = Path(args.model_dir).resolve()
    fallback_model_dir = Path(args.fallback_model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[NoisyWhisper] Audio: {audio_path}")
    print(f"[NoisyWhisper] Model: {model_dir}")
    print(f"[NoisyWhisper] Fallback model: {fallback_model_dir}")

    transcriber = TurkishWhisperTranscriber(
        model_dir=model_dir,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        fallback_model_dir=fallback_model_dir,
    )

    segments = transcriber.transcribe_file(audio_path, progress=lambda msg: print(f"[progress] {msg}"))

    requested = {fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()}
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

    print("[NoisyWhisper] Done")


if __name__ == "__main__":
    main()
