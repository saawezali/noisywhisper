from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


DEFAULT_SAMPLE_RATE = 16000


def resolve_ffmpeg_path(explicit_path: str | None = None) -> str:
    """Resolve ffmpeg path from explicit input, bundled binary, or system PATH."""
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"ffmpeg not found at: {candidate}")

    project_root = Path(__file__).resolve().parents[1]
    bundled_candidates = []
    if project_root.joinpath("ffmpeg").exists():
        bundled_candidates.append(project_root.joinpath("ffmpeg"))
    if project_root.joinpath("ffmpeg.exe").exists():
        bundled_candidates.append(project_root.joinpath("ffmpeg.exe"))

    for binary in bundled_candidates:
        if binary.is_file():
            return str(binary)

    ffmpeg_from_path = shutil.which("ffmpeg")
    if ffmpeg_from_path:
        return ffmpeg_from_path

    raise FileNotFoundError(
        "ffmpeg binary not found. Put ffmpeg in the project root or install it in PATH."
    )


def decode_to_pcm(
    file_path: str,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    ffmpeg_path: str | None = None,
) -> tuple[np.ndarray, int]:
    """
    Decode any ffmpeg-supported input into mono float32 PCM.

    Returns:
        (audio_float32, sample_rate)
    """
    source = Path(file_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {source}")

    ffmpeg_bin = resolve_ffmpeg_path(ffmpeg_path)

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]

    process = subprocess.run(cmd, capture_output=True, check=False)

    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed: {stderr or 'unknown error'}")

    if not process.stdout:
        raise RuntimeError("ffmpeg decode produced empty output")

    audio = np.frombuffer(process.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("Decoded audio is empty")

    return audio, target_sample_rate


def slice_audio_ms(
    audio: np.ndarray,
    sample_rate: int,
    start_ms: int,
    end_ms: int,
) -> np.ndarray:
    """Slice float32 PCM audio by millisecond boundaries."""
    start_idx = max(0, int((start_ms / 1000.0) * sample_rate))
    end_idx = min(audio.shape[0], int((end_ms / 1000.0) * sample_rate))
    if end_idx <= start_idx:
        return np.empty(0, dtype=np.float32)
    return np.ascontiguousarray(audio[start_idx:end_idx], dtype=np.float32)
