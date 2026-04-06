from __future__ import annotations

import logging

import numpy as np


def _energy_vad_fallback(
    audio: np.ndarray,
    sample_rate: int,
    min_speech_ms: int,
    min_silence_ms: int,
) -> list[tuple[int, int]]:
    frame_ms = 30
    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))

    if audio.size < frame_size:
        return [(0, int(audio.size * 1000.0 / sample_rate))]

    frame_count = audio.size // frame_size
    trimmed = audio[: frame_count * frame_size]
    frames = trimmed.reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)

    dynamic_floor = float(np.median(rms))
    threshold = max(0.01, dynamic_floor * 2.5)
    speech_mask = rms >= threshold

    windows: list[tuple[int, int]] = []
    start_idx = None
    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    max_gap_frames = max(1, int(min_silence_ms / frame_ms))
    gap_count = 0

    for idx, is_speech in enumerate(speech_mask):
        if is_speech:
            if start_idx is None:
                start_idx = idx
            gap_count = 0
            continue

        if start_idx is None:
            continue

        gap_count += 1
        if gap_count >= max_gap_frames:
            end_idx = idx - gap_count + 1
            if end_idx - start_idx >= min_speech_frames:
                start_ms = int((start_idx * frame_size) * 1000.0 / sample_rate)
                end_ms = int((end_idx * frame_size) * 1000.0 / sample_rate)
                windows.append((start_ms, end_ms))
            start_idx = None
            gap_count = 0

    if start_idx is not None:
        end_idx = frame_count
        if end_idx - start_idx >= min_speech_frames:
            start_ms = int((start_idx * frame_size) * 1000.0 / sample_rate)
            end_ms = int((end_idx * frame_size) * 1000.0 / sample_rate)
            windows.append((start_ms, end_ms))

    if not windows:
        return [(0, int(audio.size * 1000.0 / sample_rate))]

    return windows


def detect_speech_windows(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 350,
    logger: logging.Logger | None = None,
) -> list[tuple[int, int]]:
    """
    Return speech windows in milliseconds.

    Preferred path uses silero-vad; fallback uses a simple energy detector.
    """
    if audio.size == 0:
        return []

    log = logger or logging.getLogger(__name__)

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad()
        vad_audio = np.array(audio, dtype=np.float32, copy=True)
        timestamps = get_speech_timestamps(
            vad_audio,
            model,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
        )

        windows: list[tuple[int, int]] = []
        for item in timestamps:
            start = int(item["start"] * 1000.0 / sample_rate)
            end = int(item["end"] * 1000.0 / sample_rate)
            if end > start:
                windows.append((start, end))

        if windows:
            return windows

        log.warning("silero-vad returned no windows; using fallback")
    except Exception as exc:
        log.warning("silero-vad unavailable/failed; using fallback: %s", exc)

    return _energy_vad_fallback(audio, sample_rate, min_speech_ms, min_silence_ms)
