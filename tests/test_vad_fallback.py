import numpy as np

from core.vad import detect_speech_windows


def test_vad_fallback_detects_speech_window() -> None:
    sample_rate = 16000
    silence = np.zeros(sample_rate, dtype=np.float32)
    tone = 0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate, endpoint=False)).astype(np.float32)
    audio = np.concatenate([silence, tone, silence])

    windows = detect_speech_windows(
        audio,
        sample_rate=sample_rate,
        threshold=0.5,
        min_speech_ms=200,
        min_silence_ms=200,
    )

    assert windows, "Expected at least one speech window"
    first_start, first_end = windows[0]
    assert first_end > first_start
