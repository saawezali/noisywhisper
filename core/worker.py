from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.audio import decode_to_pcm, slice_audio_ms
from core.denoise import denoise_audio
from core.transcribe import TranscriptSegment, TurkishWhisperTranscriber
from core.vad import detect_speech_windows

ProgressCallback = Callable[[str, int, str], None]


@dataclass
class PipelineSettings:
    model_path: str = "models/turkish-noisy-v1"
    beam_size: int = 5
    compute_type: str = "int8"
    denoise_enabled: bool = True
    vad_threshold: float = 0.5
    ffmpeg_path: str | None = None


class TranscriptionPipeline:
    def __init__(
        self,
        settings: PipelineSettings,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

    def _emit(
        self,
        callback: ProgressCallback | None,
        stage: str,
        pct: int,
        text_preview: str = "",
    ) -> None:
        if callback:
            callback(stage, pct, text_preview)

    def run(
        self,
        file_path: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[TranscriptSegment]:
        audio_path = Path(file_path).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._emit(progress_callback, "Decoding audio", 5)
        audio, sample_rate = decode_to_pcm(
            str(audio_path),
            ffmpeg_path=self.settings.ffmpeg_path,
        )

        self._emit(progress_callback, "Denoising audio", 20)
        audio = denoise_audio(
            audio,
            sample_rate,
            enabled=self.settings.denoise_enabled,
            logger=self.logger,
        )

        self._emit(progress_callback, "Running voice activity detection", 35)
        windows = detect_speech_windows(
            audio,
            sample_rate,
            threshold=self.settings.vad_threshold,
            logger=self.logger,
        )

        if not windows:
            self.logger.warning("No speech windows detected; transcribing full audio")
            total_ms = int(audio.shape[0] * 1000.0 / sample_rate)
            windows = [(0, total_ms)]

        transcriber = TurkishWhisperTranscriber(
            model_path=self.settings.model_path,
            compute_type=self.settings.compute_type,
            beam_size=self.settings.beam_size,
            language="tr",
            logger=self.logger,
        )

        self._emit(progress_callback, "Transcribing", 40)
        collected: list[TranscriptSegment] = []
        total_windows = max(1, len(windows))

        for idx, (start_ms, end_ms) in enumerate(windows, start=1):
            chunk = slice_audio_ms(audio, sample_rate, start_ms, end_ms)
            if chunk.size == 0:
                continue

            segments = transcriber.transcribe_chunk(
                chunk,
                sample_rate=sample_rate,
                offset_ms=start_ms,
            )

            for segment in segments:
                if segment.text:
                    collected.append(segment)
                    self._emit(
                        progress_callback,
                        "Transcribing",
                        min(95, 40 + int(idx / total_windows * 55)),
                        segment.text,
                    )

        collected.sort(key=lambda x: x.start_ms)
        self._emit(progress_callback, "Completed", 100)
        return collected
