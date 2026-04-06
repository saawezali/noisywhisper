from __future__ import annotations

import logging
import re
from collections import Counter
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

    @staticmethod
    def _is_meaningful_text(text: str) -> bool:
        value = text.strip()
        if not value:
            return False
        return any(ch.isalnum() for ch in value)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Keep letters/digits across locales, excluding underscores.
        return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)

    def _quality_score(self, segments: list[TranscriptSegment]) -> float:
        text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0

        unique_ratio = len(set(tokens)) / len(tokens)
        top_ratio = Counter(tokens).most_common(1)[0][1] / len(tokens)
        alnum_ratio = sum(ch.isalnum() for ch in text) / max(1, len(text))

        score = (0.50 * unique_ratio) + (0.35 * (1.0 - top_ratio)) + (0.15 * alnum_ratio)
        return float(max(0.0, min(1.0, score)))

    def _is_low_quality(self, segments: list[TranscriptSegment]) -> bool:
        text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
        tokens = self._tokenize(text)
        if not tokens:
            return True

        unique_ratio = len(set(tokens)) / len(tokens)
        top_ratio = Counter(tokens).most_common(1)[0][1] / len(tokens)

        if len(tokens) >= 8 and (top_ratio > 0.45 or unique_ratio < 0.35):
            return True
        if len(tokens) >= 12 and len(set(tokens)) <= 4:
            return True

        return self._quality_score(segments) < 0.22

    def run(
        self,
        file_path: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[TranscriptSegment]:
        audio_path = Path(file_path).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._emit(progress_callback, "Decoding audio", 5)
        source_audio, sample_rate = decode_to_pcm(
            str(audio_path),
            ffmpeg_path=self.settings.ffmpeg_path,
        )

        audio = source_audio
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
                if self._is_meaningful_text(segment.text):
                    collected.append(segment)
                    self._emit(
                        progress_callback,
                        "Transcribing",
                        min(95, 40 + int(idx / total_windows * 55)),
                        segment.text,
                    )

        if not collected:
            self.logger.warning(
                "No segments from VAD-windowed transcription; retrying full-audio pass"
            )
            self._emit(progress_callback, "Retrying transcription (full audio)", 96)

            retry_segments = transcriber.transcribe_chunk(
                audio,
                sample_rate=sample_rate,
                offset_ms=0,
                use_internal_vad=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.95,
            )
            collected.extend(
                [seg for seg in retry_segments if self._is_meaningful_text(seg.text)]
            )

        if not collected and self.settings.denoise_enabled:
            self.logger.warning(
                "No segments after denoised retry; retrying original audio"
            )
            self._emit(progress_callback, "Retrying transcription (raw audio)", 97)

            retry_segments = transcriber.transcribe_chunk(
                source_audio,
                sample_rate=sample_rate,
                offset_ms=0,
                use_internal_vad=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.95,
            )
            collected.extend(
                [seg for seg in retry_segments if self._is_meaningful_text(seg.text)]
            )

        if not collected:
            self.logger.warning(
                "No meaningful segments after retries; attempting permissive final pass"
            )
            self._emit(progress_callback, "Retrying transcription (permissive)", 98)

            retry_audio = source_audio if self.settings.denoise_enabled else audio
            retry_segments = transcriber.transcribe_chunk(
                retry_audio,
                sample_rate=sample_rate,
                offset_ms=0,
                use_internal_vad=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.99,
            )
            collected.extend(
                [seg for seg in retry_segments if self._is_meaningful_text(seg.text)]
            )

        if collected and self.settings.denoise_enabled and self._is_low_quality(collected):
            current_score = self._quality_score(collected)
            self.logger.warning(
                "Low-quality transcript detected (score=%.3f); retrying raw audio permissive pass",
                current_score,
            )
            self._emit(progress_callback, "Retrying transcription (quality recovery)", 99)

            quality_retry_segments = transcriber.transcribe_chunk(
                source_audio,
                sample_rate=sample_rate,
                offset_ms=0,
                use_internal_vad=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.99,
            )
            quality_retry_segments = [
                seg for seg in quality_retry_segments if self._is_meaningful_text(seg.text)
            ]

            retry_score = self._quality_score(quality_retry_segments)
            if quality_retry_segments and retry_score > current_score:
                self.logger.info(
                    "Quality recovery pass accepted (old=%.3f new=%.3f)",
                    current_score,
                    retry_score,
                )
                collected = quality_retry_segments
            else:
                self.logger.info(
                    "Quality recovery pass discarded (old=%.3f new=%.3f)",
                    current_score,
                    retry_score,
                )

        collected.sort(key=lambda x: x.start_ms)
        self._emit(progress_callback, "Completed", 100)
        return collected
