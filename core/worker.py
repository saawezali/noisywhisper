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
    fallback_model_path: str | None = None
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

    def _looks_like_gibberish(self, text: str) -> bool:
        value = text.strip()
        if not value:
            return True

        letters = [ch.lower() for ch in value if ch.isalpha()]
        if len(letters) < 12:
            return False

        unique_chars = set(letters)
        if len(unique_chars) <= 3:
            return True

        top_char_freq = max(letters.count(ch) for ch in unique_chars) / len(letters)
        if len(letters) >= 40 and top_char_freq >= 0.55:
            return True

        tokens = self._tokenize(value)
        if len(tokens) >= 8:
            unique_tokens = set(tokens)
            if len(unique_tokens) <= 2:
                return True

            top_token_ratio = Counter(tokens).most_common(1)[0][1] / len(tokens)
            if top_token_ratio >= 0.6:
                return True

        return False

    def _is_usable_segment_text(self, text: str) -> bool:
        return bool(self._sanitize_segment_text(text))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Keep letters/digits across locales, excluding underscores.
        return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _collapse_long_runs(text: str) -> str:
        # Reduce extreme character repetition while preserving normal words.
        return re.sub(r"(.)\1{4,}", r"\1\1", text, flags=re.UNICODE)

    def _sanitize_segment_text(self, text: str) -> str:
        raw = self._collapse_long_runs(text.strip())
        if not raw:
            return ""

        cleaned = re.sub(r"(?i)\b([a-zçğıöşü]{1,3})\1{6,}\b", " ", raw)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return ""

        fragments = [frag.strip() for frag in re.split(r"[\n\r\.\!\?]+", cleaned) if frag.strip()]
        kept: list[str] = []
        for frag in fragments:
            if self._is_meaningful_text(frag) and not self._looks_like_gibberish(frag):
                kept.append(frag)

        if kept:
            return ". ".join(kept)

        if self._is_meaningful_text(cleaned) and not self._looks_like_gibberish(cleaned):
            return cleaned
        return ""

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
            fallback_model_dir=(
                Path(self.settings.fallback_model_path).expanduser().resolve()
                if self.settings.fallback_model_path
                else None
            ),
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
                sanitized = self._sanitize_segment_text(segment.text)
                if sanitized:
                    segment.text = sanitized
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
            for seg in retry_segments:
                sanitized = self._sanitize_segment_text(seg.text)
                if sanitized:
                    seg.text = sanitized
                    collected.append(seg)

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
            for seg in retry_segments:
                sanitized = self._sanitize_segment_text(seg.text)
                if sanitized:
                    seg.text = sanitized
                    collected.append(seg)

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
            for seg in retry_segments:
                sanitized = self._sanitize_segment_text(seg.text)
                if sanitized:
                    seg.text = sanitized
                    collected.append(seg)

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
            cleaned_quality_retry_segments: list[TranscriptSegment] = []
            for seg in quality_retry_segments:
                sanitized = self._sanitize_segment_text(seg.text)
                if sanitized:
                    seg.text = sanitized
                    cleaned_quality_retry_segments.append(seg)
            quality_retry_segments = cleaned_quality_retry_segments

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
