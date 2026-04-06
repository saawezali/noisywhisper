from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TranscriptWord:
    word: str
    start_ms: int
    end_ms: int
    probability: float | None = None


@dataclass
class TranscriptSegment:
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None
    words: list[TranscriptWord] = field(default_factory=list)


class TurkishWhisperTranscriber:
    def __init__(
        self,
        model_path: str,
        compute_type: str = "int8",
        beam_size: int = 5,
        language: str = "tr",
        logger: logging.Logger | None = None,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.logger = logger or logging.getLogger(__name__)
        self._model = None

    @staticmethod
    def ensure_model_exists(model_path: str) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                "Model directory not found. Expected local CTranslate2 model at: "
                f"{path}"
            )

        required_files = ("model.bin", "config.json", "tokenizer.json")
        missing = [name for name in required_files if not path.joinpath(name).exists()]
        if missing:
            raise FileNotFoundError(
                "Model directory is incomplete. Missing files: "
                f"{missing} at {path}"
            )

    def _load_model(self) -> None:
        if self._model is not None:
            return

        self.ensure_model_exists(self.model_path)

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install dependencies first."
            ) from exc

        compute_candidates = [self.compute_type]
        if self.compute_type == "auto":
            compute_candidates = ["float16", "int8"]

        last_error: Exception | None = None
        for candidate in compute_candidates:
            try:
                self._model = WhisperModel(
                    self.model_path,
                    device="auto",
                    compute_type=candidate,
                )
                self.logger.info(
                    "Loaded faster-whisper model with compute_type=%s", candidate
                )
                return
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Failed to load model with compute_type=%s: %s", candidate, exc
                )

        if last_error:
            raise RuntimeError(f"Unable to load ASR model: {last_error}") from last_error
        raise RuntimeError("Unable to load ASR model")

    def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        offset_ms: int = 0,
    ) -> list[TranscriptSegment]:
        if audio_chunk.size == 0:
            return []

        self._load_model()

        assert self._model is not None

        segments, _info = self._model.transcribe(
            audio_chunk,
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=True,
            vad_filter=False,
        )

        result: list[TranscriptSegment] = []
        for segment in segments:
            seg_start = int(offset_ms + float(segment.start) * 1000.0)
            seg_end = int(offset_ms + float(segment.end) * 1000.0)

            confidence = None
            avg_logprob = getattr(segment, "avg_logprob", None)
            if avg_logprob is not None:
                confidence = float(np.clip(np.exp(float(avg_logprob)), 0.0, 1.0))

            words: list[TranscriptWord] = []
            for word in (getattr(segment, "words", None) or []):
                w_start = int(offset_ms + float(word.start) * 1000.0)
                w_end = int(offset_ms + float(word.end) * 1000.0)
                words.append(
                    TranscriptWord(
                        word=str(word.word).strip(),
                        start_ms=w_start,
                        end_ms=w_end,
                        probability=getattr(word, "probability", None),
                    )
                )

            result.append(
                TranscriptSegment(
                    start_ms=seg_start,
                    end_ms=seg_end,
                    text=str(segment.text).strip(),
                    confidence=confidence,
                    words=words,
                )
            )

        return result
