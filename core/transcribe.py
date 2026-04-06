from __future__ import annotations

import json
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
        self._backend: str | None = None

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _resolve_hf_config(self) -> tuple[Path | None, dict | None]:
        for name in ("config.json", "config.hf.original.json"):
            candidate = Path(self.model_path).joinpath(name)
            parsed = self._read_json(candidate)
            if parsed is None:
                continue
            if parsed.get("model_type") == "whisper":
                return candidate, parsed
        return None, None

    def _has_transformers_weights(self) -> bool:
        path = Path(self.model_path)
        weight_candidates = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        return any(path.joinpath(name).exists() for name in weight_candidates)

    def _detect_backend(self) -> str:
        path = Path(self.model_path)
        has_ct2 = path.joinpath("model.bin").exists()
        hf_config_path, hf_config = self._resolve_hf_config()
        has_hf_weights = self._has_transformers_weights()

        # Prefer CTranslate2 when available to avoid duplicating runtime model copies.
        if has_ct2:
            return "faster-whisper"

        if has_hf_weights and hf_config is not None:
            mel_bins = hf_config.get("num_mel_bins")
            # Some Whisper variants use 128 mel bins and are incompatible with
            # faster-whisper's 80-bin frontend. Use transformers backend directly.
            if isinstance(mel_bins, int) and mel_bins != 80:
                self.logger.info(
                    "Selecting transformers backend (num_mel_bins=%s) for model at %s",
                    mel_bins,
                    self.model_path,
                )
                return "transformers"

        if has_hf_weights and hf_config_path is not None:
            return "transformers"

        raise FileNotFoundError(
            "No supported model backend found. Expected either CTranslate2 "
            "(model.bin) or Hugging Face Whisper weights+config in: "
            f"{self.model_path}"
        )

    @staticmethod
    def ensure_model_exists(model_path: str) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                "Model directory not found. Expected local CTranslate2 model at: "
                f"{path}"
            )

    def _prepare_transformers_runtime(self) -> tuple[Path, Path | None]:
        model_root = Path(self.model_path)
        config_path, config_payload = self._resolve_hf_config()
        if config_path is None or config_payload is None:
            raise FileNotFoundError(
                "Hugging Face config not found for transformers backend at "
                f"{model_root}"
            )

        config_override = config_path if config_path.name != "config.json" else None
        return model_root, config_override

    def _load_faster_whisper(self) -> None:
        required_files = ("model.bin", "config.json", "tokenizer.json")
        missing = [
            name
            for name in required_files
            if not Path(self.model_path).joinpath(name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "CTranslate2 model directory is incomplete. Missing files: "
                f"{missing} at {self.model_path}"
            )

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

    def _load_transformers(self) -> None:
        runtime_dir, config_override = self._prepare_transformers_runtime()

        try:
            import torch
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(
                "transformers backend dependencies are missing. "
                "Install dependencies from requirements.txt"
            ) from exc

        device = 0 if torch.cuda.is_available() else -1
        lang_name = "turkish" if self.language == "tr" else self.language
        pipeline_kwargs = {
            "model": str(runtime_dir),
            "tokenizer": str(runtime_dir),
            "feature_extractor": str(runtime_dir),
            "device": device,
        }
        if config_override is not None:
            pipeline_kwargs["config"] = str(config_override)

        self._model = pipeline(
            "automatic-speech-recognition",
            **pipeline_kwargs,
        )
        self.logger.info(
            "Loaded transformers Whisper backend from %s (device=%s, language=%s)",
            runtime_dir,
            device,
            lang_name,
        )

    def _load_model(self) -> None:
        if self._model is not None:
            return

        self.ensure_model_exists(self.model_path)

        self._backend = self._detect_backend()
        if self._backend == "faster-whisper":
            self._load_faster_whisper()
            return

        if self._backend == "transformers":
            self._load_transformers()
            return

        raise RuntimeError(f"Unsupported backend: {self._backend}")

    def _transcribe_with_transformers(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        offset_ms: int,
    ) -> list[TranscriptSegment]:
        assert self._model is not None

        lang_name = "turkish" if self.language == "tr" else self.language
        payload = {
            "array": np.asarray(audio_chunk, dtype=np.float32),
            "sampling_rate": sample_rate,
        }
        response = self._model(
            payload,
            return_timestamps=False,
            generate_kwargs={
                "language": lang_name,
                "task": "transcribe",
                "max_new_tokens": 256,
                "num_beams": max(1, int(self.beam_size)),
            },
        )

        text = ""
        if isinstance(response, dict):
            text = str(response.get("text", "")).strip()

        segments: list[TranscriptSegment] = []
        if text:
            total_ms = int((audio_chunk.shape[0] * 1000.0) / sample_rate)
            segments.append(
                TranscriptSegment(
                    start_ms=offset_ms,
                    end_ms=offset_ms + total_ms,
                    text=text,
                    confidence=None,
                    words=[],
                )
            )

        return segments

    def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        offset_ms: int = 0,
        use_internal_vad: bool = False,
        condition_on_previous_text: bool = True,
        no_speech_threshold: float | None = None,
    ) -> list[TranscriptSegment]:
        if audio_chunk.size == 0:
            return []

        self._load_model()

        assert self._model is not None

        if self._backend == "transformers":
            return self._transcribe_with_transformers(audio_chunk, sample_rate, offset_ms)

        transcribe_kwargs = {
            "language": self.language,
            "beam_size": self.beam_size,
            "word_timestamps": True,
            "condition_on_previous_text": condition_on_previous_text,
            "vad_filter": use_internal_vad,
        }
        if no_speech_threshold is not None:
            transcribe_kwargs["no_speech_threshold"] = no_speech_threshold

        segments, _info = self._model.transcribe(audio_chunk, **transcribe_kwargs)

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
