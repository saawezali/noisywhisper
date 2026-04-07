from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from faster_whisper import WhisperModel

ProgressCallback = Callable[[str], None]


@dataclass
class TranscriptSegment:
    id: int
    start_ms: int
    end_ms: int
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class TurkishWhisperTranscriber:
    def __init__(
        self,
        model_path: str | Path | None = None,
        model_dir: Path | None = None,
        compute_type: str = "int8",
        device: str = "cpu",
        beam_size: int = 5,
        language: str = "tr",
        logger: logging.Logger | None = None,
        fallback_model_dir: Path | None = None,
    ) -> None:
        resolved_model_path = model_dir or model_path
        if resolved_model_path is None:
            raise ValueError("model_path or model_dir must be provided")

        self.model_dir = Path(resolved_model_path).expanduser().resolve()
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self.language = language
        self.logger = logger or logging.getLogger(__name__)
        self.fallback_model_dir = fallback_model_dir

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.runtime_model_dir = self._ensure_ct2_model_dir(self.model_dir)

        self._model = WhisperModel(
            str(self.runtime_model_dir),
            device=self.device,
            compute_type=self.compute_type,
            local_files_only=True,
        )

        self._fallback_model: WhisperModel | None = None
        if self.fallback_model_dir and self.fallback_model_dir.exists():
            self._fallback_model = WhisperModel(
                str(self.fallback_model_dir),
                device=self.device,
                compute_type="int8",
                local_files_only=True,
            )

    def _ensure_ct2_model_dir(self, source_dir: Path) -> Path:
        source_model = source_dir / "model.bin"
        if source_model.exists():
            return source_dir

        converted_dir = source_dir.parent / f"{source_dir.name}_ct2"
        converted_model = converted_dir / "model.bin"
        if converted_model.exists():
            return converted_dir

        converter = shutil.which("ct2-transformers-converter")
        if not converter:
            raise RuntimeError(
                "ct2-transformers-converter was not found. Install ctranslate2/faster-whisper first."
            )

        cmd = [
            converter,
            "--model",
            str(source_dir),
            "--output_dir",
            str(converted_dir),
            "--quantization",
            "float32",
            "--copy_files",
            "tokenizer.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        if converted_dir.exists():
            cmd.append("--force")
        subprocess.run(cmd, check=True)
        if not converted_model.exists():
            raise RuntimeError(
                f"Model conversion finished but model.bin was not created in {converted_dir}"
            )
        return converted_dir

    def transcribe_file(
        self,
        audio_path: Path,
        progress: ProgressCallback | None = None,
    ) -> list[TranscriptSegment]:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if progress:
            progress("Transcription started")

        segments = self._run_whisper(self._model, audio_path)
        local_text = " ".join(seg.text for seg in segments if seg.text).strip()

        if progress:
            progress(f"Local model produced {len(segments)} segments")

        if self._is_degenerate_text(local_text) and self._fallback_model is not None:
            if progress:
                progress("Local model output looks invalid, trying baseline fallback model")
            fallback_segments = self._run_whisper(self._fallback_model, audio_path)
            fallback_text = " ".join(seg.text for seg in fallback_segments if seg.text).strip()
            if not self._is_degenerate_text(fallback_text):
                segments = fallback_segments
                if progress:
                    progress("Baseline fallback model selected")

        if not segments or self._is_degenerate_text(
            " ".join(seg.text for seg in segments if seg.text).strip()
        ):
            if progress:
                progress("No segments from faster-whisper, trying transformers fallback")
            fallback = self._transformers_fallback(audio_path)
            if fallback:
                segments.append(
                    TranscriptSegment(
                        id=0,
                        start_ms=0,
                        end_ms=0,
                        text=fallback,
                    )
                )

        if progress:
            progress(f"Transcription finished with {len(segments)} segments")

        return segments

    def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        offset_ms: int = 0,
        use_internal_vad: bool = True,
        condition_on_previous_text: bool = True,
        no_speech_threshold: float = 0.95,
        log_prob_threshold: float = -5.0,
    ) -> list[TranscriptSegment]:
        if audio_chunk.size == 0:
            return []

        primary_segments = self._run_whisper(
            self._model,
            np.asarray(audio_chunk, dtype=np.float32),
            sample_rate=sample_rate,
            offset_ms=offset_ms,
            use_internal_vad=use_internal_vad,
            condition_on_previous_text=condition_on_previous_text,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
        )

        primary_text = " ".join(seg.text for seg in primary_segments if seg.text).strip()
        if self._is_degenerate_text(primary_text) and self._fallback_model is not None:
            fallback_segments = self._run_whisper(
                self._fallback_model,
                np.asarray(audio_chunk, dtype=np.float32),
                sample_rate=sample_rate,
                offset_ms=offset_ms,
                use_internal_vad=use_internal_vad,
                condition_on_previous_text=condition_on_previous_text,
                no_speech_threshold=no_speech_threshold,
                log_prob_threshold=log_prob_threshold,
            )
            fallback_text = " ".join(seg.text for seg in fallback_segments if seg.text).strip()
            if not self._is_degenerate_text(fallback_text):
                return fallback_segments

        return primary_segments

    def _run_whisper(
        self,
        model: WhisperModel,
        audio_input: Path | np.ndarray,
        sample_rate: int | None = None,
        offset_ms: int = 0,
        use_internal_vad: bool = True,
        condition_on_previous_text: bool = True,
        no_speech_threshold: float = 0.95,
        log_prob_threshold: float = -5.0,
    ) -> list[TranscriptSegment]:
        transcribe_target: str | np.ndarray
        if isinstance(audio_input, Path):
            transcribe_target = str(audio_input)
        else:
            transcribe_target = np.asarray(audio_input, dtype=np.float32)

        raw_segments, _info = model.transcribe(
            transcribe_target,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=use_internal_vad,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True,
            condition_on_previous_text=condition_on_previous_text,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
        )

        segments: list[TranscriptSegment] = []
        for seg in raw_segments:
            text = seg.text.strip()
            if not text:
                continue

            start_ms = int(round(float(seg.start) * 1000.0)) + offset_ms
            end_ms = int(round(float(seg.end) * 1000.0)) + offset_ms
            segments.append(
                TranscriptSegment(
                    id=int(getattr(seg, "id", 0)),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    avg_logprob=getattr(seg, "avg_logprob", None),
                    no_speech_prob=getattr(seg, "no_speech_prob", None),
                )
            )
        return segments

    def _is_degenerate_text(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 5:
            return True

        alnum_count = sum(ch.isalnum() for ch in stripped)
        if alnum_count < 3:
            return True

        unique_chars = {ch for ch in stripped if not ch.isspace()}
        if len(unique_chars) <= 2:
            return True

        return False

    def _transformers_fallback(self, audio_path: Path) -> str:
        try:
            from transformers import pipeline
        except Exception:
            return ""

        asr = pipeline(
            "automatic-speech-recognition",
            model=str(self.model_dir),
            device="cpu",
        )
        result = asr(str(audio_path))
        text = ""
        if isinstance(result, dict):
            text = str(result.get("text") or "")
        elif isinstance(result, str):
            text = result
        return text.strip()
