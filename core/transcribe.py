from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, List

from faster_whisper import WhisperModel

ProgressCallback = Callable[[str], None]


class TurkishWhisperTranscriber:
    def __init__(
        self,
        model_dir: Path,
        compute_type: str = "int8",
        device: str = "cpu",
        beam_size: int = 5,
        fallback_model_dir: Path | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
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
    ) -> List[dict]:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if progress:
            progress("Transcription started")

        segments = self._run_whisper(self._model, audio_path)
        local_text = " ".join(seg["text"] for seg in segments if seg["text"]).strip()

        if progress:
            progress(f"Local model produced {len(segments)} segments")

        if self._is_degenerate_text(local_text) and self._fallback_model is not None:
            if progress:
                progress("Local model output looks invalid, trying baseline fallback model")
            fallback_segments = self._run_whisper(self._fallback_model, audio_path)
            fallback_text = " ".join(seg["text"] for seg in fallback_segments if seg["text"]).strip()
            if not self._is_degenerate_text(fallback_text):
                segments = fallback_segments
                if progress:
                    progress("Baseline fallback model selected")

        if not segments or self._is_degenerate_text(
            " ".join(seg["text"] for seg in segments if seg["text"]).strip()
        ):
            if progress:
                progress("No segments from faster-whisper, trying transformers fallback")
            fallback = self._transformers_fallback(audio_path)
            if fallback:
                segments.append(
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 0.0,
                        "text": fallback,
                        "avg_logprob": None,
                        "no_speech_prob": None,
                    }
                )

        if progress:
            progress(f"Transcription finished with {len(segments)} segments")

        return segments

    def _run_whisper(self, model: WhisperModel, audio_path: Path) -> List[dict]:
        raw_segments, _info = model.transcribe(
            str(audio_path),
            language="tr",
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            no_speech_threshold=1.0,
            log_prob_threshold=-5.0,
        )

        segments: List[dict] = []
        for seg in raw_segments:
            text = seg.text.strip()
            if not text:
                continue
            segments.append(
                {
                    "id": seg.id,
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                    "avg_logprob": getattr(seg, "avg_logprob", None),
                    "no_speech_prob": getattr(seg, "no_speech_prob", None),
                }
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
