import gc
import os
import re
import threading
import time

import ctranslate2
import gradio as gr
import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
FFMPEG_EXE = os.path.join(ROOT, "ffmpeg.exe")

if os.path.exists(FFMPEG_EXE):
    os.environ["PATH"] = ROOT + os.pathsep + os.environ.get("PATH", "")

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")

KNOWN_MODELS = ["medium", "large-v3-turbo", "large-v3"]


def detect_device():
    try:
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


DEVICE = detect_device()
COMPUTE_TYPE = "int8_float16" if DEVICE == "cuda" else "int8"


def model_is_present(name):
    return os.path.isfile(os.path.join(MODEL_DIR, name, "model.bin"))


def available_models():
    return [model for model in KNOWN_MODELS if model_is_present(model)]


def missing_models():
    return [model for model in KNOWN_MODELS if not model_is_present(model)]


def to_float32(audio):
    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)
    return audio


def load_audio(path, sr=16000):
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio.astype(np.float32), sr
    except Exception:
        pass

    try:
        audio, native_sr = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = to_float32(audio)
        if native_sr != sr:
            audio = librosa.resample(audio, orig_sr=native_sr, target_sr=sr)
        return audio.astype(np.float32), sr
    except Exception:
        pass

    from scipy.io import wavfile

    native_sr, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = to_float32(audio)
    if native_sr != sr:
        audio = librosa.resample(audio, orig_sr=native_sr, target_sr=sr)
    return audio.astype(np.float32), sr


def normalize_audio(audio):
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = (audio / peak) * 0.95
    return audio


def preprocess_audio(audio, sr):
    noise_samples = min(len(audio), int(0.5 * sr))
    if noise_samples <= 0:
        return audio
    noise_clip = audio[:noise_samples]
    reduced = nr.reduce_noise(y=audio, y_noise=noise_clip, sr=sr)
    return normalize_audio(reduced)


REPEAT_RE = re.compile(r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)


def strip_repetitions(text):
    previous = None
    while previous != text:
        previous = text
        text = REPEAT_RE.sub(r"\1", text)
    return text


class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._name = None

    def load(self, model_name, progress=None):
        with self._lock:
            if self._model is not None and self._name == model_name:
                return self._model, self._name

            if not model_is_present(model_name):
                raise FileNotFoundError(
                    f"Model '{model_name}' not bundled. Contact your distributor."
                )

            self._model = None
            self._name = None
            gc.collect()

            if progress is not None:
                progress(0, desc="Loading model")

            model_path = os.path.join(MODEL_DIR, model_name)
            self._model = WhisperModel(
                model_path,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
            )
            self._name = model_name

            if progress is not None:
                progress(1, desc="Model loaded")

            return self._model, self._name

    @property
    def name(self):
        return self._name


MODEL_MANAGER = ModelManager()


def format_stats(elapsed, duration, model_name):
    rtf = elapsed / duration if duration > 0 else 0.0
    device_label = "GPU" if DEVICE == "cuda" else "CPU"
    return (
        f"Elapsed: {elapsed:.2f}s | Duration: {duration:.2f}s | "
        f"RTF: {rtf:.2f} | Device: {device_label} | Model: {model_name}"
    )


def model_availability_note():
    missing = missing_models()
    if not missing:
        return "All models are bundled."
    lines = ["Missing models (not bundled):"]
    lines.extend([f"- {model} (Contact your distributor.)" for model in missing])
    return "\n".join(lines)


def resolve_default_model():
    available = available_models()
    if "large-v3-turbo" in available:
        return "large-v3-turbo"
    if available:
        return available[0]
    return "large-v3-turbo"


def on_model_change(selected, current, progress=gr.Progress()):
    if not selected:
        return gr.Dropdown.update(), current, "Select a model to load."

    if not model_is_present(selected):
        message = f"Model '{selected}' not bundled. Contact your distributor."
        if current:
            return gr.Dropdown.update(value=current), current, message
        return gr.Dropdown.update(value=selected), current, message

    MODEL_MANAGER.load(selected, progress=progress)
    message = f"Loaded '{selected}' on {DEVICE.upper()} ({COMPUTE_TYPE})."
    return gr.Dropdown.update(value=selected), selected, message


def transcribe(file_path, use_preprocess, model_name, progress=gr.Progress()):
    if not file_path:
        return "", "Error: No audio file selected."

    start_time = time.time()

    try:
        progress(0.05, desc="Loading audio")
        audio, sr = load_audio(file_path)
        if audio.size == 0:
            return "", "Error: Audio file is empty."

        duration = float(len(audio)) / float(sr)

        if use_preprocess:
            progress(0.2, desc="Reducing noise")
            audio = preprocess_audio(audio, sr)

        progress(0.4, desc="Loading model")
        model, _ = MODEL_MANAGER.load(model_name, progress=progress)

        progress(0.6, desc="Transcribing")
        segments, _ = model.transcribe(
            audio,
            language="tr",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            repetition_penalty=1.2,
        )

        text = "".join(segment.text for segment in segments).strip()
        progress(0.9, desc="Post-processing")
        text = strip_repetitions(text)

        elapsed = time.time() - start_time
        stats = format_stats(elapsed, duration, model_name)
        return text, stats
    except Exception as exc:
        return "", f"Error: {exc}"


def build_ui():
    default_model = resolve_default_model()

    with gr.Blocks(title="Noisy Whisper") as demo:
        gr.Markdown("# Noisy Whisper")
        gr.Markdown("Offline Turkish audio transcription")

        with gr.Row():
            with gr.Column(scale=1):
                audio_file = gr.File(
                    label="Audio file",
                    file_count="single",
                    type="filepath",
                    file_types=[
                        ".mp3",
                        ".wav",
                        ".m4a",
                        ".flac",
                        ".ogg",
                        ".opus",
                        ".mp4",
                        ".webm",
                    ],
                )
                preprocess = gr.Checkbox(
                    label="Enable noise reduction",
                    value=True,
                )
                model_select = gr.Dropdown(
                    label="Model",
                    choices=KNOWN_MODELS,
                    value=default_model,
                )
                model_note = gr.Markdown(value=model_availability_note())
                model_status = gr.Markdown()
                transcribe_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Transcription",
                    lines=16,
                    interactive=False,
                    show_copy_button=True,
                )
                stats = gr.Markdown()

        current_model = gr.State(value=default_model)

        model_select.change(
            on_model_change,
            inputs=[model_select, current_model],
            outputs=[model_select, current_model, model_status],
            queue=True,
        )

        transcribe_btn.click(
            transcribe,
            inputs=[audio_file, preprocess, model_select],
            outputs=[output, stats],
        )

    return demo


def main():
    demo = build_ui()
    demo.queue(concurrency_count=1, max_size=8)
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
