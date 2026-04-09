import base64
import gc
import io
import json
import math
import os
import re
import struct
import threading
import time
import tempfile
import textwrap
import wave

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")

import ctranslate2
import gradio as gr
import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

try:
    from gradio_client import utils as gr_client_utils

    _orig_schema_to_type = gr_client_utils._json_schema_to_python_type

    def _patched_schema_to_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_schema_to_type(schema, defs)

    gr_client_utils._json_schema_to_python_type = _patched_schema_to_type
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
FFMPEG_EXE = os.path.join(ROOT, "ffmpeg.exe")
FONTS_DIR = os.path.join(ROOT, "fonts")
PDF_FONT_NAME = "DejaVuSans"
PDF_FONT_FILE = os.path.join(FONTS_DIR, "DejaVuSans.ttf")

if os.path.exists(FFMPEG_EXE):
    os.environ["PATH"] = ROOT + os.pathsep + os.environ.get("PATH", "")

KNOWN_MODELS = ["medium", "large-v3-turbo", "large-v3"]
VAD_PARAMS = {"threshold": 0.35}


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


def make_beep_data_uri():
    sample_rate = 16000
    duration = 0.2
    frequency = 880.0
    volume = 0.2
    sample_count = int(sample_rate * duration)
    frames = bytearray()

    for i in range(sample_count):
        value = volume * math.sin(2 * math.pi * frequency * (i / sample_rate))
        frames.extend(struct.pack("<h", int(value * 32767)))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{data}"


BEEP_DATA_URI = make_beep_data_uri()


def completion_sound_html():
    stamp = int(time.time() * 1000)
    return (
        f"<audio autoplay='true' src='{BEEP_DATA_URI}'></audio>"
        f"<span data-ts='{stamp}'></span>"
    )


def format_srt_timestamp(seconds):
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments):
    lines = []
    index = 1
    for seg in segments or []:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end < start:
            end = start
        text = (seg.get("text") or "").strip()
        text = strip_repetitions(text)
        if not text:
            continue
        lines.append(str(index))
        lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
        index += 1
    return "\n".join(lines).strip()


def write_pdf(path, text):
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    width, height = LETTER
    margin = 0.75 * inch
    max_chars = 100
    c = canvas.Canvas(path, pagesize=LETTER)
    text_object = c.beginText(margin, height - margin)

    font_name = "Helvetica"
    if os.path.isfile(PDF_FONT_FILE):
        try:
            pdfmetrics.registerFont(TTFont(PDF_FONT_NAME, PDF_FONT_FILE))
            font_name = PDF_FONT_NAME
        except Exception:
            font_name = "Helvetica"

    text_object.setFont(font_name, 11)

    paragraphs = text.splitlines() or [""]
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(paragraph, width=max_chars) or [""]
        for line in wrapped:
            if text_object.getY() <= margin:
                c.drawText(text_object)
                c.showPage()
                text_object = c.beginText(margin, height - margin)
            text_object.textLine(line)
        if text_object.getY() <= margin:
            c.drawText(text_object)
            c.showPage()
            text_object = c.beginText(margin, height - margin)
        text_object.textLine("")

    c.drawText(text_object)
    c.save()


def export_transcript(export_format, text, segments, audio_path):
    if not text:
        return None, "No transcription to export."

    export_format = (export_format or "txt").lower()
    base_name = os.path.splitext(os.path.basename(audio_path or "transcription"))[0]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{export_format}",
        prefix=f"{base_name}_",
    ) as tmp:
        path = tmp.name

    if export_format == "txt":
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    elif export_format == "srt":
        srt_text = segments_to_srt(segments)
        if not srt_text:
            return None, "No segments available for SRT export."
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(srt_text)
    elif export_format == "docx":
        from docx import Document

        doc = Document()
        doc.add_paragraph(text)
        doc.save(path)
    elif export_format == "pdf":
        write_pdf(path, text)
    else:
        return None, "Unsupported export format."

    return path, f"Exported {export_format.upper()}."


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
    return available[0] if available else None


def build_client_js(missing):
    if not missing:
        return ""
    tooltip = "Model not bundled. Contact your distributor."
    missing_json = json.dumps(missing)
    tooltip_json = json.dumps(tooltip)
    return (
        "() => {\n"
        f"  const missing = {missing_json};\n"
        f"  const tooltip = {tooltip_json};\n"
        "  const apply = () => {\n"
        "    const root = document.getElementById('model-select');\n"
        "    if (!root) return;\n"
        "    const select = root.querySelector('select');\n"
        "    if (!select) return;\n"
        "    Array.from(select.options).forEach((opt) => {\n"
        "      if (missing.includes(opt.value)) {\n"
        "        opt.disabled = true;\n"
        "        opt.title = tooltip;\n"
        "      }\n"
        "    });\n"
        "  };\n"
        "  apply();\n"
        "  setTimeout(apply, 300);\n"
        "};"
    )


def on_model_change(selected, current, progress=gr.Progress()):
    if not selected:
        return gr.update(), current, "Select a model to load."

    if not model_is_present(selected):
        message = f"Model '{selected}' not bundled. Contact your distributor."
        if current:
            return gr.update(value=current), current, message
        return gr.update(value=selected), current, message

    MODEL_MANAGER.load(selected, progress=progress)
    message = f"Loaded '{selected}' on {DEVICE.upper()} ({COMPUTE_TYPE})."
    return gr.update(value=selected), selected, message


def transcribe(
    file_path,
    use_preprocess,
    disable_vad,
    model_name,
    progress=gr.Progress(),
):
    if not file_path:
        return "", "Error: No audio file selected.", [], "", ""
    if not model_name:
        return "", "Error: No model selected.", [], file_path or "", ""

    start_time = time.time()

    try:
        progress(0.05, desc="Loading audio")
        audio, sr = load_audio(file_path)
        if audio.size == 0:
            return "", "Error: Audio file is empty.", [], file_path, ""

        duration = float(len(audio)) / float(sr)

        if use_preprocess:
            progress(0.2, desc="Reducing noise")
            audio = preprocess_audio(audio, sr)

        progress(0.4, desc="Loading model")
        model, _ = MODEL_MANAGER.load(model_name, progress=progress)

        progress(0.6, desc="Transcribing")
        if disable_vad:
            segments, _ = model.transcribe(
                audio,
                language="tr",
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                repetition_penalty=1.2,
            )
            segments = list(segments)
        else:
            segments, _ = model.transcribe(
                audio,
                language="tr",
                beam_size=5,
                vad_filter=True,
                vad_parameters=VAD_PARAMS,
                condition_on_previous_text=False,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                repetition_penalty=1.2,
            )
            segments = list(segments)
            last_end = segments[-1].end if segments else 0.0
            if last_end < max(0.0, duration - 0.6):
                progress(0.7, desc="Retrying without VAD")
                segments, _ = model.transcribe(
                    audio,
                    language="tr",
                    beam_size=5,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    repetition_penalty=1.2,
                )
                segments = list(segments)

        text = "".join(segment.text for segment in segments).strip()
        progress(0.9, desc="Post-processing")
        text = strip_repetitions(text)

        segment_data = [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text,
            }
            for segment in segments
        ]

        elapsed = time.time() - start_time
        stats = format_stats(elapsed, duration, model_name)
        sound_html = completion_sound_html()
        return text, stats, segment_data, file_path, sound_html
    except Exception as exc:
        return "", f"Error: {exc}", [], file_path or "", ""


def build_ui():
    default_model = resolve_default_model()
    missing = missing_models()
    js = build_client_js(missing)

    with gr.Blocks(title="Noisy Whisper", js=js) as demo:
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
                disable_vad = gr.Checkbox(
                    label="Disable VAD (advanced)",
                    value=False,
                )
                model_select = gr.Dropdown(
                    label="Model",
                    choices=KNOWN_MODELS,
                    value=default_model,
                    elem_id="model-select",
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
                with gr.Row():
                    export_format = gr.Dropdown(
                        label="Export format",
                        choices=["txt", "srt", "pdf", "docx"],
                        value="txt",
                    )
                    export_btn = gr.Button("Export")
                export_file = gr.File(label="Exported file")
                export_status = gr.Markdown()
                completion_sound = gr.HTML(visible=False)

        current_model = gr.State(value=default_model)
        segments_state = gr.State(value=[])
        audio_state = gr.State(value="")

        model_select.change(
            on_model_change,
            inputs=[model_select, current_model],
            outputs=[model_select, current_model, model_status],
            queue=True,
        )

        transcribe_btn.click(
            transcribe,
            inputs=[audio_file, preprocess, disable_vad, model_select],
            outputs=[
                output,
                stats,
                segments_state,
                audio_state,
                completion_sound,
            ],
            concurrency_limit=1,
        )

        export_btn.click(
            export_transcript,
            inputs=[export_format, output, segments_state, audio_state],
            outputs=[export_file, export_status],
            concurrency_limit=1,
        )

    return demo


def main():
    demo = build_ui()
    demo.queue(max_size=8)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,
    )


if __name__ == "__main__":
    main()
