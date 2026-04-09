# Noisy Whisper — Software Specification

| Field | Value |
|---|---|
| Version | **1.1 — Draft** |
| Date | **April 2026** |
| Status | **For Review** |
| Platform | **Windows 10 / 11** |
| Interface Language | **English (Turkish localisation deferred to post-v1.0)** |

---

## 1. Purpose and Scope

This document defines the functional requirements, architecture, and delivery scope for **Noisy Whisper** — a fully offline desktop application that converts pre-recorded Turkish audio files into accurate text transcriptions, with particular emphasis on noise-degraded recordings.

The application targets Windows end users who may or may not have an NVIDIA GPU. It runs entirely locally with no cloud dependency and requires no manual software installation by the end user: setup is a single unzip-and-double-click operation.

> **Scope note:** This specification covers the v1.0 release. Live microphone input, speaker diarisation, and multi-language support are explicitly out of scope for this version.

---

## 2. Stakeholders

| Role | Party | Responsibility |
|---|---|---|
| Developer | Project team | Build, test, and package the application |
| Distributor | Client-facing stakeholder | Run `download_models.py`, zip project folder, and deliver to end users |
| End user | Windows desktop user | Unzip, double-click launcher, upload audio files, receive transcriptions |

---

## 3. System Overview

The system is a **Python-based web application** that runs locally on the user's machine. It is accessed through a standard web browser (localhost) and launched via a Windows batch file. No internet connection is required after the initial package is delivered to the end user.

### 3.1 Technology Stack

| Layer | Component | Version / Notes |
|---|---|---|
| Frontend / UI | Gradio | `>=4.44,<5.0` (pinned below 5.x to avoid breaking API changes) |
| Speech recognition | faster-whisper | `>=1.0.3` (CTranslate2 backend) |
| Model | Whisper medium / large-v3-turbo / large-v3 | OpenAI weights, bundled locally in CTranslate2 int8 format |
| Noise reduction | noisereduce | `>=3.0.2` (spectral subtraction) |
| Audio loading | librosa + soundfile | Fallback: scipy |
| Voice Activity Detection | Silero VAD | Built into faster-whisper |
| Runtime | Python 3.11 | **Embedded portable Python** (shipped inside project folder — no system Python required) |
| GPU acceleration | CUDA (optional) | Auto-detected; falls back to CPU transparently |

### 3.2 Compute Modes

The application auto-detects hardware at startup and selects the optimal compute path:

| Mode | Hardware | Compute type | Expected RTF (large-v3-turbo) |
|---|---|---|---|
| GPU | RTX 3050 or better (4 GB+ VRAM) | `int8_float16` | 0.03–0.07× (15–30× real-time) |
| CPU | Any modern x86-64 | `int8` | 0.2–0.4× (3–5× real-time) |

---

## 4. Functional Requirements

### 4.1 Audio Input

- The application must accept audio files in the following formats: MP3, WAV, M4A, FLAC, OGG, Opus, MP4, WebM.
- Files are uploaded via a drag-and-drop zone or file browser in the UI.
- There is no enforced file size limit; very long files are processed by VAD-segmented chunking inside faster-whisper.
- The application must not require the user to pre-convert audio files.

### 4.2 Preprocessing Pipeline

When noise reduction is enabled (default: **on**), the following steps are applied before transcription:

1. Audio is loaded and resampled to 16 kHz mono using librosa.
2. Spectral noise reduction is applied via `noisereduce`, estimating the noise profile from the first 500 ms of the file.
3. Peak normalisation scales the waveform to ±0.95.
4. The processed NumPy array is passed directly to faster-whisper (no intermediate file write to disk).

The user may disable preprocessing via a checkbox for recordings that are already clean.

> **Known limitation:** `noisereduce` uses spectral subtraction and performs best on stationary noise (fans, air conditioning, background hum). It degrades on non-stationary noise (overlapping voices, music, sharp transients). DeepFilterNet is a candidate replacement for v1.1 (see Section 8).

> **Known limitation:** The noise profile is estimated from the first 500 ms. Recordings that begin mid-speech may produce suboptimal noise reduction.

### 4.3 Transcription

The following faster-whisper parameters are applied on every transcription call:

| Parameter | Value | Rationale |
|---|---|---|
| `language` | `"tr"` | Fixed to Turkish; eliminates language-detection overhead and avoids misdetection on noisy audio |
| `beam_size` | `5` | Balances output quality against speed |
| `vad_filter` | `True` (Silero VAD) | Silence segments are skipped; prevents hallucination on silent passages |
| `condition_on_previous_text` | `False` | Prevents hallucination cascades — Whisper will not carry forward incorrect context from a noisy prior segment |
| `temperature` | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | Automatic fallback schedule; greedy decode (0.0) is tried first, and temperature is raised only when the decoder stalls on a noisy segment |
| `repetition_penalty` | `1.2` | Suppresses hallucinated repeated tokens |

**Post-processing:** A regex pass strips runs of three or more identical consecutive words or syllables from the output text (e.g., "di di di di di" → "di").

### 4.4 Model Selection

Users may choose from the models present in the `models/` directory:

| Model | Disk (int8) | RAM (int8, CPU) | Accuracy | Recommended for |
|---|---|---|---|---|
| `medium` | ~769 MB | ~1.5 GB | Good | CPU users who need speed |
| `large-v3-turbo` | ~809 MB | ~2.0 GB | Very good | **Default — best balance of speed and quality** |
| `large-v3` | ~1.63 GB | ~3.5 GB | Best | GPU users; highest accuracy |

The UI only displays models that are present in the `models/` directory. If a model file is absent, its option is greyed out with a tooltip: *"Model not bundled. Contact your distributor."*

**Default model:** `large-v3-turbo` is the default selection. It offers negligible WER difference versus `large-v3` for Turkish at less than half the memory footprint, making it the best choice for CPU users.

**Model switching:** Changing the model selection in the UI triggers an explicit model reload. The old model is released from memory before the new one is loaded. A progress indicator is shown during the reload. The loaded model is then cached in memory and reused for all subsequent transcriptions until the model selection changes again or the app is restarted.

### 4.5 Output

- The transcribed text is displayed in a read-only textarea in the UI.
- A **Copy to Clipboard** button is provided.
- A statistics line is shown below the output, displaying: elapsed time, audio duration, real-time factor (RTF), device used (CPU / GPU), and model name.
- Output is not persisted to disk; the user is responsible for copying or saving the text.

---

## 5. Non-Functional Requirements

### 5.1 Offline Operation

The application must function without any internet connection after the end user receives the zip package. Specifically:

- All inference runs locally using bundled model weights.
- Gradio's UI assets (CSS, JS) are bundled with the `gradio` package inside the local virtual environment; no CDN calls are made.
- No web fonts, remote APIs, or external resources are loaded at runtime.
- The application must not transmit audio data, telemetry, or any usage information.

### 5.2 Performance

- **On GPU (RTX 3050):** a 10-minute audio file must complete transcription within 90 seconds using `large-v3-turbo`.
- **On CPU (modern 4-core):** a 10-minute audio file must complete transcription within 15 minutes using `large-v3-turbo`.
- The browser UI must remain responsive during transcription (Gradio's async inference queue handles this; the transcription runs in a background thread).

### 5.3 Compatibility

- **Target OS:** Windows 10 (21H2+) and Windows 11.
- **Python:** 3.11 (embedded portable build shipped with the package).
- **Browser:** Any Chromium-based browser or Firefox (opened automatically by the launcher).
- **GPU:** NVIDIA RTX series with CUDA 11.8+ driver. CPU fallback requires no additional drivers.

### 5.4 Reliability

- The application must not crash on malformed or corrupt audio files; errors must be caught and displayed to the user in the UI.
- Hallucinations on silence are suppressed by Silero VAD and `condition_on_previous_text=False`.
- The loaded model is cached in memory for the session. Repeated transcriptions with the same model selection must not reload model weights.

---

## 6. Project File Structure

```
noisy-whisper/
├── app.py                      Main Gradio application, inference logic, preprocessing
├── requirements.txt            Python package dependencies (pinned versions)
├── launch.bat                  End-user launcher — opens the app in the default browser
├── download_models.py          Developer script to pre-download and convert models
├── python/                     Embedded portable Python 3.11 (shipped; ~25 MB)
│   ├── python.exe
│   └── ...
├── ffmpeg.exe                  Bundled ffmpeg binary — no PATH configuration required
├── models/
│   ├── medium/                 faster-whisper CTranslate2 int8 weights (optional)
│   ├── large-v3-turbo/         faster-whisper CTranslate2 int8 weights (default)
│   └── large-v3/               faster-whisper CTranslate2 int8 weights (optional)
├── venv/                       Python virtual environment — created on first launch, NOT shipped
└── README.md                   Setup and usage instructions (for the distributor)
```

> `venv/` is created automatically by `launch.bat` on first run and should be excluded from the distributed zip.

---

## 7. Deployment and Distribution

### 7.1 Developer Steps (Pre-Shipping)

1. Ensure the `python/` directory contains the embedded Python 3.11 portable build from python.org.
2. Place `ffmpeg.exe` in the project root.
3. Run `python download_models.py` and select which model(s) to bundle into `models/`.
4. Verify the `models/` directory contains the downloaded CTranslate2 int8 weights.
5. Delete `venv/` if it exists (it must not be shipped).
6. Zip the entire project folder and transmit to the client.

### 7.2 End-User Steps (First Launch)

1. Unzip the received package anywhere on the machine (e.g., Desktop or `C:\Tools\`).
2. Double-click **`launch.bat`**.
3. Wait for the virtual environment to be created and packages installed (~3–5 minutes, **requires internet once**).
4. The default browser opens automatically at `http://127.0.0.1:7860`.

On all subsequent launches, step 3 is skipped — the launcher detects that `venv/` already exists.

> **Note:** The only first-run internet requirement is pip downloading packages into `venv/`. Model weights and Python itself are already bundled.

### 7.3 How the Launcher Works

`launch.bat` executes the following logic:

```batch
@echo off
SET ROOT=%~dp0
SET PYTHON=%ROOT%python\python.exe
SET VENV=%ROOT%venv

IF NOT EXIST "%VENV%" (
    echo Creating virtual environment...
    "%PYTHON%" -m venv "%VENV%"
    "%VENV%\Scripts\pip.exe" install -r "%ROOT%requirements.txt"
)

start "" http://127.0.0.1:7860
"%VENV%\Scripts\python.exe" "%ROOT%app.py"
```

### 7.4 Prerequisites Summary

| Prerequisite | Provided by | Internet required? | Notes |
|---|---|---|---|
| Python 3.11 | Developer (bundled in `python/`) | No | Embedded portable build |
| ffmpeg | Developer (bundled as `ffmpeg.exe`) | No | Placed in project root; picked up by faster-whisper automatically |
| pip packages | End user (auto-installed on first run) | Once | Installed by `launch.bat` into `venv/` |
| Model weights | Developer (bundled in `models/`) | No | Pre-downloaded and converted by developer |
| CUDA drivers | GPU users only | Once | nvidia.com; CPU users unaffected |

---

## 8. Known Limitations

> These limitations are known and accepted for v1.0 and are candidates for v1.1.

- **Non-stationary noise:** `noisereduce` (spectral subtraction) handles stationary background noise well but degrades significantly on non-stationary noise (overlapping speech, music, sharp transients). **DeepFilterNet** is the recommended v1.1 upgrade path for better broadband noise suppression.
- **SNR floor:** Very heavy noise environments (SNR < 0 dB) may still produce degraded output regardless of preprocessing.
- **Noise profile assumption:** The `noisereduce` noise profile is estimated from the first 500 ms. Recordings that begin mid-speech may produce suboptimal noise reduction.
- **Long file RAM usage:** Files longer than 2 hours may consume significant RAM on CPU mode due to in-memory audio buffering.
- **Localhost only:** The application serves on `127.0.0.1` only and is not accessible from other devices on the network by design.
- **Model weights unencrypted:** Model files are stored unencrypted on disk. Audio files are not retained by the application, but model weights can be extracted from the `models/` directory.
- **Gradio accessibility:** The Gradio UI is not fully accessible to screen-reader users in its current configuration.
- **First-run internet dependency:** The `venv/` setup requires internet access once to download pip packages. Fully air-gapped deployments would require a pre-built `venv/` to be shipped (not in scope for v1.0).

---

## 9. Out of Scope for v1.0

- Live microphone input and real-time transcription.
- Speaker diarisation (identifying who said what).
- Languages other than Turkish.
- Translation (Turkish to English or other languages).
- Word-level timestamps or subtitle export (SRT/VTT). *(Note: faster-whisper supports this natively and is a low-effort v1.1 candidate.)*
- Standalone `.exe` packaging (PyInstaller/Nuitka).
- User authentication, multi-user sessions, or network-accessible deployment.
- Integration with cloud storage or external APIs.
- Turkish UI localisation. *(All interface text is English for v1.0. Turkish localisation is deferred to a post-release pass once core functionality is verified.)*

---

## 10. Acceptance Criteria

| # | Criterion | Priority | Test method |
|---|---|---|---|
| 1 | Audio file is uploaded and transcribed without error on both GPU and CPU machines | Must have | Manual test |
| 2 | Turkish output is produced (not English or gibberish) for a clean Turkish recording | Must have | Manual review |
| 3 | Noisy recording (café background, ~10 dB SNR) produces intelligible output with preprocessing enabled | Must have | Manual review |
| 4 | Application launches with no internet connection after first-run venv setup | Must have | Network disconnect test |
| 5 | `launch.bat` does not contact PyPI on second launch | Must have | Network monitor |
| 6 | GPU is automatically used when CUDA is available; CPU is used otherwise | Must have | Stats line check |
| 7 | A 10-minute file completes in <90 s on RTX 3050 using `large-v3-turbo` | Must have | Timed test |
| 8 | A 10-minute file completes in <15 min on CPU-only machine using `large-v3-turbo` | Must have | Timed test |
| 9 | Corrupt or empty audio file displays an error message rather than crashing | Must have | Error injection |
| 10 | `ffmpeg.exe` in project root is used without any PATH configuration | Must have | Test on clean machine with no system ffmpeg |
| 11 | `launch.bat` runs correctly on a machine with no system Python installed | Must have | Test on clean VM |
| 12 | Copy-to-clipboard button works in Chrome and Firefox | Should have | Manual test |
| 13 | Switching model in the UI reloads the model without requiring an app restart | Should have | Manual test |
| 14 | Repetitive hallucination output (e.g., repeated syllables) is suppressed | Should have | Test with known noisy sample |
| 15 | Stats line shows correct RTF, device, elapsed time, and model name | Should have | Manual review |

---

## 11. Open Questions

| # | Question | Owner |
|---|---|---|
| 1 | Should `large-v3` be omitted from the default bundle to reduce zip size, available only on explicit request? This makes the default package ~1.6 GB smaller. | Distributor to decide |
| 2 | Is the ~1-time internet requirement for `venv/` setup acceptable, or is a fully offline first-launch required? A pre-built `venv/` can be shipped but adds ~500 MB to the zip. | Distributor to confirm |
| 3 | Should word-level timestamps or SRT/VTT export be added in v1.1? faster-whisper supports this natively with minimal additional code. | For future discussion |
| 4 | Is there a typical maximum audio file length? Files >2 hours may require streaming chunked processing to avoid RAM exhaustion. | Client to advise |
| 5 | Should DeepFilterNet replace `noisereduce` in v1.1 for non-stationary noise? It requires an additional ~30 MB model but handles music/voice overlap significantly better. | Developer to evaluate |

---

## 12. Revision History

| Version | Date | Changes |
|---|---|---|
| 1.0 | April 2026 | Initial draft |
| 1.1 | April 2026 | Replaced system Python + system ffmpeg with embedded portable Python and bundled `ffmpeg.exe`; removed unquantized fine-tuned model reference; added `condition_on_previous_text` and temperature fallback schedule to transcription parameters; pinned Gradio below 5.x; clarified model-switching memory behaviour; fixed section numbering; all UI text set to English for v1.0 development |
