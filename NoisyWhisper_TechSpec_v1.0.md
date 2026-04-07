# NoisyWhisper — Technical Specification Document

**Version:** 1.0  
**Date:** April 2026  
**Status:** Pre-development  
**Project:** NoisyWhisper  

---

## Table of Contents

1. Project Overview  
2. Functional Requirements  
3. System Architecture  
4. Tech Stack  
5. Data Flow  
6. GUI Specification  
7. File & Folder Structure  
8. Phased Roadmap  
9. Known Risks & Mitigations  
10. Acceptance Criteria  

---

## 1. Project Overview

### 1.1 Purpose

NoisyWhisper is a fully offline, standalone desktop application that transcribes
Turkish-language audio files into text. It is designed to handle real-world audio
conditions including high levels of background noise, street recordings,
phone-quality microphones, echo, and reverb. The application requires no internet
connection, no cloud API, and no Python installation on the end user's machine.

### 1.2 Problem Statement

Existing online transcription services (Google Speech, AssemblyAI, Deepgram)
require internet access and transmit audio data to external servers. General-purpose
offline tools such as vanilla Whisper underperform on noisy Turkish audio without
domain-specific fine-tuning. No existing standalone desktop tool combines
Turkish-specific noise-robust ASR with a clean offline user experience in a single
distributable binary.

### 1.3 Goals

- Transcribe Turkish audio files accurately in a fully offline environment
- Handle audio with significant background noise without requiring user preprocessing
- Accept all common audio file formats as input without manual conversion
- Produce output in user-selectable formats: TXT, DOCX, PDF, SRT, JSON
- Package as a single distributable folder requiring zero installation by the end user
- Run on consumer-grade hardware without a dedicated GPU

### 1.4 Non-Goals (v1.0)

- Real-time microphone transcription (deferred to v2.0)
- Speaker diarization / identifying who said what
- Translation (Turkish to other languages)
- Mobile platforms (Android, iOS)
- Cloud or networked deployment
- Support for non-Turkish languages

---

## 2. Functional Requirements

### F-01 — Audio File Ingestion

The application shall accept audio files via drag-and-drop onto the application
window or via a file picker dialog. Supported input formats:

  Container formats: MP3, MP4, M4A, M4B, WAV, FLAC, OGG, OPUS, AAC, WMA, AIFF,
  AMR, 3GP, MKV, WEBM, RA, CAF, AU, APE, WV, TTA, MPC, DSF, DFF

  Source of support: bundled FFmpeg binary handles all container and codec parsing.
  If FFmpeg can decode it, NoisyWhisper can accept it.

### F-02 — Noise Preprocessing

Before transcription, all audio shall be passed through DeepFilterNet, a neural
two-stage noise suppression pipeline, to attenuate background noise, reverb, and
echo. This stage is enabled by default and can be toggled off in Settings for
clean-audio use cases where speed is preferred over noise handling.

### F-03 — Voice Activity Detection

The denoised audio shall be segmented using Silero-VAD to isolate speech regions
and discard silence. This prevents Whisper hallucination during silent segments
and reduces total inference time.

### F-04 — Transcription

The preprocessed audio segments shall be transcribed using
Cosmobillian/turkish_whisper_for_noisy_datas_v1, a fine-tuned Whisper Large V3
checkpoint stored locally. The transcription language is fixed to Turkish (tr).
Word-level timestamps shall be generated for all segments.

### F-05 — Export Formats

The user shall select one or more output formats before exporting:

  - TXT   Plain text transcript, no formatting or timestamps
  - DOCX  Formatted Word document with title, timestamp header, and paragraph breaks
  - PDF   Exportable PDF with title block, metadata, and formatted transcript body
  - SRT   Standard subtitle format with segment-level timestamps (HH:MM:SS,ms)
  - JSON  Machine-readable format with segment text, start_ms, end_ms, confidence

### F-06 — Progress Reporting

The application shall display real-time progress during all pipeline stages:
current stage name, percentage of audio processed, active segment text preview,
and estimated time remaining. The GUI shall remain fully responsive during processing.

### F-07 — Model Management

On first launch, the application shall check for model weights in the local
models/ directory. If absent, a download prompt shall appear with a progress bar.
Model weights are never re-downloaded if already present. Users may also manually
place pre-downloaded weights in the models/ directory.

### F-08 — Settings Persistence

All user preferences shall be saved to config.ini on change and restored on
next launch. Persistent settings include: noise reduction toggle, beam size,
default export format(s), default export folder path, compute type preference.

### F-09 — Error Handling

The application shall display clear, non-technical error messages for:
  - Unsupported or corrupted audio file
  - Insufficient disk space for export
  - Model weights missing or corrupted
  - Out-of-memory conditions during inference
All errors shall be logged to noisywhisper.log in the application directory.

---

## 3. System Architecture

### 3.1 Architectural Pattern

The application follows a layered pipeline architecture with strict separation
between the GUI layer, the processing worker, and the output layer. All heavy
computation runs exclusively on a background QThread. Communication between the
worker and GUI uses Qt Signals and Slots exclusively — no shared mutable state.

### 3.2 Pipeline Stages

```
[Audio File Input]
      |
      v
Stage 1: DECODE & NORMALIZE
  Library : ffmpeg-python (subprocess wrapper, bundled ffmpeg binary)
  Input   : Any supported audio format at any sample rate
  Output  : PCM float32, 16kHz, mono
  Notes   : Handles all containers, codecs, stereo-to-mono downmix, resampling

      |
      v
Stage 2: NOISE REDUCTION
  Library : deepfilternet (df package, CPU inference)
  Input   : PCM float32, 16kHz mono (resampled to 48kHz internally)
  Output  : Cleaned PCM float32, 16kHz mono
  Notes   : Two-stage neural suppression — ERB envelope + deep harmonic filtering
            Handles stationary noise, reverb, echo, wind, crowd noise
            Throughput ~10-20x realtime on modern CPU

      |
      v
Stage 3: VOICE ACTIVITY DETECTION
  Library : silero-vad (ONNX runtime, no PyTorch required)
  Input   : Clean PCM float32, 16kHz mono
  Output  : List of speech windows [(start_ms, end_ms), ...]
  Notes   : Silence discarded before inference to prevent hallucination
            Configurable speech/silence threshold in Settings

      |
      v
Stage 4: TRANSCRIPTION
  Library : faster-whisper (CTranslate2 backend)
  Model   : Cosmobillian/turkish_whisper_for_noisy_datas_v1 (local)
  Input   : PCM audio chunk per VAD speech window
  Output  : Segment text + word-level timestamps + confidence score
  Config  : language=tr, beam_size=5, compute_type=int8 (CPU) or float16 (GPU)
            condition_on_previous_text=True, word_timestamps=True

      |
      v
Stage 5: OUTPUT ASSEMBLY
  Input   : All segments with timestamps
  Output  : TXT / DOCX / PDF / SRT / JSON on user request
  Notes   : DOCX via python-docx, PDF via reportlab, SRT with line-length limits
```

### 3.3 Threading Model

```
Main Thread (Qt Event Loop)
  |
  |-- GUI interactions --> emit signals
  |
  +-- QThread: TranscriptionWorker
        |
        |-- ffmpeg decode
        |-- DeepFilterNet denoise
        |-- silero-vad segmentation
        |-- faster-whisper inference (blocking per chunk)
        +-- emit progressUpdate(stage, pct, segment_text)
                  |
                  +-- GUI slot: update progress bar + transcript viewer
```

---

## 4. Tech Stack

| Component          | Technology                          | Version   |
|--------------------|-------------------------------------|-----------|
| Language           | Python                              | 3.11      |
| GUI Framework      | PyQt6                               | 6.7+      |
| Audio Decode       | ffmpeg-python + bundled FFmpeg bin  | FFmpeg 7.x|
| Noise Reduction    | deepfilternet (df)                  | 0.5.6+    |
| VAD                | silero-vad (ONNX)                   | 5.x       |
| ASR Engine         | faster-whisper                      | 1.x       |
| ASR Backend        | CTranslate2                         | 4.x       |
| ASR Model          | turkish_whisper_for_noisy_datas_v1  | HF snap   |
| DOCX Export        | python-docx                         | 1.x       |
| PDF Export         | reportlab                           | 4.x       |
| Settings           | configparser                        | stdlib    |
| Logging            | logging                             | stdlib    |
| Packaging          | PyInstaller (--onedir)              | 6.x       |

---

## 5. Data Flow Diagram

```
[File on Disk] ──ffmpeg──> [PCM 16kHz mono]
                                  |
              ┌───────────────────┘
              | resample 48kHz
              v
        [DeepFilterNet]
              |
              | resample back 16kHz
              v
        [Silero-VAD] ──> [(t0,t1),(t2,t3),...] speech windows
              |
              | foreach window
              v
       [faster-whisper]
              |
              v
    [segment: text, t_start, t_end, confidence]
              |
         ┌────┴────────────────────────────┐
         v         v          v       v    v
       [TXT]    [DOCX]      [PDF]  [SRT] [JSON]
```

---

## 6. GUI Layout Specification

### 6.1 Main Window

```
+----------------------------------------------------------+
|  NoisyWhisper                              [-][□][X]     |
+----------------------------------------------------------+
|                                                          |
|   +--------------------------------------------------+  |
|   |                                                  |  |
|   |        Drag & drop audio file here               |  |
|   |          or  [ Browse File... ]                  |  |
|   |                                                  |  |
|   |  Selected: interview_recording.mp3   [X Clear]   |  |
|   +--------------------------------------------------+  |
|                                                          |
|  [ ▶  Transcribe ]       [ ⚙ Settings ]  [ ? About ]   |
|                                                          |
|  --------------------------------------------------------|
|  Stage: Denoising audio...                              |
|  [████████████░░░░░░░░░]  58%     ETA: 2m 04s           |
|  --------------------------------------------------------|
|                                                          |
|  +--------------------------------------------------+   |
|  | Transcript                           [Copy All]  |   |
|  |                                                  |   |
|  | [00:00:03] Merhaba, bugün hava çok güzel...      |   |
|  | [00:00:09] Evet, sokakta insanlar yürüyor...     |   |
|  | [00:00:14] Çocuklar parkta oynuyor...            |   |
|  |                                                  |   |
|  +--------------------------------------------------+   |
|                                                          |
|  Export as:  [✓ TXT] [✓ DOCX] [ PDF] [ SRT] [ JSON]    |
|  Export to:  /Users/user/Desktop/    [ Change... ]      |
|                                                          |
|  [ Export Selected ]                                     |
+----------------------------------------------------------+
```

### 6.2 Settings Dialog

```
+------------------------------------------+
|  Settings                          [X]   |
+------------------------------------------+
|  Transcription                           |
|    Noise Reduction:  [ ON  ]  [ OFF ]    |
|    Beam Size:        [----o--------] 5   |
|                       1           10     |
|    Compute Type:     [ Auto ▼ ]          |
|                      (Auto/int8/float16) |
|                                          |
|  VAD Sensitivity                         |
|    Speech Threshold: [------o------]     |
|                       Low         High   |
|                                          |
|  Export                                  |
|    Default Format(s): [✓TXT][DOCX][PDF] |
|                       [SRT ][JSON]       |
|    Export Folder:  /Desktop/  [Browse]   |
|                                          |
|  Advanced                                |
|    Log Level:    [ INFO ▼ ]              |
|    Model Path:   ./models/  [Browse]     |
|                                          |
|      [ Save ]          [ Cancel ]        |
+------------------------------------------+
```

---

## 7. File & Folder Structure

### 7.1 Distribution Layout (End User)

```
NoisyWhisper/
├── NoisyWhisper.exe             <- PyInstaller bundle (Windows)
├── NoisyWhisper                 <- PyInstaller bundle (Linux)
├── models/
│   └── turkish-noisy-v1/
│       ├── model.bin            <- CTranslate2 weights (~1.6 GB)
│       ├── tokenizer.json
│       ├── vocabulary.json
│       └── config.json
├── ffmpeg.exe                   <- bundled FFmpeg binary (Windows)
├── ffmpeg                       <- bundled FFmpeg binary (Linux)
├── config.ini                   <- auto-generated on first launch
└── noisywhisper.log             <- runtime log, auto-rotated at 5 MB
```

### 7.2 Source Repository Layout (Developer)

```
noisywhisper/
├── main.py                      <- app entry point, Qt app init
├── ui/
│   ├── mainwindow.py            <- PyQt6 main window, drag-drop, progress
│   └── settings_dialog.py      <- settings panel dialog
├── core/
│   ├── worker.py                <- QThread TranscriptionWorker
│   ├── audio.py                 <- ffmpeg decode + PCM normalization
│   ├── denoise.py               <- DeepFilterNet wrapper
│   ├── vad.py                   <- silero-vad wrapper
│   └── transcribe.py           <- faster-whisper wrapper
├── output/
│   ├── txt_writer.py
│   ├── docx_writer.py           <- python-docx formatter
│   ├── pdf_writer.py            <- reportlab formatter
│   ├── srt_writer.py            <- SRT formatter with line-length limits
│   └── json_writer.py
├── utils/
│   ├── config.py                <- configparser wrapper
│   └── logger.py                <- rotating log handler setup
├── models/                      <- gitignored, populated at runtime
├── assets/
│   └── icon.ico
├── requirements.txt
├── noisywhisper.spec            <- PyInstaller spec file
└── README.md
```

---

## 8. Phased Development Roadmap

### Phase 1 — CLI MVP (v0.1)

Deliverable: `python main.py --file audio.mp3 --format txt` works end-to-end.  
Scope:
- Audio decode via ffmpeg-python
- DeepFilterNet denoising
- Silero-VAD segmentation
- faster-whisper transcription, language=tr
- TXT output only
- No GUI

### Phase 2 — GUI (v0.2)

Deliverable: Full PyQt6 window, all export formats functional.  
Scope:
- Main window with drag-and-drop
- Progress bar with stage labels and ETA
- Live transcript viewer updating per segment
- All five export formats: TXT, DOCX, PDF, SRT, JSON
- Settings dialog (noise toggle, beam size, export path)

### Phase 3 — Packaging (v0.3)

Deliverable: Distributable NoisyWhisper/ folder, zero end-user dependencies.  
Scope:
- PyInstaller .spec configured
- CTranslate2 binary hooks manually added
- DeepFilterNet ONNX runtime bundled via --collect-all df
- FFmpeg binary bundled as data file
- Model auto-download on first launch with progress dialog
- Windows and Linux builds validated

### Phase 4 — Polish & Release (v0.4 → v1.0)

Deliverable: Stable v1.0 release candidate.  
Scope:
- Error handling with user-friendly messages
- Log rotation and crash reporting to file
- Config persistence across updates
- DOCX/PDF output formatting refinement (title block, font, margins)
- End-to-end testing on clean Windows 10 and Ubuntu 22.04 machines

### Future — v2.0

- Live microphone transcription with streaming Whisper
- Optional speaker diarization (pyannote.audio, separate process)
- Batch transcription (multiple files queued)

---

## 9. Known Risks & Mitigations

| Risk                                         | Likelihood | Mitigation                                                        |
|----------------------------------------------|------------|-------------------------------------------------------------------|
| CTranslate2 dynamic lib not found by PyInstaller | High   | Manually add AVX2/AVX512 .dll paths in .spec binaries list        |
| DeepFilterNet ONNX bundling failure          | Medium     | Pin deepfilternet==0.5.6, add --collect-all df hook               |
| Model weights absent on first launch         | Medium     | Auto-detect at startup, show download dialog before any action    |
| Whisper hallucination on silence             | Low        | Silero-VAD strips silence before inference (Stage 3)              |
| Long audio file (>1 hr) causing memory spike | Low        | Stream in 30s VAD-defined chunks, never load full file to RAM     |
| python-docx / reportlab bundling conflict    | Low        | Test packaging early in v0.3; these are pure-Python libraries     |
| FFmpeg binary missing or wrong architecture  | Low        | Bundle platform-specific FFmpeg binary, validate at startup       |

---

## 10. Acceptance Criteria (v1.0)

- A .mp3 Turkish audio file with street background noise transcribes with
  intelligible output and no crash on a clean Windows 10 machine
- The exported .srt file plays correctly synced in VLC against the original audio
- The exported .docx opens without errors in Microsoft Word and LibreOffice
- The exported .pdf renders correctly in all major PDF viewers
- The application launches on a clean machine with no Python installed and
  no internet connection (model weights pre-placed)
- Peak RAM during transcription of a 10-minute file does not exceed 5 GB
  on CPU-only mode
- The full distributable folder (excluding model weights) does not exceed 1.5 GB
- All five export formats produce correct output from the same transcription run

---

*End of NoisyWhisper Technical Specification v1.0*
