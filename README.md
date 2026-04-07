# NoisyWhisper

Offline Turkish transcription pipeline for noisy audio.

Current state: Core development phases are implemented (CLI + GUI + export + packaging baseline).

## Implemented

- Audio decode/normalize via ffmpeg (bundled or PATH)
- Optional DeepFilterNet denoise stage (graceful fallback if unavailable)
- Silero-VAD speech segmentation (energy fallback included)
- faster-whisper transcription with model conversion support (HF -> CTranslate2)
- Robust fallback model path for reliability when primary output is unusable
- Export outputs: TXT, SRT, JSON, DOCX, PDF
- Desktop GUI with background worker and live progress updates
- Config persistence (`config.ini`) and rotating logs (`noisywhisper.log`)
- Packaging baseline with `noisywhisper.spec` and Windows build script

## Repository Layout

```
noisywhisper/
├── main.py
├── core/
├── output/
├── ui/
├── utils/
├── requirements.txt
└── README.md
```

## Quick Start (CLI)

1. Create a Python environment (3.11+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Model setup options:

- Option A (recommended): first run auto-downloads model into local model folder.
- Default primary model source: `Cosmobillian/turkish_whisper_for_noisy_datas`.
- If the downloaded checkpoint is Transformers format, NoisyWhisper converts it to
	CTranslate2 automatically for faster-whisper.
- Option B (manual/offline): place converted CTranslate2 model files in:

```text
models/turkish-noisy-v1/
```

4. Run CLI transcription:

```bash
python main.py --file /path/to/audio.mp3 --format txt
```

Disable auto-download and require local model only:

```bash
python main.py --file /path/to/audio.mp3 --format txt --no-model-download
```

## Quick Start (GUI)

```bash
python main.py --gui
```

In GUI mode:
- Select or drag-and-drop an audio file.
- Click `Transcribe` to run offline pipeline on a background thread.
- Choose export formats and click `Export Selected`.
- Open `Settings` to persist beam size, compute type, VAD threshold, model path, and export defaults.

## CLI Examples

Single format:

```bash
python main.py --file sample.mp3 --format txt
```

Multiple formats:

```bash
python main.py --file sample.mp3 --format txt srt json
```

Disable denoise:

```bash
python main.py --file sample.wav --format txt --no-denoise
```

Custom output directory:

```bash
python main.py --file sample.m4a --format txt docx pdf --out-dir outputs
```

Override model repositories used for bootstrap:

```bash
python main.py --file sample.mp3 --format txt --model-repo Cosmobillian/turkish_whisper_for_noisy_datas --fallback-model-repo Systran/faster-whisper-large-v3
```

Force local primary/fallback models and skip downloads:

```bash
python main.py --file sample.mp3 --format txt srt json --model-path models/faster-whisper-small --fallback-model-path models/faster-whisper-small --no-model-download
```

## Testing

Run unit tests:

```bash
python -m pytest -q
```

Run compile checks:

```bash
python -m compileall main.py core output utils ui tests
```

Run full phase test script (Windows PowerShell):

```powershell
./scripts/run_phase_tests.ps1
```

## Packaging (Windows)

Build distributable from spec:

```bash
python -m PyInstaller --noconfirm --clean noisywhisper.spec
```

Or use the helper script:

```powershell
./scripts/build_windows.ps1
```

Output directory:

```text
dist/NoisyWhisper/
```

## Notes

- If DeepFilterNet or Silero-VAD are unavailable at runtime, the pipeline falls back gracefully and logs a warning.
- If DeepFilterNet fails with torchaudio backend import errors, install compatible torch/torchaudio versions from requirements.
- If you override model repos, NoisyWhisper will try to use CTranslate2 directly first, then auto-convert from Transformers checkpoint when possible.
- PyInstaller warnings in build logs can include optional GPU/TensorRT/TensorFlow references; CPU-only distribution still builds successfully.
