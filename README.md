# NoisyWhisper

Offline Turkish transcription pipeline for noisy audio.

Current state: Phase 1 CLI MVP scaffold is implemented.

## Implemented (v0.1 baseline)

- Audio decode/normalize with bundled or system ffmpeg
- Optional DeepFilterNet denoise stage (graceful fallback)
- Silero-VAD segmentation (with energy-based fallback)
- faster-whisper transcription wrapper (Turkish language)
- Export writers: TXT, SRT, JSON, DOCX, PDF
- Config persistence (`config.ini`) and rotating logs (`noisywhisper.log`)

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

## Quick Start

1. Create a Python 3.11 environment.
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

If you want to disable auto-download and require a local model only:

```bash
python main.py --file /path/to/audio.mp3 --format txt --no-model-download
```

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

## Notes

- If DeepFilterNet or Silero-VAD are unavailable at runtime, the pipeline falls back gracefully and logs a warning.
- If DeepFilterNet fails with torchaudio backend import errors, install compatible torch/torchaudio versions from requirements.
- If you override model repos, NoisyWhisper will try to use CTranslate2 directly first, then auto-convert from Transformers checkpoint when possible.
- The GUI files are currently shell components for Phase 2.
- Packaging hardening is planned for Phase 3.
