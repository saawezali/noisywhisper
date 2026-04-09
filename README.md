# Noisy Whisper

Offline Turkish audio transcription for Windows 10/11. The app runs locally in a browser and uses faster-whisper with optional noise reduction.

## Quick start (distributor)

1. Place embedded Python 3.11 in the `python/` folder (from python.org portable build).
2. Place `ffmpeg.exe` in the project root.
3. Run `python download_models.py` and select the model(s) to bundle.
4. Verify `models/` contains the downloaded CTranslate2 weights.
5. Delete `venv/` if it exists.
6. Zip the entire folder and deliver to end users.

## End user first launch

1. Unzip the folder anywhere (Desktop or `C:\Tools\`).
2. Double-click `launch.bat`.
3. Wait for the first-run venv setup (internet required once).
4. The browser opens at `http://127.0.0.1:7860`.

## Model bundles

The UI only loads models present in `models/`. Default is `large-v3-turbo` when available.

## Export

Transcriptions can be exported as TXT, SRT, PDF, or DOCX from the UI.

PDF export uses `fonts/DejaVuSans.ttf` to render Turkish characters. The font
license is included at `fonts/DejaVuSans.LICENSE.txt`.

## Notes

- No internet is required after the first run.
- Audio is processed locally and is not uploaded anywhere.
- If a model is missing, the UI will warn to contact the distributor.
- A short sound plays when transcription completes.
