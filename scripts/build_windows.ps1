$ErrorActionPreference = "Stop"

Write-Host "[NoisyWhisper] Installing/validating build dependencies..."
python -m pip install --upgrade pip
python -m pip install pyinstaller

Write-Host "[NoisyWhisper] Running tests before packaging..."
python -m pytest -q

Write-Host "[NoisyWhisper] Building distributable via PyInstaller spec..."
pyinstaller --noconfirm --clean noisywhisper.spec

Write-Host "[NoisyWhisper] Build complete. Output folder: dist/NoisyWhisper"
