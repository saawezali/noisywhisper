$ErrorActionPreference = "Stop"

Write-Host "[NoisyWhisper] Installing/validating build dependencies..."
python -m pip install --upgrade pip
python -m pip install pyinstaller

Write-Host "[NoisyWhisper] Running tests before packaging..."
python -m pytest -q

Write-Host "[NoisyWhisper] Building ONEFILE distributable via PyInstaller spec..."
python -m PyInstaller --noconfirm --clean noisywhisper_onefile.spec

Write-Host "[NoisyWhisper] Build complete. Output file: dist/NoisyWhisper.exe"
