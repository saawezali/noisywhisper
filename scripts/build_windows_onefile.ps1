$ErrorActionPreference = "Stop"

$PreferredPython = "C:/Users/alisa/AppData/Local/Programs/Python/Python313/python.exe"
$PythonExe = if (Test-Path $PreferredPython) { $PreferredPython } else { "python" }

Write-Host "[NoisyWhisper] Installing/validating build dependencies..."
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install pyinstaller

Write-Host "[NoisyWhisper] Running tests before packaging..."
& $PythonExe -m pytest -q

Write-Host "[NoisyWhisper] Building ONEFILE distributable via PyInstaller spec..."
& $PythonExe -m PyInstaller --noconfirm --clean noisywhisper_onefile.spec

Write-Host "[NoisyWhisper] Build complete. Output file: dist/NoisyWhisper.exe"
