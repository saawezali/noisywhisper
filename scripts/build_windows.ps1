$ErrorActionPreference = "Stop"

$PreferredPython = "C:/Users/alisa/AppData/Local/Programs/Python/Python313/python.exe"
$PythonExe = if (Test-Path $PreferredPython) { $PreferredPython } else { "python" }

# Prevent DLL pollution from active Conda sessions (e.g., anaconda3\Library\bin\icu*.dll).
$OldPath = $env:PATH
$pathParts = $env:PATH -split ';' | Where-Object {
	$_ -and ($_ -notmatch '(?i)\\anaconda3(\\|$)') -and ($_ -notmatch '(?i)\\miniconda3(\\|$)')
}
$env:PATH = ($pathParts -join ';')

Write-Host "[NoisyWhisper] Installing/validating build dependencies..."
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install pyinstaller

Write-Host "[NoisyWhisper] Running tests before packaging..."
& $PythonExe -m pytest -q

Write-Host "[NoisyWhisper] Building distributable via PyInstaller spec..."
& $PythonExe -m PyInstaller --noconfirm --clean noisywhisper.spec

Write-Host "[NoisyWhisper] Build complete. Output folder: dist/NoisyWhisper"

# Restore original PATH for interactive shell continuity.
$env:PATH = $OldPath
