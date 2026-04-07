$ErrorActionPreference = "Stop"

Write-Host "[NoisyWhisper] Running compile checks..."
C:/Users/alisa/AppData/Local/Programs/Python/Python313/python.exe -m compileall main.py core output utils ui tests

Write-Host "[NoisyWhisper] Running unit tests..."
C:/Users/alisa/AppData/Local/Programs/Python/Python313/python.exe -m pytest -q

Write-Host "[NoisyWhisper] Running CLI smoke test on sample.mp3..."
C:/Users/alisa/AppData/Local/Programs/Python/Python313/python.exe main.py --file sample.mp3 --format txt srt json --out-dir outputs --model-path models/faster-whisper-small --fallback-model-path models/faster-whisper-small --no-model-download

Write-Host "[NoisyWhisper] Phase test suite completed."
