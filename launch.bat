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
