@echo off
REM Start the Breathing Patterns ML frontend (uses Python 3.12 for TensorFlow support)
cd /d "%~dp0"

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12 not found. Trying default python...
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in PATH
        echo Install Python 3.12 from https://www.python.org for full support including Deep Learning
        pause
        exit /b 1
    )
    set PYCMD=python
) else (
    set PYCMD=py -3.12
)

echo Starting Streamlit...
echo Open the URL shown below in your browser. Press Ctrl+C to stop.
echo.
%PYCMD% -m streamlit run run_frontend.py
pause
