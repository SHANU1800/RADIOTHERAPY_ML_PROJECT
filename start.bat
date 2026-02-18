@echo off
REM Start the Breathing Patterns ML frontend (no checks, no train prompt)
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Starting Streamlit...
echo Open the URL shown below in your browser. Press Ctrl+C to stop.
echo.
python -m streamlit run run_frontend.py
pause
