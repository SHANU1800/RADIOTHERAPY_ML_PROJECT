@echo off
REM Quick Start Script for Breathing Patterns ML Project
REM Run this file to start the frontend dashboard (uses Python 3.12 for TensorFlow + DL)
cd /d "%~dp0"

echo ========================================
echo  Breathing Patterns ML - Quick Start
echo ========================================
echo.

REM Prefer Python 3.12 (has TensorFlow); fallback to default python
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    set PYCMD=python
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in PATH
        echo.
        echo SOLUTION: Install Python 3.12 from https://www.python.org
        echo Make sure to check "Add Python to PATH" during installation
        echo.
        pause
        exit /b 1
    )
    echo Using default Python (Deep Learning may be unavailable without 3.12)
) else (
    set PYCMD=py -3.12
    echo Using Python 3.12
)

echo.
echo [1/4] Checking dependencies...
%PYCMD% -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies (this may take a few minutes)...
    %PYCMD% -m pip install --upgrade pip --quiet
    %PYCMD% -m pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo.
        echo SOLUTION: Try running manually: py -3.12 -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    echo Dependencies installed successfully!
) else (
    echo Dependencies OK
)

echo.
echo [2/4] Checking dataset analysis...
if not exist "analysis\output\file_summary.csv" (
    echo Running dataset analyzer...
    %PYCMD% analysis\analyze_dataset.py
    if errorlevel 1 (
        echo WARNING: Dataset analysis failed, but continuing...
    ) else (
        echo Dataset analysis completed successfully!
    )
) else (
    echo Dataset analysis already exists
)

echo.
echo [3/4] Checking trained model...
if not exist "models\best_model.pkl" (
    echo WARNING: No trained model found!
    echo.
    set /p train="Train model now? Enter y for yes or n for no: "
    if /i "%train%"=="y" (
        echo Training model (this may take a few minutes)...
        %PYCMD% -m src.train --task breath_hold
        if errorlevel 1 (
            echo ERROR: Model training failed. You can train later: %PYCMD% -m src.train --task breath_hold
        ) else (
            echo Model training completed successfully!
        )
    ) else (
        echo Skipping model training. Train later with: %PYCMD% -m src.train --task breath_hold
    )
) else (
    echo Model found!
)

echo.
echo [4/4] Starting Streamlit frontend...
echo.
echo ========================================
echo  Frontend will open in your browser
echo  Press Ctrl+C to stop the server
echo ========================================
echo.

%PYCMD% -m streamlit run run_frontend.py

pause
