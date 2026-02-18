@echo off
REM Quick Start Script for Breathing Patterns ML Project
REM Run this file to start the frontend dashboard

echo ========================================
echo  Breathing Patterns ML - Quick Start
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo SOLUTION: Install Python 3.8+ from https://www.python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [1/4] Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies (this may take a few minutes)...
    python -m pip install --upgrade pip --quiet
    python -m pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo.
        echo SOLUTION: Try running manually: pip install -r requirements.txt
        echo Check your internet connection and Python installation
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
    python analysis\analyze_dataset.py
    if errorlevel 1 (
        echo WARNING: Dataset analysis failed, but continuing...
        echo You can run it manually later: python analysis\analyze_dataset.py
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
        python -m src.train --task breath_hold
        if errorlevel 1 (
            echo ERROR: Model training failed
            echo.
            echo SOLUTION: Check that dataset/ contains data files
            echo Run dataset analysis first: python analysis\analyze_dataset.py
            echo You can train later: python -m src.train --task breath_hold
            echo.
            pause
            exit /b 1
        )
        echo Model training completed successfully!
    ) else (
        echo Skipping model training. You can train later with: python -m src.train --task breath_hold
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

python -m streamlit run run_frontend.py

pause
