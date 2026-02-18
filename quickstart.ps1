# Quick Start Script for Breathing Patterns ML Project (PowerShell)
# Run: .\quickstart.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Breathing Patterns ML - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check dependencies
Write-Host "[1/4] Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import streamlit" 2>&1 | Out-Null
    Write-Host "[OK] Dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "[INFO] Installing dependencies..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Dependencies installed!" -ForegroundColor Green
}

Write-Host ""

# Check dataset analysis
Write-Host "[2/4] Checking dataset analysis..." -ForegroundColor Yellow
if (-not (Test-Path "analysis\output\file_summary.csv")) {
    Write-Host "[INFO] Running dataset analyzer..." -ForegroundColor Yellow
    python analysis\analyze_dataset.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARNING] Analyzer had issues, but continuing..." -ForegroundColor Yellow
    }
} else {
    Write-Host "[OK] Dataset analysis exists" -ForegroundColor Green
}

Write-Host ""

# Check trained model
Write-Host "[3/4] Checking trained model..." -ForegroundColor Yellow
if (-not (Test-Path "models\best_model.pkl")) {
    Write-Host "[WARNING] No trained model found!" -ForegroundColor Yellow
    Write-Host ""
    $train = Read-Host "Do you want to train a model now? (y/n)"
    if ($train -eq "y" -or $train -eq "Y") {
        Write-Host "[INFO] Training model (this may take a few minutes)..." -ForegroundColor Yellow
        python -m src.train --task breath_hold
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Model training failed" -ForegroundColor Red
            exit 1
        }
        Write-Host "[OK] Model trained!" -ForegroundColor Green
    } else {
        Write-Host "[INFO] Skipping model training. Train later with: python -m src.train --task breath_hold" -ForegroundColor Yellow
    }
} else {
    Write-Host "[OK] Model found!" -ForegroundColor Green
}

Write-Host ""
Write-Host "[4/4] Starting Streamlit frontend..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Frontend will open in your browser" -ForegroundColor Cyan
Write-Host " Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

streamlit run run_frontend.py
