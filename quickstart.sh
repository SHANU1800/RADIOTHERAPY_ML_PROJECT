#!/bin/bash
# Quick Start Script for Breathing Patterns ML Project (Linux/Mac)
# Run: bash quickstart.sh

echo "========================================"
echo " Breathing Patterns ML - Quick Start"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "[ERROR] Python is not installed or not in PATH"
    exit 1
fi

echo "[OK] Python found: $(python --version)"
echo ""

# Check dependencies
echo "[1/4] Checking dependencies..."
if ! python -c "import streamlit" 2>/dev/null; then
    echo "[INFO] Installing dependencies..."
    python -m pip install -r requirements.txt --quiet
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies"
        exit 1
    fi
    echo "[OK] Dependencies installed!"
else
    echo "[OK] Dependencies installed"
fi

echo ""

# Check dataset analysis
echo "[2/4] Checking dataset analysis..."
if [ ! -f "analysis/output/file_summary.csv" ]; then
    echo "[INFO] Running dataset analyzer..."
    python analysis/analyze_dataset.py
    if [ $? -ne 0 ]; then
        echo "[WARNING] Analyzer had issues, but continuing..."
    fi
else
    echo "[OK] Dataset analysis exists"
fi

echo ""

# Check trained model
echo "[3/4] Checking trained model..."
if [ ! -f "models/best_model.pkl" ]; then
    echo "[WARNING] No trained model found!"
    echo ""
    read -p "Do you want to train a model now? (y/n): " train
    if [ "$train" = "y" ] || [ "$train" = "Y" ]; then
        echo "[INFO] Training model (this may take a few minutes)..."
        python -m src.train --task breath_hold
        if [ $? -ne 0 ]; then
            echo "[ERROR] Model training failed"
            exit 1
        fi
        echo "[OK] Model trained!"
    else
        echo "[INFO] Skipping model training. Train later with: python -m src.train --task breath_hold"
    fi
else
    echo "[OK] Model found!"
fi

echo ""
echo "[4/4] Starting Streamlit frontend..."
echo ""
echo "========================================"
echo " Frontend will open in your browser"
echo " Press Ctrl+C to stop the server"
echo "========================================"
echo ""

streamlit run run_frontend.py
