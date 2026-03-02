@echo off
REM Train deep learning models (all 7 architectures) for breath-hold classification.
REM Requires: Python 3.9–3.12 and tensorflow.
REM
REM Models trained: LSTM, CNN1D, CNN-LSTM, BiLSTM, GRU, Attention-LSTM, ResNet1D
REM Features: Multi-channel (volume + derivatives + balloon + envelope)
REM           50%% overlapping windows, class weight balancing
cd /d "%~dp0"

echo ========================================
echo  Deep Learning Model Training
echo ========================================
echo.

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    set PYCMD=python
    echo WARNING: Python 3.12 not found. Using default python.
    echo TensorFlow requires Python 3.9-3.12.
) else (
    set PYCMD=py -3.12
    echo Using Python 3.12
)

echo.
echo Training all 7 DL architectures with multi-channel input...
echo This may take a while depending on your hardware.
echo.

%PYCMD% -m src.dl_train --task breath_hold --model all --multichannel --overlap 0.5 --epochs 50 %*

if errorlevel 1 (
  echo.
  echo ERROR: Training failed.
  echo If you see "No matching distribution found for tensorflow", use Python 3.11 or 3.12.
  echo Example: py -3.12 -m pip install tensorflow
  echo.
)

echo.
echo Training complete. Check the models/ directory for results.
pause
