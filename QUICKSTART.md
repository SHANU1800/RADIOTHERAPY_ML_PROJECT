# Quick Start Guide

## One-Command Start (Recommended)

### Windows (Double-click or run):
```
quickstart.bat
```

### Windows PowerShell:
```powershell
.\quickstart.ps1
```

### Linux/Mac:
```bash
bash quickstart.sh
```

The script will automatically:
1. ✅ Check and install dependencies if needed
2. ✅ Run dataset analysis if not already done
3. ✅ Check for trained model (prompts to train if missing)
4. ✅ Start the Streamlit frontend dashboard

The frontend will open automatically in your browser at `http://localhost:8501`.

---

## Manual Start (Alternative)

If you prefer to run commands manually:

### 1. Install dependencies (first time only):
```bash
pip install -r requirements.txt
```

### 2. Run dataset analyzer (optional):
```bash
python analysis/analyze_dataset.py
```

### 3. Train model (if not already trained):
```bash
python -m src.train --task breath_hold
```

### 4. Start frontend:
```bash
streamlit run run_frontend.py
```

---

## What Each Script Does

- **quickstart.bat** - Windows batch file (double-click to run)
- **quickstart.ps1** - PowerShell script (right-click → Run with PowerShell)
- **quickstart.sh** - Bash script for Linux/Mac

All scripts perform the same checks and start the frontend automatically.

---

## Troubleshooting

- **"Python not found"**: Install Python 3.8+ and add it to PATH
- **"Permission denied"** (Linux/Mac): Run `chmod +x quickstart.sh` first
- **"Streamlit not found"**: The script will auto-install, or run `pip install -r requirements.txt` manually
- **"Model not found"**: The script will prompt you to train one, or run `python -m src.train --task breath_hold`

---

## Next Steps

Once the frontend is running:
1. Go to **Home** page to see overview
2. Try **Upload & Predict** to test predictions
3. Explore **Dataset Explorer** to browse patient data
4. Check **Model Performance** for metrics

See `frontend/README.md` for detailed frontend documentation.
