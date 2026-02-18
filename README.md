# Breathing Patterns ML Project

**ML pipeline and web dashboard for analyzing breathing patterns in radiation therapy gating data.**

## 🚀 Quick Start

**Just double-click `quickstart.bat` (Windows) or run `bash quickstart.sh` (Linux/Mac)**

The script will automatically:
- Install dependencies
- Check dataset analysis
- Check for trained model
- Start the web dashboard

See `START_HERE.txt` for the simplest instructions, or `QUICKSTART.md` for detailed guide.

---

## 📁 Project Structure

```
KMCProject/
├── quickstart.bat          # Windows quick start (double-click!)
├── quickstart.ps1          # PowerShell quick start
├── quickstart.sh           # Linux/Mac quick start
├── START_HERE.txt          # Simple instructions
│
├── dataset/                # Patient breathing curve data (.dat, .txt, .csv)
├── analysis/               # Dataset analysis scripts
│   └── output/            # Summary CSVs
├── src/                    # ML pipeline code
│   ├── load_data.py       # Unified data loader
│   ├── features.py        # Feature engineering
│   ├── labels.py          # Label definitions
│   └── train.py           # Training script
├── models/                 # Trained models (created after training)
├── frontend/               # Streamlit web dashboard
│   ├── app.py             # Main app
│   ├── pages/             # Dashboard pages
│   └── utils/             # Frontend utilities
└── config.py              # Configuration
```

---

## 🎯 Features

### ML Pipeline
- ✅ Unified data loading (.dat, .txt, .csv)
- ✅ Patient-based train/test splits
- ✅ Classical ML models (RandomForest, XGBoost, SVM, LogisticRegression)
- ✅ Classification: Breath-hold vs Free-breathing, Gating OK vs Not OK
- ✅ Feature engineering with windowed statistics

### Web Dashboard
- ✅ Upload files and get predictions
- ✅ Interactive visualizations (Plotly)
- ✅ Dataset explorer
- ✅ Model performance metrics
- ✅ Batch file processing
- ✅ CSV export
- ✅ AI Assistant (LLM integration for explanations and Q&A)

---

## 📚 Documentation

### Quick Start Guides
- **`START_HERE.txt`** - Simplest quick start guide
- **`QUICKSTART.md`** - Detailed quick start with troubleshooting
- **`HOW_TO_USE.txt`** - Complete user guide with step-by-step instructions

### Technical Documentation
- **`HOW_IT_WORKS.txt`** - Technical architecture and system design
- **`WHAT_IT_HAS.txt`** - Complete feature list and capabilities
- **`LIMITATIONS.txt`** - System constraints and what it cannot do
- **`TROUBLESHOOTING.txt`** - Common errors and solutions

### Specialized Guides
- **`README_ML.md`** - ML pipeline documentation
- **`ML_WORKFLOW_GUIDE.txt`** - How ML works & adding new data (START HERE for understanding workflow)
- **`frontend/README.md`** - Frontend dashboard guide
- **`analysis/README.md`** - Dataset analysis guide
- **`analysis/DATASET_SUMMARY.txt`** - Dataset analysis summary report

---

## 🔧 Manual Setup (if quickstart doesn't work)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run dataset analyzer:**
   ```bash
   python analysis/analyze_dataset.py
   ```

3. **Train model:**
   ```bash
   python -m src.train --task breath_hold
   ```

4. **Start frontend:**
   ```bash
   streamlit run run_frontend.py
   ```

---

## 🎓 Usage Examples

### Training a model:
```bash
python -m src.train --task breath_hold
python -m src.train --task gating_ok
```

### Using the ML pipeline in Python:
```python
from src.load_data import load_all_patients
from src.features import build_windows, get_X_y

# Load data
df, _ = load_all_patients("dataset")

# Build features
windows = build_windows(df, window_sec=2.0)
X, y, patient_ids = get_X_y(windows, task="breath_hold")
```

### Using the frontend:
1. Run `quickstart.bat` or `streamlit run run_frontend.py`
2. Open browser to `http://localhost:8501`
3. Upload files, explore dataset, view metrics

---

## 📊 Data Format

Breathing curve files contain time-series data:
- **Session Time** (seconds)
- **Volume (liters)**
- **Balloon Valve Status** (1=deflated, 4=inflated)
- **Patient Switch** (0/1)
- **Gating Mode** (Automated/Manual Overide)

See `README_ML.md` for detailed format specification.

---

## 🤖 AI Assistant (LLM Integration)

The project includes optional LLM integration for AI-powered explanations and Q&A.

### Features
- **Explain Predictions**: Get detailed explanations of prediction results
- **Answer Questions**: Ask questions about the dataset, models, or system
- **Generate Reports**: Create analysis reports automatically

### Setup
1. Install and run Ollama: https://ollama.ai
2. Configure in `config.py`:
   ```python
   OLLAMA_API_URL = "http://localhost:11434/api/generate"
   OLLAMA_MODEL = "llama2"  # or "mistral", "codellama", etc.
   ```
3. Restart the frontend

The AI Assistant page provides a chat interface and integration points throughout the dashboard.

---

## 🐛 Troubleshooting

For detailed troubleshooting, see **`TROUBLESHOOTING.txt`**.

Quick fixes:
- **Python not found**: Install Python 3.8+ and add to PATH
- **Dependencies error**: Run `pip install -r requirements.txt`
- **Model not found**: Train with `python -m src.train --task breath_hold`
- **Import errors**: Make sure you're in the project root directory
- **Blank pages**: Refresh browser, check console (F12)

---

## 📝 License

This project is for research/clinical use in radiation therapy gating.
