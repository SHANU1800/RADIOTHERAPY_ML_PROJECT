# Quick Start: Frontend Dashboard

## Prerequisites

1. **Trained model**: Make sure you have trained a model first:
   ```bash
   python -m src.train --task breath_hold
   ```
   This creates `models/best_model.pkl` and `models/metrics_models.json`.

2. **Dependencies**: Install frontend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Frontend

From the project root directory:

```bash
streamlit run run_frontend.py
```

The app will automatically open in your browser at `http://localhost:8501`.

## First Steps

1. **Home Page**: Check that model is loaded and see dataset stats
2. **Upload & Predict**: 
   - Upload a `.dat`, `.txt`, or `.csv` file from `dataset/` folder
   - Select task (breath_hold or gating_ok)
   - Click "Run Prediction"
   - View interactive plot and download predictions CSV
3. **Dataset Explorer**: Browse existing patient files and visualize curves
4. **Model Performance**: View confusion matrices and feature importance
5. **Batch Analysis**: Upload multiple files for batch processing

## Troubleshooting

- **"Model not found"**: Train a model first using the command above
- **Import errors**: Make sure you're in the project root directory
- **File upload errors**: Check file format matches expected structure (see `README_ML.md`)

## Features

- ✅ Interactive Plotly visualizations
- ✅ Real-time predictions
- ✅ Batch file processing
- ✅ CSV export
- ✅ Model performance metrics
- ✅ Feature importance visualization
