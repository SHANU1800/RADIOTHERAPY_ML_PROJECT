# Frontend: Breathing Patterns ML Dashboard

Streamlit web application for interactive analysis and prediction of breathing patterns.

## Quick Start

From the project root:

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the frontend
streamlit run run_frontend.py
```

The app will open in your browser at `http://localhost:8501`.

## Pages

1. **Home**: Overview with quick stats and navigation
2. **Upload & Predict**: Upload a curve file and get predictions
3. **Dataset Explorer**: Browse and visualize existing patient data
4. **Model Performance**: View model metrics and confusion matrices
5. **Batch Analysis**: Process multiple files at once

## Features

- **File Upload**: Supports `.dat`, `.txt`, and `.csv` breathing curve files
- **Interactive Visualizations**: Plotly charts with zoom, pan, and hover tooltips
- **Real-time Predictions**: Upload files and get instant predictions
- **Model Metrics**: View confusion matrices, feature importance, and performance metrics
- **Batch Processing**: Process multiple files simultaneously
- **CSV Export**: Download predictions and analysis results

## Requirements

See `requirements.txt` for full list. Key dependencies:
- streamlit>=1.28.0
- plotly>=5.17.0
- pandas, numpy, scikit-learn (from ML pipeline)

## Troubleshooting

- **Model not found**: Train a model first using `python -m src.train --task breath_hold`
- **Import errors**: Make sure you're running from the project root directory
- **File upload errors**: Check that the file format matches expected structure (see README_ML.md)
