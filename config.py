"""
Configuration for ML pipeline: paths, split seed, and optional parameters.
"""
from pathlib import Path

# Project root (directory containing dataset/, analysis/, src/)
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Reproducibility
RANDOM_STATE = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature / labeling
WINDOW_SEC = 2.0
SAMPLE_RATE_HZ = 50
MIN_WINDOW_ROWS = 50  # exclude very short segments

# Balloon Valve Status: 1=deflated, 2=ready inhale, 3=ready exhale, 4=inflated, 5=fault
# Breath-hold / stable: inflated (4) or ready at threshold
BALLOON_BREATH_HOLD_VALUES = ("4", 4)  # inflated = breath hold
BALLOON_FREE_BREATHING_VALUES = ("1", 1)  # deflated = free breathing

# Gating Mode: "Manual Overide" vs "Automated" etc.
GATING_OK_MODES = ("Automated",)
GATING_NOT_OK_MODES = ("Manual Overide", "Manual Override")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LLM Integration (Ollama API)
# Configure these when you have Ollama API available
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama local API
# Recommended for this project: llama3.2:3b (fast, good) or mistral:7b (quality). See OLLAMA_MODELS.txt
OLLAMA_MODEL = "llama3.2:3b"  # Change to e.g. "llama3.2:3b" or "mistral" after: ollama pull <name>
OLLAMA_TIMEOUT = 180  # Seconds to wait for response (increase if using larger models)
# To use a different Ollama instance, change OLLAMA_API_URL
