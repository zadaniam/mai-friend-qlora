import os

from src.config import (
    FINAL_DATASET_PATH,
    OUTPUT_DIR
)

# Path (diimpor dari src.config)
DATA_PATH = str(FINAL_DATASET_PATH)

# Deteksi Jalur Dataset Otomatis (Kaggle vs Lokal/CI)
KAGGLE_PATH = "/kaggle/input/datasets/zadaniammusthofa/dataset-qlora/dataset_sft_final_ready.jsonl"
ACTIVE_PATH = KAGGLE_PATH if os.path.exists(KAGGLE_PATH) else DATA_PATH
DATA_PATH = str(ACTIVE_PATH)

# Jalur untuk output train.py & smoke test
OUTPUT_DIR = str(OUTPUT_DIR)

# ==============================================================================
# KONFIGURASI MODEL & HYPERPARAMETERS (Untuk 100 Data Sintetis)
# ==============================================================================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512
BATCH_SIZE_PER_DEVICE = 2
GRADIENT_ACC_STEPS = 4
