import os
from pathlib import Path

# ==============================================================================
# 1. DIREKTORI & JALUR FILE (PATH CONFIGURATION)
# ==============================================================================
# Deteksi otomatis folder root proyek secara dinamis (__file__ ada di /src)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Jalur folder lokal
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = str(ROOT_DIR / "training-results")

# Deteksi Jalur Dataset Otomatis (Kaggle vs Lokal/CI)
KAGGLE_PATH = "/kaggle/input/datasets/zadaniammusthofa/dataset-qlora/dataset_sft_final_ready.jsonl"
LOCAL_PATH = DATA_PROCESSED_DIR / "dataset_sft_final_ready.jsonl"

# Pilih jalur yang aktif dan ubah ke string agar kompatibel dengan library HF
ACTIVE_PATH = KAGGLE_PATH if os.path.exists(KAGGLE_PATH) else LOCAL_PATH
DATA_PATH = str(ACTIVE_PATH)


# ==============================================================================
# 2. KONFIGURASI MODEL & HYPERPARAMETERS (Untuk 100 Data Sintetis)
# ==============================================================================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512
BATCH_SIZE_PER_DEVICE = 2
GRADIENT_ACC_STEPS = 4
