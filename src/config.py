from pathlib import Path

# Root direktori proyek (src/ → parent = project root)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Path data
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GENERATION_DIR = DATA_DIR / "generation"

# File data spesifik
SEED_PROMPTS_PATH = GENERATION_DIR / "seed_prompts.json"
FINAL_DATASET_PATH = PROCESSED_DIR / "dataset_sft_final_ready.jsonl"

# Path output training
OUTPUT_DIR = ROOT_DIR / "training-results"