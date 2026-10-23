# src/training/__init__.py
# Package untuk modul training QLoRA SFT.
#
# Modul yang tersedia dalam package ini:
#   - config        : Konfigurasi path direktori dan hyperparameter training
#   - train         : Pipeline training utama (QLoRA SFT)
#   - utils         : Utilitas (chat template, BitsAndBytes config, monkey patch)
#   - merge_and_quantize : Roadmap untuk merge adaptor LoRA dan kuantisasi model