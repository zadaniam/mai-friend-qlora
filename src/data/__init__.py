# src/data/__init__.py
# Package untuk modul pipeline data.
#
# Modul yang tersedia dalam package ini:
#   - generate_dataset : Generate data sintetis melalui Gemini API
#   - process_dataset  : Merge batch JSON menjadi satu file JSONL untuk SFT
#   - analyze_token    : Analisis distribusi panjang token dataset