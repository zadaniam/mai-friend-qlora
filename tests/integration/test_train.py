import os
import sys
from pathlib import Path
import pytest

# Pastikan Python bisa menemukan folder /src dari folder root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# --- PERBAIKAN DI SINI: Sesuaikan jalur impor config ke subfolder training ---
from src.training import config
from src.training.train import main

def test_pipeline_training_smoke(monkeypatch, tmp_path):
    """
    Smoke test untuk memastikan seluruh pipeline training berjalan lancar
    tanpa error (menggunakan lingkungan terisolasi CPU dan folder sementara).
    """
    # 1. Gunakan folder sementara agar tidak mengotori komputer lokal Anda
    output_dir = tmp_path / "test-training-results"
    
    # 2. Paksa konfigurasi agar berjalan di mode CPU
    monkeypatch.setenv("FORCE_CPU", "true")
    
    # 3. Lakukan monkeypatch pada objek modul config yang sudah di-import dengan benar
    monkeypatch.setattr(config, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(config, "EPOCHS", 1)  # Cukup 1 epoch untuk tes cepat

    success = False
    
    # 4. Jalankan fungsi utama training
    try:
        main()
        success = True
    except Exception as e:
        pytest.fail(f"Pipeline training gagal memicu error: {e}")

    # 5. Verifikasi akhir
    assert success is True
