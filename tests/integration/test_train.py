import os
import sys
import shutil
from pathlib import Path
import pytest

# Pastikan Python bisa menemukan folder /src dari folder root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.train import main

def test_pipeline_training_smoke(monkeypatch, tmp_path):
    """
    Smoke test untuk memastikan seluruh pipeline training berjalan lancar
    tanpa error (menggunakan lingkungan terisolasi CPU dan folder sementara).
    """
    # 1. Gunakan folder sementara agar tidak mengotori komputer lokal Anda
    output_dir = tmp_path / "test-training-results"
    
    # 2. Paksa konfigurasi agar berjalan di mode CPU dan menyimpan ke folder sementara
    monkeypatch.setenv("FORCE_CPU", "true")
    monkeypatch.setattr("src.config.OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr("src.config.EPOCHS", 1)  # Cukup 1 epoch untuk tes cepat

    # 3. Jalankan fungsi utama training
    try:
        main()
        success = True
    except Exception as e:
        pytest.fail(f"Pipeline training gagal memicu error: {e}")

    # 4. Verifikasi akhir: karena di CPU, ia tidak menyimpan adapter, 
    # namun pastikan proses eksekusi berhasil sampai baris terakhir
    assert success is True
