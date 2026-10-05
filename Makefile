SHELL := /bin/bash

# ==============================================================================
# 1. DEKLARASI PERINTAH SEMU (.PHONY)
# ==============================================================================
.PHONY: install init test test-unit test-integration train clean help

# ==============================================================================
# 2. MENU BANTUAN (HELP MENU)
# ==============================================================================
help:
	@echo "Perintah yang tersedia untuk proyek Mai Friend:"
	@echo "  make install          - Sinkronisasi environment & dependensi menggunakan uv sync"
	@echo "  make init             - Setup awal proyek (install dependensi + pasang pre-commit)"
	@echo "  make test             - Jalankan semua pengujian (Unit & Integration Test)"
	@echo "  make test-unit        - Jalankan pengujian unit pemformatan chat template saja"
	@echo "  make test-integration - Jalankan pengujian integrasi simulasi training skala mini"
	@echo "  make train            - Jalankan QLoRA SFT Training via uv (Butuh GPU)"
	@echo "  make clean            - Hapus virtual environment dan berkas sampah cache"

# ==============================================================================
# 3. ALUR KERJA AUTOMATION & TESTING
# ==============================================================================
install:
	# Cek apakah 'uv' sudah terinstall di sistem, jika belum maka install otomatis
	@command -v uv >/dev/null 2>&1 || pip install --user uv
	# uv sync otomatis membaca pyproject.toml dan menginstall grup [dependency-groups.dev]
	uv sync --group dev

init: install
	@echo "📦 Memasang sistem pre-commit hooks standar enterprise..."
	uv run pre-commit install
	@echo "✅ Inisialisasi proyek Mai Friend berhasil diselesaikan!"

# Target gabungan untuk menjalankan seluruh rangkaian tes
test: test-unit test-integration

test-unit:
	FORCE_CPU="true" uv run pytest tests/unit

test-integration:
	FORCE_CPU="true" uv run pytest tests/integration -v

train:
	# Menjalankan training di dalam lingkungan terisolasi .venv milik uv
	uv run python -m src.train

clean:
	rm -rf .venv .uv .pytest_cache tests/__pycache__ src/__pycache__
