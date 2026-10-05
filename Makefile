SHELL := /bin/bash

.PHONY: install test train clean help

help:
	@echo "Perintah yang tersedia:"
	@echo "  make install  - Sinkronisasi environment & dependensi menggunakan uv sync"
	@echo "  make test     - Jalankan automated testing di CPU via uv"
	@echo "  make train    - Jalankan QLoRA SFT Training via uv (Butuh GPU)"
	@echo "  make clean    - Hapus virtual environment dan cache"

install:
	# Cek apakah 'uv' sudah terinstall di sistem, jika belum maka install otomatis
	@command -v uv >/dev/null 2>&1 || pip install --user uv
	# uv sync otomatis membuat .venv, membaca pyproject.toml, dan menginstall grup [test]
	uv sync --extra test

test-unit:
	FORCE_CPU="true" uv run pytest tests/unit

test-integration:
	FORCE_CPU="true" uv run pytest tests/integration -v

train:
	# Menjalankan training di dalam lingkungan terisolasi .venv milik uv
	uv run python -m src.train

clean:
	rm -rf .venv .uv .pytest_cache tests/__pycache__ src/__pycache__
