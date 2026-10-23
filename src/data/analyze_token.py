import json
import os

import numpy as np
from transformers import AutoTokenizer

# Konfigurasi Path & Model Bawaan
DATA_PATH = os.path.join("data", "processed", "dataset_sft_final_ready.jsonl")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def analyze_dataset_sequence_length(data_path: str, model_id: str):
    if not os.path.exists(data_path):
        # Mendorong pesan kesalahan ke stdout agar mudah ditangkap pengujian biasa
        print(f"❌ File dataset tidak ditemukan di: {data_path}")
        return

    print(f"Memuat tokenizer untuk {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    daftar_panjang_token = []

    print("Memulai analisis seluruh baris dataset...")
    with open(data_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            if not line.strip():
                continue

            # --- SEKARANG AMAN: Dibungkus try-except untuk menangani json.loads yang rusak ---
            try:
                data_row = json.loads(line)
                # 1. Terapkan chat template resmi Qwen
                teks_mentah = tokenizer.apply_chat_template(
                    data_row["messages"], tokenize=False
                )
                # 2. Encode untuk mendapatkan token IDs
                token_ids = tokenizer.encode(teks_mentah)
                daftar_panjang_token.append(len(token_ids))
            except (json.JSONDecodeError, KeyError) as e:
                print(
                    f"⚠️ Peringatan: Melewati baris {index} karena format salah/rusak. ({e})"
                )

    if not daftar_panjang_token:
        print("❌ Tidak ada data percakapan valid yang berhasil dianalisis.")
        return

    # 3. Hitung statistik distribusi menggunakan numpy
    max_len = max(daftar_panjang_token)
    min_len = min(daftar_panjang_token)
    mean_len = int(np.mean(daftar_panjang_token))

    # Persentil penting untuk best practice
    p95 = int(np.percentile(daftar_panjang_token, 95))
    p99 = int(np.percentile(daftar_panjang_token, 99))

    print("\n==================================================")
    print("📊 HASIL ANALISIS DISTRIBUSI TOKEN DATASET SFT")
    print("==================================================")
    print(f"- Total Data Percakapan : {len(daftar_panjang_token)} data")
    print(f"- Token Terpendek       : {min_len} token")
    print(f"- Rata-rata (Mean)      : {mean_len} token")
    print(f"- Token Terpanjang (Max): {max_len} token")
    print("--------------------------------------------------")
    print(f"📌 Rekomendasi P95      : {p95} token (95% data Anda di bawah ini)")
    print(f"📌 Rekomendasi P99      : {p99} token (99% data Anda di bawah ini)")
    print("==================================================")

    # Aturan pengambilan keputusan max_seq_length
    if p99 <= 245:  # Diturunkan sedikit dari 256 agar memberi ruang padding aman
        rekomendasi = 256
    elif p99 <= 490:
        rekomendasi = 512
    elif p99 <= 980:
        rekomendasi = 1024
    else:
        rekomendasi = int(
            2 ** np.ceil(np.log2(p99))
        )  # Mengambil pangkat dua terdekat di atas P99

    print(f"💡 STRATEGI TERBAIK: Gunakan max_seq_length = {rekomendasi}")
    print(
        f"   (Hasil P99 Anda {p99}. Angka {rekomendasi} akan menghemat VRAM & mempercepat komputasi GPU)."
    )
    print("==================================================\n")


if __name__ == "__main__":
    analyze_dataset_sequence_length(DATA_PATH, MODEL_ID)
