import json
import os
import random

from src.config import FINAL_DATASET_PATH, RAW_DIR

# ==========================================
# CONFIGURATION / PATH CONSTANTS
# ==========================================
# Folder tempat batch JSON disimpan (sesuai output generator sebelumnya)
INPUT_DIR = RAW_DIR
# Lokasi file output final (.jsonl)
OUTPUT_FILE = FINAL_DATASET_PATH


def merge_and_format_datasets(input_dir: str, output_file: str, num_batches: int = 10):
    """Membaca banyak file batch JSON, mengacak data, mengubah role,

    dan menyimpannya ke dalam satu file .jsonl untuk standard SFT.
    """
    semua_data = []

    print(f"Memulai proses pembacaan file json dari folder '{input_dir}'...")

    # 1. Loop untuk membaca dataset_sft_batch_1.json sampai batch_X.json
    for i in range(1, num_batches + 1):
        file_name = f"dataset_sft_batch_{i}.json"
        file_path = os.path.join(input_dir, file_name)

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data_json = json.load(f)

                    # Memastikan datanya berbentuk list/array
                    if isinstance(data_json, list):
                        semua_data.extend(data_json)
                        print(
                            f"✅ Berhasil memuat {len(data_json)} data dari {file_name}"
                        )
                    else:
                        semua_data.append(data_json)
                        print(f"✅ Berhasil memuat 1 data objek dari {file_name}")

                except json.JSONDecodeError:
                    print(
                        f"❌ Gagal membaca file: {file_name}. Periksa sintaks JSON (kurung/koma)."
                    )
        else:
            print(f"⚠️ File tidak ditemukan: {file_path}")

    if not semua_data:
        print("❌ Tidak ada data yang berhasil dimuat. Proses penggabungan dibatalkan.")
        return

    # 2. Lakukan Shuffling (Penting untuk SFT agar model tidak bias pada klaster tertentu)
    random.seed(42)  # Mengunci seed agar hasil acak konsisten jika di-run ulang
    random.shuffle(semua_data)
    print(f"🔄 Selesai mengacak total {len(semua_data)} data percakapan.")

    # 3. Tulis langsung ke dalam satu file .jsonl (Format satu baris per satu percakapan)
    try:
        # Memastikan folder untuk output_file sudah ada jika berupa path lengkap
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in semua_data:
                # Mengubah role 'model' menjadi 'assistant' agar kompatibel dengan standard Hugging Face
                if "messages" in item:
                    for msg in item["messages"]:
                        if msg["role"] == "model":
                            msg["role"] = "assistant"

                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"\n🎉 Selesai! File final siap pakai disimpan di: {output_file}")

    except OSError as e:
        print(f"❌ Gagal menyimpan file karena masalah sistem/koneksi disk: {e}")
    except (KeyError, TypeError) as e:
        print(f"❌ Gagal memproses struktur data SFT (Format salah): {e}")


if __name__ == "__main__":
    # Menjalankan fungsi penggabungan dengan memasukkan konstanta global sebagai argumen
    merge_and_format_datasets(
        input_dir=INPUT_DIR, output_file=OUTPUT_FILE, num_batches=10
    )
