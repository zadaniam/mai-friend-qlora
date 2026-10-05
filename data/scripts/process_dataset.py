import json
import os
import random

# File output final
file_output = "dataset_sft_final_ready.jsonl"
semua_data = []

print("Memulai proses pembacaan file json...")

# 1. Loop untuk membaca dataset_sft_batch_1.json sampai batch_10.json
for i in range(1, 11):
    file_name = f"dataset_sft_batch_{i}.json"
    
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            try:
                data_json = json.load(f)
                # Pastikan datanya berbentuk list/array seperti di gambar kamu
                if isinstance(data_json, list):
                    semua_data.extend(data_json)
                    print(f"✅ Berhasil memuat {len(data_json)} data dari {file_name}")
                else:
                    semua_data.append(data_json)
            except json.JSONDecodeError:
                print(f"❌ Gagal membaca file: {file_name}. Periksa apakah ada kurung atau koma yang kurang.")
    else:
        print(f"⚠️ File tidak ditemukan: {file_name}")


# 2. Lakukan Shuffling (Sangat penting untuk SFT agar model tidak bias pada klaster tertentu)
random.seed(42)  # Mengunci seed agar hasil acak konsisten jika di-run ulang
random.shuffle(semua_data)
print(f"🔄 Selesai mengacak total {len(semua_data)} data percakapan.")


# 3. Tulis langsung ke dalam satu file .jsonl (Format satu baris per satu percakapan)
with open(file_output, 'w', encoding='utf-8') as f:
    for item in semua_data:

        # Mengubah role 'model' menjadi 'assistant' agar kompatibel dengan standard Hugging Face
        for msg in item["messages"]:
            if msg["role"] == "model":
                msg["role"] = "assistant"

        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\n🎉 Selesai! File final siap pakai disimpan dengan nama: {file_output}")
