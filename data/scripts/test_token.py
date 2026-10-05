import torch
from transformers import AutoTokenizer

# 1. Inisialisasi Tokenizer Qwen (Simulasi)
# Pastikan nama model sesuai dengan yang Anda gunakan di Kaggle (misal: "Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

# 2. Hardcode Teks Terpanjang Anda
sample_data = {"messages": [{"role": "user", "content": "Aku lagi sering banget insecure sama bentuk fisikku akhir-akhir ini. Pas lihat foto-foto di Instagram, temen-temenku kok pada kelihatan glowing dan proporsional semua ya."}, {"role": "assistant", "content": "Capek banget pasti ya kalau mata kita terus-terusan disuguhi standar penampilan orang lain di media sosial. Sejak kapan kamu mulai sering ngerasa insecure kayak gini?"}, {"role": "user", "content": "Belakangan ini sih, semenjak sering scrolling reels sebelum tidur. Pas ngaca jadi ngerasa kurang in a way."}, {"role": "assistant", "content": "Pantas saja perasaanmu jadi nggak tenang akhir-akhir ini. Bagian mana dari diri kamu yang lagi paling bikin kamu nggak nyaman?"}]}

# 3. Ubah Menjadi Chat Template Qwen (tokenize=False untuk melihat teks mentahnya dulu)
teks_lengkap = tokenizer.apply_chat_template(sample_data["messages"], tokenize=False)

# 4. Hitung Jumlah Token ID yang Dihasilkan
token_ids = tokenizer.encode(teks_lengkap)
jumlah_token = len(token_ids)

print(f"Hasil Analisis Teks:")
print(f"- Jumlah Karakter Teks Mentah : {len(teks_lengkap)} karakter")
print(f"- Jumlah Token setelah di-encode: {jumlah_token} token")

