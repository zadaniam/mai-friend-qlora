import json
import os
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# 1. Definisi Skema Output Menggunakan Pydantic (Wajib untuk SFT agar format konsisten)
class SingleMessage(BaseModel):
    role: str = Field(description="Harus berupa 'user' atau 'model'")
    content: str = Field(description="Isi percakapan pada turn tersebut")

class ConversationTurn(BaseModel):
    messages: list[SingleMessage] = Field(description="Daftar percakapan multiturn antara user dan model")

class DatasetResponse(BaseModel):
    conversations: list[ConversationTurn] = Field(description="Kumpulan 10 data percakapan multiturn")

# 2. Inisialisasi Gemini Client
client = genai.Client()


# 3. Prompt Pengaturan
PROMPT_GENERAL = """
Anda adalah generator data sintetis AI yang ahli. Tugas Anda adalah membuat contoh percakapan.
Bertindaklah sebagai ahli pembuat dataset AI (Data Engineer) untuk melatih Small Language Model (SLM) menjadi teman cerita yang tulus dan berempati tinggi. 
Tugasmu adalah menghasilkan data percakapan multi-turn antara "user" (orang yang sedang curhat) dan "model" (teman dekat tempat curhat). 

Kamu WAJIB mengikuti aturan ketat berikut dalam menyusun respons "model":

1. GAYA BAHASA & PERSONA
- Gunakan bahasa Indonesia kasual yang santai, tulus, dewasa, dan tenang.
- Gunakan kata ganti "aku-kamu".
- JANGAN GUNAKAN bahasa yang terlalu formal, gunakan diksi yang lebih kasual (misalnya: eh, hmm, bentar, deh, sih, aja, nggak)
- Tetapi JANGAN GUNAKAN bahasa gaul daerah yang terlalu spesifik atau terkesan "sok asyik" (Hindari kata: gue, lu, asli, parah sih, toxic, bro, dll).
- Gaya mengetik harus natural seperti sahabat lama yang sedang bertukar pesan di WhatsApp/Telegram.

2. BATASAN PANJANG KALIMAT (MAKSIMAL 1-2 KALIMAT)
- Setiap respons "model" harus sangat singkat, padat, dan langsung ke inti perasaan. Maksimal hanya 1-2 kalimat per giliran (turn).
- JANGAN membuat paragraf panjang yang melelahkan dibaca.
- Buat minimal 3 turn dan maksimal 5 turn.

3. ANTI-SOK TAHU & TANPA ASUMSI
- Respons harus didasarkan HANYA pada informasi yang sudah jelas-jelas diucapkan oleh "user".
- JANGAN pernah mendahului informasi, berasumsi tentang penyebab masalah, atau menyimpulkan terlalu cepat sebelum "user" menceritakannya sendiri.

4. ALUR EMPATI YANG NATURAL (ANTI-ROBOT)
- JANGAN memberikan daftar solusi, tips numerik (1, 2, 3), atau nasihat panjang kecuali jika "user" meminta solusi secara eksplisit.
- Berikan validasi emosi yang instan dan ringkas di awal
- WAJIB memberikan mengakhiri tanggapan setiap turn dengan pertanyaan lembut agar "user" mau melanjutkan ceritanya secara bertahap.

Catatan Penting: Ikuti skema JSON output yang diminta oleh sistem. Pastikan dalam setiap percakapan (conversation) diisi minimal 3-5 giliran (turn) interaksi pasang user-model secara berurutan, diawali dengan pesan dari user.
"""


# Daftar variasi topik/prompt spesifik untuk setiap batch (Total 10 batch)
PROMPT_BATCH_SPESIFIK = [
    # KLASTER 1: Dunia Kerja & Akademik
    "Sekarang, buatkan 10 data percakapan multi-turn yang unik mengenai: Masalah dengan Atasan atau Rekan Kerja (seperti bos yang toksik, merasa tidak dihargai, disalahkan secara tidak adil, atau rekan kerja lepas tanggung jawab). Pastikan setiap skenario cerita berbeda.",
    
    "Sekarang, buatkan 10 data percakapan multi-turn yang unik mengenai: Burnout & Kebingungan Arah Akademik/Karja (seperti lelah dengan beban tugas, merasa salah jurusan, cemas menghadapi ujian besar, atau kehilangan motivasi hidup). Pastikan setiap skenario cerita berbeda.",

    # KLASTER 2: Hubungan Asmara & Patah Hati
    "Sekarang, buatkan 10 data percakapan multi-turn yang unik mengenai: Konflik & Kekecewaan dalam Hubungan Asmara (seperti pasangan yang tiba-tiba cuek, menyembunyikan sesuatu, atau merasa hubungan sudah tidak sehat tapi sulit untuk lepas). Pastikan setiap skenario cerita berbeda.",
    
    "Sekarang, buatkan 10 data percakapan multi-turn yang unik mengenai: Kedukaan Pasca-Putus Cinta / Heartbreak (seperti menangis di malam hari karena rindu mantan, sulit merelakan, atau merasa tidak akan bisa membuka hati lagi). Pastikan setiap skenario cerita berbeda.",

    # KLASTER 3: Krisis Kepercayaan Diri
    "Sekarang, buatkan 10 data percakapan multi-turn yang unik mengenai: Insecurity & Membandingkan Diri (seperti minder melihat pencapaian teman di media sosial, merasa fisik atau finansial kurang, atau merasa tertinggal jauh di usia sekarang). Pastikan setiap skenario cerita berbeda.",
    
    "Sekarang, buatkan 10 data percakapan multi-turn yang unik mengenai: Imposter Syndrome & Ketakutan Gagal (seperti merasa kesuksesannya hanya keberuntungan, takut mengecewakan ekspektasi orang tua, atau cemas berlebihan sebelum mencoba hal baru). Pastikan setiap skenario cerita berbeda.",

    # KLASTER 4: Dinamika Keluarga & Pertemanan
    "Sekarang, buatkan 10 data percakapan multi-turn yang unik mengenai: Tekanan dan Konflik Internal Keluarga (seperti selalu dibanding-bandingkan oleh orang tua, dituntut cepat sukses/menikah, atau suasana rumah yang tidak harmonis). Pastikan setiap skenario cerita berbeda.",
    
    "Sekarang, buatkan 10 data percakapan multi-turn yang unik mengenai: Kekecewaan dalam Lingkaran Pertemanan (seperti merasa dijauhi secara sepihak, dikhianati sahabat dekat, atau merasa hanya dimanfaatkan saat mereka butuh saja). Pastikan setiap skenario cerita berbeda.",

    # KLASTER 5: Kesepian & Krisis Eksistensial
    "Sekarang, buatkan 10 data percakapan multi-turn yang unik mengenai: Rasa Kesepian yang Mendalam / Loneliness (seperti pulang ke kos malam-malam merasa sepi, merasa tidak ada yang peduli, atau tidak punya tempat berbagi cerita sehari-hari). Pastikan setiap skenario cerita berbeda.",
    
    #"Sekarang, buatkan 10 data percakapan multi-turn yang unik mengenai: Quarter-Life Crisis & Cemas Masa Depan (seperti bingung hidup mau dibawa ke mana, merasa tidak punya bakat menonjol, atau takut menghadapi fase kedewasaan). Pastikan setiap skenario cerita berbeda."
]


all_synthetic_data = []

print("Memulai pembuatan data sintetis SFT...")

# 4. Proses Batching Loop (10 Batch x 10 Data = 100 Data)
for index, spesifik_prompt in enumerate(PROMPT_BATCH_SPESIFIK, start=1):
    print(f"\n[Batch {index}/10] Sedang memproses topik: {spesifik_prompt.split('.')[0]}...")
    
    # Gabungkan Prompt General dengan Prompt Spesifik Batch ini
    prompt_full = f"{PROMPT_GENERAL}\nInstruksi Khusus Batch ini: {spesifik_prompt}\nBuat tepat 10 pasang dialog multiturn unik."

    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite', # Disarankan menggunakan gemini-2.5-flash untuk performa terbaik saat ini
            contents=prompt_full,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DatasetResponse, # Memaksa output mengikuti skema Pydantic
                temperature=0.8 # Temperature agak tinggi agar hasil lebih bervariasi dan kreatif
            ),
        )
        
        # Parse hasil JSON dari Gemini
        batch_json = json.loads(response.text)
        batch_data = batch_json.get("conversations", [])
        
        # --- PERUBAHAN DI SINI: Simpan langsung per batch ke file terpisah ---
        batch_filename = f"dataset_sft_batch_{index}.json"
        with open(batch_filename, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
        print(f"-> Berhasil mendapatkan {len(batch_data)} data dan disimpan ke '{batch_filename}'.")
        
        # (Opsional) Tetap gabungkan ke list utama jika Anda masih ingin rekap total di akhir terminal
        all_synthetic_data.extend(batch_data)
        
    except Exception as e:
        print(f"-> Gagal memproses Batch {index} karena error: {e}")
    
    # Jeda antar batch untuk menghindari hit Rate Limit (TPM/RPM) Gemini
    if index < len(PROMPT_BATCH_SPESIFIK):
        print("Menunggu jeda aman antar batch...")
        time.sleep(8)

# 5. Cetak Laporan Akhir Saja (Tidak perlu simpan file gabungan lagi di sini)
print(f"\n Selesai! Semua file batch terpisah berhasil disimpan. Total keseluruhan: {len(all_synthetic_data)} data.")
