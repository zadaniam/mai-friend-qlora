import json
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ==========================================
# CONFIGURATION / PATH CONSTANTS
# ==========================================
PROMPT_PATH = os.path.join("data", "generation", "seed_prompts.json")
RAW_OUTPUT_DIR = os.path.join("data", "raws")

# 1. Skema Output Pydantic
class SingleMessage(BaseModel):
    role: str = Field(description="Harus berupa 'user' atau 'model'")
    content: str = Field(description="Isi pesan kasual pada turn tersebut. Jika role adalah model, maksimal 1-2 kalimat.")

class ConversationTurn(BaseModel):
    messages: list[SingleMessage] = Field(
        description="Daftar percakapan multiturn bolak-balik antara user dan model. WAJIB berisi 6 sampai 10 pesan (artinya 3 sampai 5 pasang interaksi user-model secara berurutan, diawali user)."
    )

class DatasetResponse(BaseModel):
    conversations: list[ConversationTurn] = Field(description="Kumpulan tepat 10 data percakapan multiturn unik untuk topik yang diminta.")


def load_prompts(prompt_path: str) -> tuple[str, list[str]]:
    """Membaca file prompt eksternal."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"File prompt tidak ditemukan di: {prompt_path}")
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # --- JIKA PROMPT_GENERAL ADALAH LIST, GABUNGKAN DENGAN NEWLINE ---
    prompt_general_raw = data["prompt_general"]
    if isinstance(prompt_general_raw, list):
        prompt_general = "\n".join(prompt_general_raw)
    else:
        prompt_general = prompt_general_raw
        
    return prompt_general, data["prompt_batch_spesifik"]


# --- PERUBAHAN DI SINI: Fungsi sekarang menerima parameter ---
def run_generation(prompt_file: str, output_dir: str):
    """Fungsi utama untuk menjalankan pipeline generator data sintetis."""
    load_dotenv()
    
    # Gunakan variabel dari parameter, bukan konstanta global langsung
    os.makedirs(output_dir, exist_ok=True)
    
    # Muat prompt dari file eksternal
    try:
        prompt_general, prompt_batch_spesifik = load_prompts(prompt_file)
        print("-> Berhasil memuat prompt templates.")
    except Exception as e:
        print(f"-> Gagal memuat prompt: {e}")
        return

    # Inisialisasi Gemini Client (memasukkan API key secara otomatis dari .env)
    client = genai.Client()
    all_synthetic_data = []

    print(f"Memulai pembuatan data sintetis SFT (Total: {len(prompt_batch_spesifik)} Batch)...")

    for index, spesifik_prompt in enumerate(prompt_batch_spesifik, start=1):
        # Mengambil potongan teks untuk log agar terminal tetap rapi
        if ":" in spesifik_prompt:
            topik_log = spesifik_prompt.split(":")[1].split("(")[0].strip()
        else:
            topik_log = spesifik_prompt[:40]

        print(f"\n[Batch {index}/{len(prompt_batch_spesifik)}] Memproses topik: {topik_log}...")
        
        prompt_full = f"{prompt_general}\nInstruksi Khusus Batch ini: {spesifik_prompt}\nBuat tepat 10 pasang dialog multiturn unik."

        try:
            # Menggunakan SDK google-genai terbaru v1.0+
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite', 
                contents=prompt_full,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DatasetResponse,
                    temperature=0.8 
                ),
            )
            
            batch_json = json.loads(response.text)
            batch_data = batch_json.get("conversations", [])
            
            # --- PERUBAHAN DI SINI: Simpan ke output_dir dari parameter ---
            batch_filename = os.path.join(output_dir, f"dataset_sft_batch_{index}.json")
            with open(batch_filename, "w", encoding="utf-8") as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
                
            print(f"   ✓ Sukses! {len(batch_data)} data disimpan ke '{batch_filename}'.")
            all_synthetic_data.extend(batch_data)
            
        except Exception as e:
            print(f"   ✗ Gagal pada Batch {index} karena error: {e}")
        
        # Jeda anti-rate limit
        if index < len(prompt_batch_spesifik):
            print("   ... Menunggu jeda aman (8 detik) ...")
            time.sleep(8)

    print(f"\n Selesai! Semua file disimpan di '{output_dir}'. Total keseluruhan: {len(all_synthetic_data)} data.")


if __name__ == "__main__":
    # Menjalankan skrip dengan memasukkan konstanta global sebagai argumen
    run_generation(prompt_file=PROMPT_PATH, output_dir=RAW_OUTPUT_DIR)
