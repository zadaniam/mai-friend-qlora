import json
import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.data.generate_dataset import DatasetResponse, load_prompts, run_generation

# ==============================================================================
# 1. UNIT TEST: FUNGSI `load_prompts` (Menguji Logika File I/O & Parsing Data)
# ==============================================================================

def test_load_prompts_success_with_list_general():
    """Menguji jika file prompt ada dan 'prompt_general' berbentuk list."""
    mock_data = {
        "prompt_general": ["Baris 1", "Baris 2"],
        "prompt_batch_spesifik": ["Topik A", "Topik B"]
    }
    mock_json_string = json.dumps(mock_data)

    # Patch 'os.path.exists' agar selalu mengembalikan True (seolah file ada)
    # Patch 'open' menggunakan mock_open untuk mensimulasikan pembacaan file
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=mock_json_string)):
        
        general, spesifik = load_prompts("data/fake_path.json")
        
        # Memastikan list digabungkan dengan newline (\n)
        assert general == "Baris 1\nBaris 2"
        assert len(spesifik) == 2
        assert spesifik[0] == "Topik A"


def test_load_prompts_file_not_found():
    """Menguji apakah fungsi melempar FileNotFoundError jika file tidak eksis."""
    with patch("os.path.exists", return_value=False), pytest.raises(FileNotFoundError):
        load_prompts("data/non_existent_file.json")


# ==============================================================================
# 2. UNIT TEST: FUNGSI UTAMA `run_generation` (Menguji Integrasi Client & API Mock)
# ==============================================================================

@patch("src.data.generate_dataset.load_dotenv")
@patch("src.data.generate_dataset.os.makedirs")
@patch("src.data.generate_dataset.time.sleep")  
@patch("src.data.generate_dataset.genai.Client")
def test_run_generation_success(mock_genai_client, mock_sleep, mock_makedirs, mock_load_dotenv):
    """Menguji alur sukses run_generation dari memuat prompt hingga menulis file output."""
    
    # Simulasikan output dari load_prompts agar tidak membaca file asli
    mock_prompt_general = "Instruksi Umum AI"
    mock_batch_spesifik = ["Topik Khusus: Teknologi"]
    
    # Mock respons teks JSON tiruan dari API Gemini yang sesuai dengan DatasetResponse
    mock_api_response_data = {
        "conversations": [
            {
                "messages": [
                    {"role": "user", "content": "Halo"},
                    {"role": "model", "content": "Hai, ada yang bisa dibantu?"}
                ]
            }
        ]
    }
    
    # Setup Mock untuk SDK google-genai
    mock_response_obj = MagicMock()
    mock_response_obj.text = json.dumps(mock_api_response_data)
    
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response_obj
    mock_genai_client.return_value = mock_client_instance

    # Mock proses penulisan berkas json.dump agar tidak benar-benar menulis ke disk
    m_open = mock_open()

    with patch("src.data.generate_dataset.load_prompts", return_value=(mock_prompt_general, mock_batch_spesifik)), \
         patch("builtins.open", m_open):
        
        # Jalankan fungsi utama
        run_generation(prompt_file="dummy_prompt.json", output_dir="dummy_raws")

        # --- VERIFIKASI ---
        # 1. Pastikan folder output dibuat
        mock_makedirs.assert_called_once_with("dummy_raws", exist_ok=True)
        
        # 2. Pastikan API Gemini dipanggil dengan parameter yang benar (Model & Skema Pydantic)
        mock_client_instance.models.generate_content.assert_called_once()
        _called_args, called_kwargs = mock_client_instance.models.generate_content.call_args
        
        assert called_kwargs['model'] == 'gemini-3.5-flash-lite'
        assert called_kwargs['config'].response_schema == DatasetResponse
        assert called_kwargs['config'].response_mime_type == "application/json"

        # 3. Pastikan fungsi menulis file dataset hasil generasi
        m_open.assert_called_with(os.path.join("dummy_raws", "dataset_sft_batch_1.json"), "w", encoding="utf-8")
