# tests/unit/test_utils.py
import pytest
from transformers import AutoTokenizer
from src.utils import format_chat_template

def test_format_chat_template():
    # PERUBAHAN DI SINI: Gunakan tokenizer Qwen2/2.5 versi tiny agar serasi
    tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Mock data synthetic mini berformat OpenAI JSONL/ChatML
    mock_batch = {
        "messages": [
            [
                {"role": "user", "content": "Halo model empati"},
                {"role": "assistant", "content": "Halo, saya mendengarkan Anda."}
            ]
        ]
    }
    
    result = format_chat_template(mock_batch, tokenizer)
    
    # Verifikasi struktur output
    assert "text" in result
    assert len(result["text"]) == 1
    assert "Halo model empati" in result["text"][0]
    
    # Opsi Tambahan: Memastikan chat template khas Qwen (<|im_start|>) tersemat dengan benar
    assert "<|im_start|>user" in result["text"][0]
    assert "<|im_start|>assistant" in result["text"][0]
