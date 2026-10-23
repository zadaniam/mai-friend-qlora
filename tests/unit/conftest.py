import json
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_sft_data_list():
    """Fixture global yang mengembalikan struktur data SFT berbentuk list objek."""
    return [
        {"messages": [{"role": "user", "content": "Halo"}, {"role": "assistant", "content": "Hai"}]},
        {"messages": [{"role": "user", "content": "Apa kabar?"}, {"role": "assistant", "content": "Kabar baik!"}]}
    ]

@pytest.fixture
def mock_sft_jsonl_str(mock_sft_data_list):
    """Fixture global yang mengubah list data menjadi format string JSONL."""
    return "\n".join(json.dumps(row) for row in mock_sft_data_list) + "\n"

@pytest.fixture
def generic_mock_tokenizer():
    """Fixture global untuk menyediakan Mock Tokenizer siap pakai untuk Hugging Face."""
    mock_instance = MagicMock()
    # Mengembalikan string chat template tiruan
    mock_instance.apply_chat_template.return_value = "<im_start>user\nHalo<im_end>\n"
    # Mengembalikan 150 token IDs tiruan
    mock_instance.encode.return_value = [i for i in range(150)]
    return mock_instance
