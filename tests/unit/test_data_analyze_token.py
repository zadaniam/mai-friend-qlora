import json
from unittest.mock import MagicMock, mock_open, patch

from src.data.analyze_token import analyze_dataset_sequence_length

# ==============================================================================
# PENGUJIAN UNIT (Seluruh fixture dimuat dari conftest.py secara otomatis)
# ==============================================================================

@patch("src.data.analyze_token.AutoTokenizer")
def test_analyze_dataset_sequence_length_success(mock_autotokenizer, mock_sft_jsonl_str, generic_mock_tokenizer, capsys):
    """Menguji alur sukses analisis token menggunakan fixture global dari conftest.py."""
    
    # Pasangkan generic_mock_tokenizer global ke AutoTokenizer
    mock_autotokenizer.from_pretrained.return_value = generic_mock_tokenizer

    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=mock_sft_jsonl_str)):
        
        analyze_dataset_sequence_length(data_path="dummy_processed/ready.jsonl", model_id="dummy-model")

        # Verifikasi pemanggilan internal tokenizer
        mock_autotokenizer.from_pretrained.assert_called_once_with("dummy-model", trust_remote_code=True)
        assert generic_mock_tokenizer.apply_chat_template.call_count == 2
        assert generic_mock_tokenizer.encode.call_count == 2

        # Mengambil teks stdout menggunakan capsys secara aman
        captured = capsys.readouterr()
        output_text = captured.out
        
        assert "HASIL ANALISIS DISTRIBUSI TOKEN DATASET SFT" in output_text
        assert "Total Data Percakapan : 2 data" in output_text
        assert "max_seq_length = 256" in output_text


@patch("src.data.analyze_token.AutoTokenizer")
def test_analyze_dataset_sequence_length_file_not_found(mock_autotokenizer, capsys):
    """Menguji apakah skrip berhenti secara aman jika file .jsonl tidak ditemukan."""
    with patch("os.path.exists", return_value=False):
        
        analyze_dataset_sequence_length(data_path="invalid_path/file.jsonl", model_id="dummy-model")
        
        mock_autotokenizer.from_pretrained.assert_not_called()
        
        captured = capsys.readouterr()
        error_output = captured.out
        
        assert "File dataset tidak ditemukan di" in error_output


@patch("src.data.analyze_token.AutoTokenizer")
def test_analyze_dataset_sequence_length_with_corrupted_jsonl(mock_autotokenizer, capsys):
    """Menguji kekokohan skrip saat mendapati baris JSONL yang rusak/salah format."""
    # Baris pertama normal, baris kedua sengaja dirusak
    corrupted_data = json.dumps({"messages": [{"role": "user", "content": "Hi"}]}) + "\n" + '{"messages": [{"role": "user" \n'
    
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.apply_chat_template.return_value = "text"
    mock_tokenizer_instance.encode.return_value = [1, 2, 3]
    mock_autotokenizer.from_pretrained.return_value = mock_tokenizer_instance

    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=corrupted_data)):
        
        analyze_dataset_sequence_length(data_path="dummy_processed/corrupted.jsonl", model_id="dummy-model")
        
        captured = capsys.readouterr()
        output_text = captured.out
        
        assert "Peringatan: Melewati baris 2 karena format salah/rusak." in output_text
        assert "Total Data Percakapan : 1 data" in output_text
