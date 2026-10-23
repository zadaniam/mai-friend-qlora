import json
import os
from unittest.mock import MagicMock, mock_open, patch

from src.data.process_dataset import merge_and_format_datasets


@patch("src.data.process_dataset.os.path.exists")
@patch("src.data.process_dataset.os.makedirs")
def test_merge_and_format_datasets_success(mock_makedirs, mock_exists, mock_sft_data_list, capsys):
    def side_effect_exists(path):
        return "dataset_sft_batch_1.json" in path or "dataset_sft_batch_2.json" in path
    mock_exists.side_effect = side_effect_exists

    batch_1_content = [mock_sft_data_list]
    batch_2_content = [mock_sft_data_list]

    mock_files = {
        os.path.join("data", "raw", "dataset_sft_batch_1.json"): json.dumps(batch_1_content),
        os.path.join("data", "raw", "dataset_sft_batch_2.json"): json.dumps(batch_2_content),
    }

    def side_effect_open(path, mode="r", *args, **kwargs):
        if mode == "r":
            return mock_open(read_data=mock_files.get(path, "[]"))()
        return mock_open()()

    m_open = MagicMock()
    m_open.side_effect = side_effect_open

    with patch("builtins.open", m_open):
        merge_and_format_datasets(
            input_dir="data/raw", 
            output_file="data/processed/dataset_sft_final_ready.jsonl", 
            num_batches=2
        )

    mock_makedirs.assert_called_once_with("data/processed", exist_ok=True)
    captured = capsys.readouterr()
    output_text = captured.out

    assert "✅ Berhasil memuat 1 data dari dataset_sft_batch_1.json" in output_text
    assert "✅ Berhasil memuat 1 data dari dataset_sft_batch_2.json" in output_text
    assert "🔄 Selesai mengacak total 2 data percakapan." in output_text
    assert "🎉 Selesai! File final siap pakai disimpan di: data/processed/dataset_sft_final_ready.jsonl" in output_text

@patch("src.data.process_dataset.os.path.exists", return_value=False)
def test_merge_and_format_datasets_no_data_found(mock_exists, capsys):
    merge_and_format_datasets(input_dir="data/raw", output_file="data/processed/final.jsonl", num_batches=3)
    captured = capsys.readouterr()
    output_text = captured.out
    assert "⚠️ File tidak ditemukan:" in output_text
    assert "❌ Tidak ada data yang berhasil dimuat. Proses penggabungan dibatalkan." in output_text

@patch("src.data.process_dataset.os.path.exists", return_value=True)
def test_merge_and_format_datasets_corrupted_json(mock_exists, capsys):
    corrupted_json_content = '[{"messages": []}' 
    with patch("builtins.open", mock_open(read_data=corrupted_json_content)):
        merge_and_format_datasets(input_dir="data/raw", output_file="data/processed/final.jsonl", num_batches=1)
    captured = capsys.readouterr()
    output_text = captured.out
    assert "❌ Gagal membaca file: dataset_sft_batch_1.json. Periksa sintaks JSON" in output_text
    assert "❌ Tidak ada data yang berhasil dimuat. Proses penggabungan dibatalkan." in output_text
