import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from src import config
from src.utils import format_chat_template, BoundMethodWrapper, get_bnb_config

def main():
    print("Memulai pipeline QLoRA SFT...")
    
    # 1. Muat Dataset & Tokenizer
    dataset = load_dataset("json", data_files=config.DATA_PATH, split="train")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Transformasikan dataset dengan Chat Template
    processed_dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    # 3. Muat Model dengan Proteksi CPU/GPU
    bnb_config = get_bnb_config()
    is_cpu = os.environ.get("FORCE_CPU") == "true"
    
    # Jika di CPU, gunakan tiny model agar tidak crash & download cepat
    active_model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5" if is_cpu else config.MODEL_ID
    
    print(f"Sedang memuat model: {active_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        active_model_id,
        quantization_config=bnb_config,
        device_map=None if is_cpu else "auto"
    )
    
    if not is_cpu:
        model.gradient_checkpointing_enable()

    # Terapkan MONKEY PATCH penjinak bug TRL v1.13 + Qwen
    model.forward = BoundMethodWrapper(model.forward)

    # 4. Konfigurasi LoRA Adapter
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. Definisikan Argumen Menggunakan SFTConfig
    training_args = SFTConfig(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=config.GRADIENT_ACC_STEPS,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.EPOCHS,
        logging_steps=1,
        optim="adamw_torch" if is_cpu else "paged_adamw_8bit",
        bf16=not is_cpu,
        fp16=False,
        gradient_checkpointing=not is_cpu,
        report_to="none",
        use_cpu=is_cpu,
        max_length=config.MAX_SEQ_LENGTH,
        packing=False,
        loss_type="nll"
    )

    # 6. Jalankan Inisialisasi & Pelatihan
    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_dataset, 
        peft_config=peft_config,
        args=training_args
    )
    
    print("Mengeksekusi proses training...")
    trainer.train()

    # 7. Simpan Hasil (Hanya jika dijalankan di GPU asli)
    if not is_cpu:
        print("\nTraining selesai! Sedang menyimpan adaptor LoRA...")
        trainer.model.save_pretrained(config.OUTPUT_DIR)
        tokenizer.save_pretrained(config.OUTPUT_DIR)
        print(f"Adaptor berhasil disimpan di: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
