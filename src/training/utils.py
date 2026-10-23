# src/utils.py
import os
import torch
from transformers import BitsAndBytesConfig

def format_chat_template(batch, tokenizer):
    formatted_texts = []
    for messages in batch["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_texts.append(text)
    return {"text": formatted_texts}

class BoundMethodWrapper:
    def __init__(self, target_callable):
        self.target_callable = target_callable
        
    def __call__(self, *args, **kwargs):
        return self.target_callable(*args, **kwargs)
        
    @property
    def __func__(self):
        return getattr(self.target_callable, "func", self.target_callable)

def get_bnb_config():
    # Jika dipaksa berjalan di CPU (lokal atau GitHub CI), matikan kuantisasi GPU
    if os.environ.get("FORCE_CPU") == "true":
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
