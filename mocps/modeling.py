from __future__ import annotations

from pathlib import Path
import torch
from transformers import AutoTokenizer, BigBirdForMaskedLM

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_tokenizer(base_model: str | Path):
    return AutoTokenizer.from_pretrained(str(base_model), local_files_only=True)

def load_model(base_model: str | Path, device: torch.device, checkpoint: str | Path | None = None, attention_type: str = 'original_full'):
    model = BigBirdForMaskedLM.from_pretrained(str(base_model), local_files_only=True)
    if checkpoint is not None:
        state_dict = torch.load(str(checkpoint), map_location=device)
        model.load_state_dict(state_dict)
    model.bert.set_attention_type(attention_type)
    model.to(device)
    model.eval()
    return model

def load_tokenizer_and_model(base_model: str | Path, device: torch.device | None = None, checkpoint: str | Path | None = None, attention_type: str = 'original_full'):
    device = device or get_device()
    tokenizer = load_tokenizer(base_model)
    model = load_model(base_model, device, checkpoint, attention_type)
    return tokenizer, model, device
