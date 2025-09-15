from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


class OpenAPIDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, max_len: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        spec = self.samples[idx]
        tokens = self.tokenizer.tokenize(spec)
        if tokens.size(0) > self.max_len:
            tokens = tokens[: self.max_len]
            tokens[-1] = self.tokenizer.special_tokens["[EOS]"]
        attn_mask = torch.ones_like(tokens)
        return {
            "input_ids": tokens,
            "attention_mask": attn_mask,
            "target_ids": tokens.clone(),  # teacher-forcing same target for simplicity
            "meta": spec,
        }


def collate_batch(batch: List[Dict]) -> Dict:
    # Pad to max length in batch
    pad_id = batch[0]["input_ids"].new_tensor([0])[0]  # [PAD] assumed 0
    max_len = max(item["input_ids"].size(0) for item in batch)

    def pad(seq: torch.Tensor) -> torch.Tensor:
        if seq.size(0) < max_len:
            pad_size = max_len - seq.size(0)
            return torch.cat([seq, torch.full((pad_size,), pad_id, dtype=seq.dtype)])
        return seq

    input_ids = torch.stack([pad(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([pad(item["attention_mask"]) for item in batch])
    target_ids = torch.stack([pad(item["target_ids"]) for item in batch])
    metas = [item["meta"] for item in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
        "metas": metas,
    }


