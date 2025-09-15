from typing import Dict

import torch
import torch.nn as nn
from fastjsonschema import compile as compile_schema


def decode_output_greedy(logits: torch.Tensor, tokenizer) -> Dict:
    # logits: [B, T, V] -> take argmax and detokenize first sample
    ids = logits.argmax(dim=-1)[0].detach().cpu().tolist()
    return tokenizer.detokenize(ids)


class SchemaLoss(nn.Module):
    def __init__(self, schema: Dict, tokenizer, ce_weight: float = 0.7, schema_weight: float = 0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.validator = compile_schema(schema)
        self.ce_weight = ce_weight
        self.schema_weight = schema_weight
        self.tokenizer = tokenizer

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, V]; targets: [B, T]
        B, T, V = logits.shape
        loss_ce = self.ce(logits.view(B * T, V), targets.view(B * T))
        try:
            spec = decode_output_greedy(logits, self.tokenizer)
            self.validator(spec)
            loss_valid = logits.new_tensor(0.0)
        except Exception:
            loss_valid = logits.new_tensor(1.0)
        return self.ce_weight * loss_ce + self.schema_weight * loss_valid


