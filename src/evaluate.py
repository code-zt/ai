import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import f1_score

from .model import OpenAPIGenerator
from .tokenizer import APITokenizer


def load_checkpoint(path: str, device: str, retries: int = 5, delay: float = 2.0):
    last_err = None
    for _ in range(retries):
        try:
            state = torch.load(path, map_location=device)
            break
        except Exception as e:
            last_err = e
            time.sleep(delay)
    else:
        raise last_err
    vocab_size = state.get("vocab_size") or len(state.get("tokenizer_vocab", {}))
    model = OpenAPIGenerator(vocab_size=vocab_size).to(device)
    model.load_state_dict(state["model_state"])
    tokenizer = APITokenizer()
    tokenizer.vocab = state.get("tokenizer_vocab", tokenizer.vocab)
    tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer.frozen = True
    return model, tokenizer


def tokens_f1(pred: List[int], true: List[int]) -> float:
    # micro F1 over vocabulary ids excluding PAD
    y_true, y_pred = [], []
    for t, p in zip(true, pred):
        if t == 0:  # PAD
            continue
        y_true.append(t)
        y_pred.append(p)
    if not y_true:
        return 0.0
    labels = sorted(set(y_true) | set(y_pred))
    return f1_score(y_true, y_pred, labels=labels, average='micro')


def evaluate_dataset(data_path: str = "./data/synthetic.json", ckpt: str = "./artifacts/checkpoint.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(ckpt, device)
    with open(data_path) as f:
        data = json.load(f)

    f1s = []
    for spec in data[:200]:  # quick eval subset
        input_ids = tokenizer.tokenize(spec).unsqueeze(0).to(device)
        attn = torch.ones_like(input_ids).to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn, tgt_ids=input_ids)
        pred = logits.argmax(dim=-1)[0].detach().cpu().tolist()
        true = input_ids[0].detach().cpu().tolist()[: len(pred)]
        f1s.append(tokens_f1(pred, true))
    print(f"Token F1 (subset): {sum(f1s)/len(f1s):.3f}")


if __name__ == "__main__":
    evaluate_dataset()


