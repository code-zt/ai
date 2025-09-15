import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .tokenizer import APITokenizer
from .dataset import OpenAPIDataset, collate_batch
from .model import OpenAPIGenerator
from .losses import SchemaLoss


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.98)
    warmup_ratio: float = 0.03
    patience: int = 8
    max_len: int = 256
    seed: int = 42
    freeze_bert_epochs: int = 0


def split_dataset(data: List[Dict], val_ratio: float = 0.2):
    random.shuffle(data)
    n_val = max(1, int(len(data) * val_ratio))
    return data[n_val:], data[:n_val]


def build_schema() -> Dict:
    # Minimal schema for our generated subset
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "method": {"type": "string"},
            "parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "in": {"type": "string"},
                        "required": {"type": "boolean"},
                    },
                    "required": ["name", "type"],
                },
            },
        },
        "required": ["path", "method", "parameters"],
    }


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(train_data: List[Dict], val_data: List[Dict], device: str = None, output_dir: str = "./artifacts"):
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = APITokenizer()
    train_ds = OpenAPIDataset(train_data, tokenizer, max_len=cfg.max_len)
    # Freeze vocab after building on train set to avoid OOV index spikes on GPU
    tokenizer.frozen = True
    val_ds = OpenAPIDataset(val_data, tokenizer, max_len=cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)

    model = OpenAPIGenerator(vocab_size=len(tokenizer.vocab)).to(device)
    criterion = SchemaLoss(build_schema(), tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)

    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)

    best_val = float("inf")
    wait = 0

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        # no freeze/unfreeze in token-Transformer variant

        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target_ids"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, tgt_ids=targets)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["target_ids"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask, tgt_ids=targets)
                loss = criterion(logits, targets)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch:02d} | train_loss={epoch_loss/len(train_loader):.4f} | val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save({
                "model_state": model.state_dict(),
                "tokenizer_vocab": tokenizer.vocab,
                "vocab_size": len(tokenizer.vocab),
            }, str(Path(output_dir) / "checkpoint.pt"))
        else:
            wait += 1
            if wait >= cfg.patience:
                print("Early stopping.")
                break

    return str(Path(output_dir) / "checkpoint.pt")


def main():
    data_path = Path("./data/synthetic.json")
    with data_path.open() as f:
        data = json.load(f)
    train, val = split_dataset(data)
    ckpt = train_model(train, val)
    print(f"Saved checkpoint to {ckpt}")


if __name__ == "__main__":
    main()


