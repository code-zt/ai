import json
from pathlib import Path
from typing import Dict

import torch

from .model import OpenAPIGenerator
from .tokenizer import APITokenizer


def load_checkpoint(path: str, device: str):
    state = torch.load(path, map_location=device)
    vocab_size = state.get("vocab_size") or len(state.get("tokenizer_vocab", {}))
    model = OpenAPIGenerator(vocab_size=vocab_size).to(device)
    model.load_state_dict(state["model_state"])
    tokenizer = APITokenizer()
    tokenizer.vocab = state.get("tokenizer_vocab", tokenizer.vocab)
    tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer.frozen = True
    return model, tokenizer


def greedy_generate(model: OpenAPIGenerator, tokenizer: APITokenizer, prompt_spec: Dict, max_len: int = 128):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.tokenize(prompt_spec).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    generated = input_ids[:, :1]  # [BOS]
    for _ in range(max_len):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, tgt_ids=generated)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        if int(next_id.item()) == tokenizer.special_tokens["[EOS]"]:
            break

    minimal = tokenizer.detokenize(generated[0].tolist())
    # fallback: if the model produced an empty or root path, keep prompt path
    gen_path = minimal.get("path") or "/"
    if gen_path.strip() in ("", "/") and prompt_spec.get("path"):
        gen_path = prompt_spec["path"]
    gen_method = (minimal.get("method") or prompt_spec.get("method") or "GET").lower()
    # Wrap into minimal OpenAPI 3.0 document
    openapi = {
        "openapi": "3.0.0",
        "info": {"title": "Generated API", "version": "1.0.0"},
        "paths": {
            gen_path: {
                gen_method: {
                    "parameters": [
                        {
                            "name": p.get("name", "param"),
                            "in": p.get("in", "query"),
                            "required": p.get("required", False),
                            "schema": {"type": p.get("type", "string")},
                        }
                        for p in minimal.get("parameters", [])
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    }
    return openapi


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "./artifacts/checkpoint.pt"
    model, tokenizer = load_checkpoint(ckpt, device)
    prompt = {"path": "/api/users", "method": "GET", "parameters": []}
    spec = greedy_generate(model, tokenizer, prompt)
    print(json.dumps(spec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


