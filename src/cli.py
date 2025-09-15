import argparse
import json
from pathlib import Path

from .extract import extract_endpoints
from .synthesize_data import synthesize
from .train import train_model, split_dataset
from .generate import load_checkpoint, greedy_generate


def main():
    parser = argparse.ArgumentParser(description="OpenAPI Generator CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_ext = sub.add_parser("extract")
    p_ext.add_argument("project")
    p_ext.add_argument("framework", choices=["FASTAPI", "FLASK", "DJANGO", "GO"]) 
    p_ext.add_argument("--out", default="./data/extracted.json")

    p_synth = sub.add_parser("synth")
    p_synth.add_argument("--n", type=int, default=600)
    p_synth.add_argument("--out", default="./data/synthetic.json")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", default="./data/synthetic.json")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--data", default="./data/synthetic.json")

    p_gen = sub.add_parser("gen")
    p_gen.add_argument("--ckpt", default="./artifacts/checkpoint.pt")
    p_gen.add_argument("--path", default="/api/users")
    p_gen.add_argument("--method", default="GET")

    args = parser.parse_args()
    Path("./data").mkdir(parents=True, exist_ok=True)

    if args.cmd == "extract":
        specs = extract_endpoints(args.project, args.framework)
        with open(args.out, "w") as f:
            json.dump(specs, f, ensure_ascii=False, indent=2)
        print(f"Extracted {len(specs)} endpoints -> {args.out}")
        return

    if args.cmd == "synth":
        data = synthesize(args.n)
        with open(args.out, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(data)} samples -> {args.out}")
        return

    if args.cmd == "train":
        with open(args.data) as f:
            data = json.load(f)
        train, val = split_dataset(data)
        ckpt = train_model(train, val)
        print(f"Saved checkpoint -> {ckpt}")
        return

    if args.cmd == "eval":
        from .evaluate import evaluate_dataset
        evaluate_dataset(args.data)
        return

    if args.cmd == "gen":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_checkpoint(args.ckpt, device)
        prompt = {"path": args.path, "method": args.method, "parameters": []}
        spec = greedy_generate(model, tokenizer, prompt)
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()


