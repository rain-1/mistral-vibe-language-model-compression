#!/usr/bin/env python3
"""Generate C code from a checkpoint given a prompt file or string."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default=None, help="text prompt (default: read stdin or use --prompt-file)")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)

    if args.prompt_file:
        text = Path(args.prompt_file).read_text()
    elif args.prompt is not None:
        text = args.prompt
    else:
        text = sys.stdin.read() if False else "/*\n * "
    ids = tok.encode(text).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0, len(ids):].tolist()))


if __name__ == "__main__":
    main()
