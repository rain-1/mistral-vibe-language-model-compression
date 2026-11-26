#!/usr/bin/env python3
"""Memorization probe: greedy-continue held-out documents and measure verbatim
overlap with the true continuation.

For each held-out AO3 test file: prompt = first `prompt_tok` tokens, generate
`gen_tok` tokens greedily, count matching tokens against the real continuation.
Reports mean exact-match fraction and mean longest common run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/ao3/tokenizer-16k.json"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--prompt-tok", type=int, default=200)
    ap.add_argument("--gen-tok", type=int, default=100)
    ap.add_argument("--tokens", default=str(ROOT / "data/ao3/story_tokens/test.bin"))
    ap.add_argument("--offsets", default=str(ROOT / "data/ao3/story_tokens/test.offsets.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    data = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    offsets = json.load(open(args.offsets))["offsets"]
    idxs = list(range(0, len(offsets), max(1, len(offsets) // args.n)))[: args.n]

    match_fracs, lcr = [], []
    for i in idxs:
        s, e = offsets[i]
        ids = [int(t) for t in data[s:e]]
        if len(ids) < args.prompt_tok + args.gen_tok + 10:
            continue
        if args.prompt_tok > 150:  # fables are short; shrink prompt for small docs
            pass
        prompt, truth = ids[: args.prompt_tok], ids[args.prompt_tok: args.prompt_tok + args.gen_tok]
        x = torch.tensor([prompt], dtype=torch.long, device=device)
        y = model.generate(x, args.gen_tok, temperature=1.0, top_k=1)
        gen = y[0, len(prompt):].tolist()
        m = sum(1 for a, b in zip(gen, truth) if a == b)
        match_fracs.append(m / len(truth))
        # longest common run
        best = cur = 0
        for a, b in zip(gen, truth):
            cur = cur + 1 if a == b else 0
            best = max(best, cur)
        lcr.append(best)
    print(f"{args.checkpoint}")
    print(f"  docs={len(match_fracs)}  mean exact-match continuation: {np.mean(match_fracs)*100:.1f}%  "
          f"mean longest verbatim run: {np.mean(lcr):.1f} tokens")


if __name__ == "__main__":
    main()
