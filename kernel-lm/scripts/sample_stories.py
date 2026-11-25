#!/usr/bin/env python3
"""Generate a batch of free prose samples from a story checkpoint."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent

PROMPTS = [
    "She opened the door and found ",
    "The letter began, \"My dearest ",
    "It was the best of times, it was ",
    "In a small village by the sea, ",
    "He had not seen her in ten years, ",
    "The machine hummed to life as ",
    "\"I never thought I'd see you again,\" she said. \"",
    "Chapter 1\n\n",
    "The stars were unusually bright that night. ",
    "Once upon a time, in a kingdom far away, ",
    "The detective examined the magnifying glass and ",
    "Dear diary, today something strange happened: ",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--real-prefixes", type=int, default=8,
                    help="add N prompts = first 256 tokens of random story val files")
    ap.add_argument("--val-tokens", default=str(ROOT / "data/ao3/story_tokens/val.bin"))
    ap.add_argument("--val-offsets", default=str(ROOT / "data/ao3/story_tokens/val.offsets.json"))
    ap.add_argument("--out", default=str(ROOT / "results/samples/story_batch.txt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)

    prompts = list(PROMPTS)
    if args.real_prefixes:
        rng = random.Random(args.seed)
        data = np.memmap(args.val_tokens, dtype=np.uint16, mode="r")
        with open(args.val_offsets) as f:
            offsets = json.load(f)["offsets"]
        idxs = list(range(len(offsets)))
        rng.shuffle(idxs)
        for i in idxs[: args.real_prefixes]:
            s, e = offsets[i]
            ids = [int(t) for t in data[s:min(s + 256, e)]]
            prompts.append(tok.decode(ids, skip_special_tokens=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, p in enumerate(prompts):
            ids = tok.encode(p).ids
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature,
                               top_k=args.top_k, repetition_penalty=args.repetition_penalty)
            text = tok.decode(y[0].tolist(), skip_special_tokens=True)
            f.write(f"===== sample {i} (temp={args.temperature} top_k={args.top_k} rp={args.repetition_penalty}) =====\n")
            f.write(text + "\n\n")
            print(f"sample {i} done", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
