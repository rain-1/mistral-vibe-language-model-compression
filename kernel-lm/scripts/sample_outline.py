#!/usr/bin/env python3
"""Generate stories from bullet-list outlines with a fine-tuned checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent

DEMO_OUTLINES = [
    "- A shy librarian discovers a book that writes itself\n- She tries to change the ending but the book resists\n- The story's hero turns out to be her own reflection\n- She must decide whether to close the book forever\n- In the end she writes her own final chapter",
    "- Two rival food trucks park on the same corner\n- The owners used to be best friends\n- A food critic visits on the busiest day of the year\n- Both trucks run out of their signature dish\n- They team up and invent a new menu together",
    "- An old lighthouse keeper hears a song from the fog\n- Every night the same melody plays at midnight\n- He rows out to find a shipwrecked singer\n- The lighthouse light has been guiding her home for years\n- He fixes the light and the song finally stops",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/ao3/tokenizer-16k.json"))
    ap.add_argument("--pairs", default=str(ROOT / "data/ft/pairs.jsonl"),
                    help="if set, also continue outlines from held-out val pairs")
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default=str(ROOT / "results/samples/outline_ft.txt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)

    prompts = [f"OUTLINE:\n{o}\n\nSTORY:\n" for o in DEMO_OUTLINES]
    if args.pairs:
        import random
        val = [json.loads(l) for l in open(args.pairs) if json.loads(l)["split"] == "val"]
        rng = random.Random(args.seed)
        rng.shuffle(val)
        for p in val[: args.n_val]:
            prompts.append(f"OUTLINE:\n{p['outline']}\n\nSTORY:\n")
            prompts[-1] += ""  # story hidden; we generate it

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, p in enumerate(prompts):
            ids = tok.encode(p).ids
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature,
                               top_k=args.top_k, repetition_penalty=args.repetition_penalty)
            text = tok.decode(y[0, len(ids):].tolist(), skip_special_tokens=True)
            f.write(f"===== outline sample {i} =====\n{p}\n--- generated story ---\n{text}\n\n")
            print(f"sample {i} done", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
