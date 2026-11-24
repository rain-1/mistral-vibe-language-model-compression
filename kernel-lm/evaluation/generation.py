#!/usr/bin/env python3
"""Code generation evaluation (spec section 11), kept lightweight:

For a sample of held-out files, pick random line boundaries, feed the preceding
context, and complete the next line:
  - greedy: argmax decoding until newline
  - sampled: temperature sampling until newline
Metrics: exact line match, token-level accuracy, mean edit distance (normalized),
plus a naive syntactic check (balanced (), {}, [] in the completion).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent


def next_line_completion(model, tok, ids: list[int], cut: int, ctx: int,
                         device: str, max_new: int = 200, temperature: float = 0.0):
    """ids[:cut] is context; complete until a newline token appears."""
    ctx_ids = ids[max(0, cut - ctx):cut]
    x = torch.tensor([ctx_ids], dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(x[:, -ctx:])
            logits = logits[0, -1]
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            else:
                nxt = int(logits.argmax())
            out.append(nxt)
            if tok.decode([nxt], skip_special_tokens=False) == "\n":
                break
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return out


def edit_distance(a: str, b: str) -> int:
    # simple DP, capped
    if abs(len(a) - len(b)) > 200:
        return 999
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def balanced(s: str) -> bool:
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", default=str(ROOT / "data/tokens/val.bin"))
    ap.add_argument("--offsets", default=str(ROOT / "data/tokens/val.offsets.json"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tokenizer.json"))
    ap.add_argument("--max-files", type=int, default=100)
    ap.add_argument("--cuts-per-file", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)
    data = np.memmap(args.corpus, dtype=np.uint16, mode="r")
    with open(args.offsets) as f:
        offsets = json.load(f)["offsets"]
    rng = random.Random(args.seed)

    files = offsets[: args.max_files]
    stats = {"n": 0, "exact": 0, "tok_acc": [], "norm_edit": [], "balanced": 0}
    for s, e in files:
        ids = [int(t) for t in data[s:e]]
        if len(ids) < 100:
            continue
        # find line-boundary token positions (token decodes to a string containing \n)
        for _ in range(args.cuts_per_file):
            cut = rng.randrange(50, len(ids) - 10)
            # advance to just after the next newline for a clean line start
            while cut < len(ids) - 1 and "\n" not in tok.decode([ids[cut]], skip_special_tokens=False):
                cut += 1
            cut += 1
            if cut >= len(ids) - 2:
                continue
            # true next line: tokens until the following newline
            j = cut
            true_ids = []
            while j < len(ids) and "\n" not in tok.decode([ids[j]], skip_special_tokens=False):
                true_ids.append(ids[j])
                j += 1
            if not true_ids or len(true_ids) > 150:
                continue
            pred = next_line_completion(model, tok, ids, cut, mc.context_length,
                                        device, temperature=args.temperature)
            pred_str = tok.decode(pred, skip_special_tokens=False)
            true_str = tok.decode(true_ids, skip_special_tokens=False)
            stats["n"] += 1
            if pred_str.strip() == true_str.strip():
                stats["exact"] += 1
            # token accuracy
            common = min(len(pred), len(true_ids))
            acc = sum(1 for a, b in zip(pred[:common], true_ids[:common]) if a == b) / max(len(true_ids), 1)
            stats["tok_acc"].append(acc)
            stats["norm_edit"].append(edit_distance(pred_str, true_str) / max(len(true_str), 1))
            if balanced(pred_str):
                stats["balanced"] += 1
    res = {
        "checkpoint": str(args.checkpoint), "temperature": args.temperature,
        "n": stats["n"],
        "exact_line_match": stats["exact"] / max(stats["n"], 1),
        "token_accuracy": float(np.mean(stats["tok_acc"])) if stats["tok_acc"] else 0,
        "norm_edit_distance": float(np.mean(stats["norm_edit"])) if stats["norm_edit"] else 0,
        "balanced_ratio": stats["balanced"] / max(stats["n"], 1),
    }
    print(json.dumps(res, indent=2))
    out = Path(args.out) if args.out else ROOT / "results/eval" / f"gen_{Path(args.checkpoint).stem}_t{args.temperature}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
