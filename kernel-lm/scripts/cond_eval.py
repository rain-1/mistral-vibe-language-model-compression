#!/usr/bin/env python3
"""Conditional evaluation for the outline->story fine-tune.

For each held-out pair (outline, story), compute:
  - unconditional: NLL of the story tokens alone
  - conditional:   NLL of the story tokens given the "OUTLINE:...\\n\\nSTORY:\\n" prefix
Reports both as BPB (story bytes). Also runs on AO3 test stories with
extractive outlines (built the same way as the FT pairs, from held-out AO3).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Transformer, ModelConfig
from build_ft_pairs import make_outline, sentences

ROOT = Path(__file__).resolve().parent.parent


def nll_of_region(model, ids: list[int], region_start: int, ctx: int, device: str) -> tuple[float, int]:
    """NLL over tokens [region_start, len) using non-overlapping ctx windows."""
    total_nll, total_tok = 0.0, 0
    for s in range(0, len(ids) - 1, ctx):
        w = ids[s:s + ctx + 1]
        if len(w) < 2:
            continue
        x = torch.tensor([w[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([w[1:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x)
        logp = torch.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)[0]
        # token position p in window predicts ids[s+p+1]; count those >= region_start
        for p in range(len(w) - 1):
            if s + p + 1 >= region_start:
                total_nll += float(nll[p])
                total_tok += 1
    return total_nll, total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/ao3/tokenizer-16k.json"))
    ap.add_argument("--pairs", default=str(ROOT / "data/ft/pairs.jsonl"))
    ap.add_argument("--ao3-test", type=int, default=150, help="AO3 test stories with extractive outlines")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)

    pairs = [json.loads(l) for l in open(args.pairs) if json.loads(l)["split"] == "val"]
    pairs = pairs[: args.max_files]
    # add AO3 test stories with extractive outlines
    ao3 = [json.loads(l) for l in open(ROOT / "data/ao3/story_manifest.jsonl")
           if json.loads(l)["split"] == "test"]
    rng = __import__("random").Random(7)
    rng.shuffle(ao3)
    n_added = 0
    for r in ao3:
        if n_added >= args.ao3_test:
            break
        story = (ROOT / "data/ao3/stories" / r["path"]).read_text(encoding="utf-8", errors="surrogateescape")
        if not (300 <= len(story.encode()) <= 4000):
            continue
        o = make_outline(sentences(story))
        if o is None:
            continue
        pairs.append({"outline": o, "story": story, "id": "ao3-" + r["story_id"]})
        n_added += 1
    print(f"evaluating {len(pairs)} pairs ({n_added} from AO3 test)")

    res = {"uncond": [0.0, 0, 0], "cond": [0.0, 0, 0]}  # nll, tokens, bytes
    for i, p in enumerate(pairs):
        prefix = f"OUTLINE:\n{p['outline']}\n\nSTORY:\n"
        full_ids = tok.encode(prefix + p["story"]).ids
        story_ids = tok.encode(p["story"]).ids
        prefix_ids = tok.encode(prefix).ids
        region_start = len(prefix_ids) - 1  # approx boundary
        nll_c, tok_c = nll_of_region(model, full_ids, region_start, mc.context_length, device)
        nll_u, tok_u = nll_of_region(model, story_ids, 1, mc.context_length, device)
        nb = len(p["story"].encode("utf-8", errors="surrogateescape"))
        res["cond"][0] += nll_c; res["cond"][1] += tok_c; res["cond"][2] += nb
        res["uncond"][0] += nll_u; res["uncond"][1] += tok_u; res["uncond"][2] += nb
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pairs)}", flush=True)

    out = {}
    for k in ("uncond", "cond"):
        nll, toks, nb = res[k]
        ce = nll / max(toks, 1)
        out[k] = {"nll_nats": nll, "tokens": toks, "bytes": nb,
                  "ce_nats": ce, "ppl": math.exp(ce),
                  "bits_per_token": ce / math.log(2),
                  "bits_per_byte": nll / (math.log(2) * nb) if nb else None}
        print(f"{k}: ce={ce:.4f} ppl={math.exp(ce):.2f} bpt={ce/math.log(2):.3f} BPB={out[k]['bits_per_byte']:.4f}")
    out["outline_gain_BPB"] = out["uncond"]["bits_per_byte"] - out["cond"]["bits_per_byte"]
    print(f"outline conditioning saves {out['outline_gain_BPB']:.4f} BPB "
          f"({out['uncond']['bits_per_byte']/out['cond']['bits_per_byte']:.2f}x)")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
