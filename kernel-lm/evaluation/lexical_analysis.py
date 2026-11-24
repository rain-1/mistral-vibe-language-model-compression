#!/usr/bin/env python3
"""Break down predictive loss by source-code lexical category (spec section 10).

For each evaluated file we classify every *predicted* token (decoded string):
  whitespace, comment, string, char, number, preprocessor, keyword, type,
  identifier_seen (occurred earlier in the same file), identifier_novel,
  operator/punctuation.

Also reports identifier NLL vs distance to previous occurrence in the file.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent

C_KEYWORDS = set("""auto break case char const continue default do double else enum extern
float for goto if inline int long register restrict return short signed sizeof static
struct switch typedef union unsigned void volatile while _Alignas _Alignof _Atomic
_Bool _Complex _Generic _Imaginary _Noreturn _Static_assert _Thread_local""".split())
C_TYPES = set("""int char void long short float double signed unsigned _Bool struct union enum
size_t ssize_t u8 u16 u32 u64 s8 s16 s32 s64 __u8 __u16 __u32 __u64 __s8 __s16 __s32 __s64
bool int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t""".split())

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NUMBER = re.compile(r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|\d+\.?\d*([eE][+-]?\d+)?[uUlLfF]*)$")


def classify(tok_str: str, line_start: bool, seen: set) -> str:
    if tok_str == "":
        return "empty"
    if tok_str.strip() == "":
        return "whitespace"
    if tok_str.startswith("//") or tok_str.startswith("/*") or tok_str.startswith("*"):
        return "comment"
    if tok_str.startswith('"'):
        return "string"
    if tok_str.startswith("'"):
        return "char"
    if line_start and tok_str.startswith("#"):
        return "preprocessor"
    if NUMBER.match(tok_str):
        return "number"
    if tok_str in C_TYPES:
        return "type"
    if tok_str in C_KEYWORDS:
        return "keyword"
    if IDENT.match(tok_str):
        return "identifier_seen" if tok_str in seen else "identifier_novel"
    return "operator"


@torch.no_grad()
def analyze(model, tok: Tokenizer, bin_path: Path, offsets: list, ctx: int,
            device: str, max_files: int | None = None) -> dict:
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    model.eval()
    cat_nll = defaultdict(float)
    cat_cnt = defaultdict(int)
    id_dist_nll = defaultdict(list)  # distance bucket -> [nll...]
    files = offsets[:max_files] if max_files else offsets
    for fi, (s, e) in enumerate(files):
        ids = np.asarray(data[s:e], dtype=np.int64)
        if len(ids) < 2:
            continue
        # token strings for the whole file
        strs = [tok.decode([int(t)], skip_special_tokens=False) for t in ids]
        seen: set = set()
        for wstart in range(0, len(ids) - 1, ctx):
            wend = min(wstart + ctx + 1, len(ids))
            seg = ids[wstart:wend]
            x = torch.tensor([seg[:-1]], dtype=torch.long, device=device)
            logits, _ = model(x)
            logp = torch.log_softmax(logits[0].float(), dim=-1).cpu().numpy()
            for i in range(len(seg) - 1):
                t = int(seg[i + 1])
                nll = -float(logp[i, t])
                # line-start tracking: a token is at line start if preceding
                # text since last newline contains only whitespace
                prefix = "".join(strs[max(0, i + wstart - 20):i + wstart + 1])
                line_start = prefix.split("\n")[-1].strip() == ""
                cat = classify(strs[i + wstart + 1], line_start, seen)
                cat_nll[cat] += nll
                cat_cnt[cat] += 1
                if cat in ("identifier_seen", "identifier_novel"):
                    tok_str = strs[i + wstart + 1]
                    # distance to previous occurrence
                    dist = 0
                    for j in range(i + wstart - 1, -1, -1):
                        if strs[j] == tok_str:
                            dist = i + wstart - j
                            break
                    else:
                        dist = -1
                    bucket = "0" if dist == 0 else ("1-8" if 0 < dist <= 8 else
                              "9-64" if dist <= 64 else "65-512" if dist <= 512 else "512+" if dist > 0 else "novel")
                    id_dist_nll[bucket].append(nll)
        for st in strs:
            if IDENT.match(st):
                seen.add(st)
    out = {"categories": {}, "identifier_distance": {}}
    for c in cat_nll:
        out["categories"][c] = {"count": cat_cnt[c], "nll_nats": cat_nll[c],
                                "mean_nll": cat_nll[c] / cat_cnt[c],
                                "bits": cat_nll[c] / math.log(2) / cat_cnt[c]}
    for b, v in id_dist_nll.items():
        out["identifier_distance"][b] = {"count": len(v), "mean_nll": float(np.mean(v))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", default=str(ROOT / "data/tokens/val.bin"))
    ap.add_argument("--offsets", default=str(ROOT / "data/tokens/val.offsets.json"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tokenizer.json"))
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    tok = Tokenizer.from_file(args.tokenizer)
    with open(args.offsets) as f:
        offsets = json.load(f)["offsets"]
    res = analyze(model, tok, Path(args.corpus), offsets, mc.context_length, device, args.max_files)
    print(json.dumps(res, indent=2, sort_keys=True))
    out = Path(args.out) if args.out else ROOT / "results/eval" / f"lexical_{Path(args.checkpoint).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"checkpoint": str(args.checkpoint), "corpus": args.corpus, **res}, f, indent=2, sort_keys=True)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
