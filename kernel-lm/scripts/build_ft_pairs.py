#!/usr/bin/env python3
"""Build outline->story fine-tuning pairs from Fimfiction stories.

Outlines are EXTRACTIVE skeletons: up to 6 evenly-spaced sentences of the story
as bullets. This teaches the model the OUTLINE -> STORY format and content
grounding without relying on its (weak) instruction following.

Pairs:   data/ft/pairs.jsonl          {id, title, outline, story, split}
Tokens:  data/ft_tokens/{train,val}.bin (+ .offsets.json), <|file|> separated,
         each record = "OUTLINE:\\n- ...\\n\\nSTORY:\\n<story>"
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent


def sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [s.strip() for s in sents if len(s.strip()) > 20]


def make_outline(sents: list[str], k: int = 5) -> str | None:
    if len(sents) < 3:
        return None
    k = min(k, len(sents))
    idx = sorted({round(i * (len(sents) - 1) / (k - 1)) for i in range(k)})
    bullets = []
    for i in idx:
        b = sents[i]
        if len(b) > 160:
            b = b[:157].rstrip() + "..."
        bullets.append(b)
    if len(bullets) < 3:
        return None
    return "\n".join(f"- {b}" for b in bullets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--min-bytes", type=int, default=500)
    ap.add_argument("--max-bytes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    eval_paths = set()
    meta_p = ROOT / "data/story_eval_tokens/fimfiction.meta.json"
    if meta_p.exists():
        eval_paths = {m["path"] for m in json.load(open(meta_p))["files"]}

    cands = []
    for l in open(ROOT / "data/fimfiction/story_manifest.jsonl"):
        r = json.loads(l)
        if r["path"] not in eval_paths and args.min_bytes <= r["bytes"] <= args.max_bytes:
            cands.append(r)
    rng = random.Random(args.seed)
    rng.shuffle(cands)
    print(f"{len(cands)} candidate stories")

    pairs = []
    for r in cands:
        if len(pairs) >= args.n:
            break
        story = (ROOT / "data/fimfiction/stories" / r["path"]).read_text(
            encoding="utf-8", errors="surrogateescape")
        outline = make_outline(sentences(story))
        if outline is None:
            continue
        pairs.append({"id": r["story_id"], "title": r.get("title", ""),
                      "outline": outline, "story": story})
    rng.shuffle(pairs)
    n_val = 300
    for i, p in enumerate(pairs):
        p["split"] = "val" if i < n_val else "train"
    with open(ROOT / "data/ft/pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs ({n_val} val) -> data/ft/pairs.jsonl")

    # ---- tokenize to memmaps
    tok = Tokenizer.from_file(str(ROOT / "data/ao3/tokenizer-16k.json"))
    sep = tok.token_to_id("<|file|>")
    out_dir = ROOT / "data/ft_tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        recs = [p for p in pairs if p["split"] == split]
        texts = [f"OUTLINE:\n{p['outline']}\n\nSTORY:\n{p['story']}" for p in recs]
        encs = tok.encode_batch(texts)
        ids_all = []
        offsets = []
        for e in encs:
            if ids_all:
                ids_all.append(sep)
            s = len(ids_all)
            ids_all.extend(e.ids)
            offsets.append([s, len(ids_all)])
        arr = np.array(ids_all, dtype=np.uint16)
        arr.tofile(out_dir / f"{split}.bin")
        json.dump({"sep_id": sep, "offsets": offsets}, open(out_dir / f"{split}.offsets.json", "w"))
        print(f"{split}: {len(recs)} files, {len(arr)} tokens -> {out_dir/(split+'.bin')}")


if __name__ == "__main__":
    main()
