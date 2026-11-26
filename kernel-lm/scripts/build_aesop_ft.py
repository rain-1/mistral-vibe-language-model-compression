#!/usr/bin/env python3
"""Build outline->fable fine-tune data from Aesop's Fables (Vernon Jones 1912,
Project Gutenberg #11339 via the PG-19 GCS mirror — public domain).

Parses the story section (titles in CAPS separated by blank lines), builds
extractive bullet outlines (first / middle / last sentences), and writes:
  data/ft3/pairs.jsonl        {title, outline, story, split}
  data/ft3_tokens/{train,val}.bin (+offsets)
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/pg19/raw/train/11339.txt"


def parse_fables(text: str):
    # story section starts at the LAST "AESOP'S FABLES" header
    i = text.rfind("AESOP'S FABLES")
    tail = text[i:]
    # split on title lines: 2+ newlines, ALL-CAPS title, 2+ newlines
    parts = re.split(r"\n{3,}([A-Z][A-Z0-9 ,'&\-!?]{3,60})\n{2,}", tail)
    fables = []
    # parts = [preamble, title1, body1, title2, body2, ...]
    for j in range(1, len(parts) - 1, 2):
        title, body = parts[j].strip(), parts[j + 1]
        body = re.sub(r"\n{2,}", "\n\n", body).strip()
        body = re.sub(r"\s+", " ", body.replace("\n", " "))
        if 150 <= len(body) <= 1600:
            fables.append((title.title(), body))
    return fables


def sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def make_outline(sents, k=5):
    if len(sents) < 3:
        return None
    k = min(k, len(sents))
    idx = sorted({round(i * (len(sents) - 1) / (k - 1)) for i in range(k)})
    bullets = [sents[i] if len(sents[i]) <= 160 else sents[i][:157] + "..." for i in idx]
    return "\n".join(f"- {b}" for b in bullets)


def main():
    text = SRC.read_text(encoding="utf-8", errors="replace")
    fables = parse_fables(text)
    print(f"parsed {len(fables)} fables")
    pairs = []
    for title, body in fables:
        o = make_outline(sentences(body))
        if o:
            pairs.append({"title": title, "outline": o, "story": body})
    rng = random.Random(42)
    rng.shuffle(pairs)
    n_val = 40
    for i, p in enumerate(pairs):
        p["split"] = "val" if i < n_val else "train"
    out = ROOT / "data/ft3"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs ({n_val} val)")

    tok = Tokenizer.from_file(str(ROOT / "data/ao3/tokenizer-16k.json"))
    sep = tok.token_to_id("<|file|>")
    out_dir = ROOT / "data/ft3_tokens"
    out_dir.mkdir(exist_ok=True)
    for split in ("train", "val"):
        recs = [p for p in pairs if p["split"] == split]
        texts = [f"OUTLINE:\n{p['outline']}\n\nSTORY:\n{p['story']}" for p in recs]
        encs = tok.encode_batch(texts)
        ids, offsets = [], []
        for e in encs:
            if ids:
                ids.append(sep)
            s = len(ids)
            ids.extend(e.ids)
            offsets.append([s, len(ids)])
        np.array(ids, dtype=np.uint16).tofile(out_dir / f"{split}.bin")
        json.dump({"sep_id": sep, "offsets": offsets}, open(out_dir / f"{split}.offsets.json", "w"))
        print(f"{split}: {len(recs)} files, {len(ids)} tokens")


if __name__ == "__main__":
    main()
