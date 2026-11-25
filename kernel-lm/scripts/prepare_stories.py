#!/usr/bin/env python3
"""Build the story-corpus manifest for the PG-19 pack.

Reads the individually-downloaded PG-19 books under data/pg19/raw and writes
data/story_manifest.jsonl (same schema as the kernel manifest) with file-level
splits:

  - train : 4,650 of the 4,800 downloaded pack books
  - val   : 150 carved from the pack (seeded, hash-grouped so exact duplicates
            never straddle splits) + the 50 official PG-19 validation books
  - test  : the 100 official PG-19 test books (never trained on, eval only)

Also writes data/story_splits_summary.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PG19 = DATA / "pg19"
RAW = PG19 / "raw"
REVISION = "deepmind/pg19 @ HF 4d28bd77 (storage.googleapis.com/deepmind-gutenberg)"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    titles = {}
    with open(PG19 / "lists" / "metadata.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                titles[row[0]] = row[1]

    pack = [l.strip() for l in open(PG19 / "pack_train_ids.txt") if l.strip()]
    val_ids = [l.strip() for l in open(PG19 / "val_ids.txt") if l.strip()]
    test_ids = [l.strip() for l in open(PG19 / "test_ids.txt") if l.strip()]

    def rec(book_id: str, sub: str, split: str) -> dict:
        p = RAW / sub / f"{book_id}.txt"
        return {
            "path": f"{sub}/{book_id}.txt",
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
            "repo": "deepmind/pg19",
            "revision": REVISION,
            "split": split,
            "book_id": book_id,
            "title": titles.get(book_id, ""),
            "tag": "pg19",
        }

    # carve 150 pack books into val, grouped by content hash (dedup-safe)
    rng = random.Random(42)
    rng.shuffle(pack)
    n_val = 150
    carve = pack[:n_val]
    train = pack[n_val:]

    records = [rec(b, "train", "train") for b in train]
    records += [rec(b, "train", "val") for b in carve]
    records += [rec(b, "validation", "val") for b in val_ids]
    records += [rec(b, "test", "test") for b in test_ids]

    # exact-duplicate guard: same sha256 must not straddle train/val|test
    by_hash: dict[str, set[str]] = {}
    for r in records:
        by_hash.setdefault(r["sha256"], set()).add(r["split"])
    straddle = {h: s for h, s in by_hash.items() if len(s) > 1}
    if straddle:
        # move the offending val/test records to train-side split of the majority
        print(f"WARNING: {len(straddle)} duplicate-hash groups straddle splits; fixing")
        for r in records:
            if r["sha256"] in straddle and r["split"] != "train":
                r["split"] = "train"

    rng.shuffle(records)
    out = DATA / "story_manifest.jsonl"
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    summary = {"repo": "deepmind/pg19", "revision": REVISION, "splits": {}}
    for split in ("train", "val", "test"):
        rs = [r for r in records if r["split"] == split]
        summary["splits"][split] = {
            "files": len(rs),
            "bytes": sum(r["bytes"] for r in rs),
        }
    with open(DATA / "story_splits_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
