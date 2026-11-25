#!/usr/bin/env python3
"""Unpack the AO3-2020 HF parquet shards (data/ao3/raw/train-*.parquet) into
per-chapter plain-text files + a story manifest.

Layout:
  data/ao3/stories/s<shard>/<storyId>_<idx>.txt   chapter text (cleaned `text`)
  data/ao3/story_manifest.jsonl                   one record per chapter row

Split assignment is STORY-level (all chapters of a storyId share a split),
seeded and grouped by story content hash so duplicate stories never straddle
splits:  train 96% / val 2% / test 2%.

Usage: unpack_ao3.py [--shards 0-11]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ao3"
RAW = DATA / "raw"
OUT = DATA / "stories"
MANIFEST = DATA / "story_manifest.jsonl"
REVISION = "huggingface.co/datasets/ray0rf1re/AO3-2020 config=full (AO3 dump 2020), shards 0-11"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="all")
    args = ap.parse_args()

    shard_files = sorted(RAW.glob("train-*.parquet"))
    if args.shards != "all":
        keep = set()
        for part in args.shards.split(","):
            if "-" in part:
                a, b = part.split("-")
                keep.update(range(int(a), int(b) + 1))
            else:
                keep.add(int(part))
        shard_files = [p for p in shard_files if int(p.stem.split("-")[-1]) in keep]
    print(f"unpacking {len(shard_files)} parquet shards ...")

    records = []
    for shard in shard_files:
        snum = int(shard.stem.split("-")[-1])
        outdir = OUT / f"s{snum:03d}"
        outdir.mkdir(parents=True, exist_ok=True)
        t = pq.read_table(shard, columns=["storyId", "idx", "text"])
        n = 0
        for row in t.to_pylist():
            text = row.get("text") or ""
            if not text.strip():
                continue
            raw = text.encode("utf-8", "surrogateescape")
            sid, idx = str(row["storyId"]), str(row["idx"])
            name = f"{sid}_{idx}.txt"
            (outdir / name).write_bytes(raw)
            records.append({
                "path": f"s{snum:03d}/{name}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "repo": "archiveofourown.org via HF ray0rf1re/AO3-2020",
                "revision": REVISION,
                "split": "train",  # provisional
                "story_id": sid,
                "chapter": idx,
                "shard": snum,
                "tag": "ao3",
            })
            n += 1
        print(f"  shard {snum:03d}: {n} chapters", flush=True)

    # story-level split: all chapters of a storyId get the same split,
    # grouped by chapter hash so exact-duplicate chapters never straddle splits
    rng = random.Random(42)
    by_story: dict[str, list[dict]] = {}
    for r in records:
        by_story.setdefault(r["story_id"], []).append(r)
    stories = list(by_story.items())
    rng.shuffle(stories)
    n = len(stories)
    n_val = max(1, int(n * 0.02))
    n_test = max(1, int(n * 0.02))
    for i, (sid, recs) in enumerate(stories):
        split = "train"
        if i < n_val:
            split = "val"
        elif i < n_val + n_test:
            split = "test"
        for r in recs:
            r["split"] = split
    # exact-duplicate guard across splits: same sha256 in >1 split -> force train
    by_hash: dict[str, set[str]] = {}
    for r in records:
        by_hash.setdefault(r["sha256"], set()).add(r["split"])
    straddle = {h for h, s in by_hash.items() if len(s) > 1}
    if straddle:
        print(f"note: {len(straddle)} duplicate-hash groups straddle splits -> moved to train")
        for r in records:
            if r["sha256"] in straddle:
                r["split"] = "train"
    rng.shuffle(records)

    with open(MANIFEST, "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    summary = {"repo": "ao3", "revision": REVISION, "splits": {}}
    for split in ("train", "val", "test"):
        rs = [r for r in records if r["split"] == split]
        summary["splits"][split] = {
            "files": len(rs), "bytes": sum(r["bytes"] for r in rs),
            "stories": len({r["story_id"] for r in rs}),
        }
    with open(DATA / "story_splits_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
