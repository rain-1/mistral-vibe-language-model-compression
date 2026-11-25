#!/usr/bin/env python3
"""Unpack the Fimfiction torrent shards (fimfiction_XX.jsonl.zst) into
individual per-story text files + a story manifest.

Layout:
  data/stories/fimfiction/s<shard>/<story_id>.txt   raw story content, verbatim
  data/story_manifest.jsonl                          one record per story

Split assignment is story-level (a "file" = one story), seeded and grouped by
content hash so exact duplicates never straddle splits:
  train 96% / val 2% / test 2%

Usage: unpack_fimfiction.py [--shards 0-19] [--max-per-shard N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fimfiction"
SHARD_DIR = DATA / "fimfiction"
OUT = DATA / "stories"
MANIFEST = DATA / "story_manifest.jsonl"
REVISION = "academictorrents.com/details/08d162a9c91f0801549e2e3907f9b6884b5a2547 (Fimfiction 2025-04-25)"


def story_text(rec: dict) -> tuple[str, str, str]:
    """Return (story_id, title, content) from a jsonl record, tolerating schema."""
    sid = rec.get("id") or rec.get("_id") or rec.get("story_id") or rec.get("url")
    title = rec.get("title") or ""
    content = rec.get("content") or rec.get("text") or rec.get("body") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    if sid is None:
        sid = hashlib.sha256(content.encode("utf-8", "surrogateescape")).hexdigest()[:16]
    return str(sid), str(title), content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="all", help="e.g. '0-4,7' or 'all'")
    ap.add_argument("--max-per-shard", type=int, default=0, help="cap stories per shard (0=all)")
    args = ap.parse_args()

    shard_files = sorted(SHARD_DIR.glob("fimfiction_*.jsonl.zst"))
    if args.shards != "all":
        keep = set()
        for part in args.shards.split(","):
            if "-" in part:
                a, b = part.split("-")
                keep.update(range(int(a), int(b) + 1))
            else:
                keep.add(int(part))
        shard_files = [p for p in shard_files if int(p.stem.split("_")[-1]) in keep]
    print(f"unpacking {len(shard_files)} shards ...")

    records = []
    for shard in shard_files:
        snum = int(re.search(r"(\d+)", shard.stem).group(1))
        outdir = OUT / f"s{snum:02d}"
        outdir.mkdir(parents=True, exist_ok=True)
        n = 0
        proc = subprocess.Popen(["zstd", "-dc", str(shard)], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        buf = b""
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue  # padding/null lines between records
            sid, title, content = story_text(rec)
            if not content.strip():
                continue
            raw = content.encode("utf-8", "surrogateescape")
            p = outdir / f"{sid}.txt"
            if p.exists():
                # id collision: disambiguate
                sid = sid + "_" + hashlib.sha256(raw).hexdigest()[:8]
                p = outdir / f"{sid}.txt"
            p.write_bytes(raw)
            records.append({
                "path": f"s{snum:02d}/{p.name}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "repo": "fimfiction.net via academictorrents",
                "revision": REVISION,
                "split": "train",  # provisional; assigned below
                "story_id": sid,
                "title": title[:300],
                "shard": snum,
                "tag": "fimfiction",
            })
            n += 1
            if args.max_per_shard and n >= args.max_per_shard:
                break
        proc.wait()
        print(f"  shard {snum:02d}: {n} stories -> {outdir}", flush=True)

    # story-level seeded split, grouped by content hash (dedup-safe)
    rng = random.Random(42)
    by_hash: dict[str, list[dict]] = {}
    for r in records:
        by_hash.setdefault(r["sha256"], []).append(r)
    groups = list(by_hash.values())
    rng.shuffle(groups)
    n = len(groups)
    n_val = max(1, int(n * 0.02))
    n_test = max(1, int(n * 0.02))
    for i, g in enumerate(groups):
        split = "train"
        if i < n_val:
            split = "val"
        elif i < n_val + n_test:
            split = "test"
        for r in g:
            r["split"] = split
    rng.shuffle(records)

    with open(MANIFEST, "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    summary = {"repo": "fimfiction", "revision": REVISION, "splits": {}}
    for split in ("train", "val", "test"):
        rs = [r for r in records if r["split"] == split]
        summary["splits"][split] = {"files": len(rs), "bytes": sum(r["bytes"] for r in rs)}
    with open(DATA / "story_splits_summary.json", "w") as f:  # noqa
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
