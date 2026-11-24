#!/usr/bin/env python3
"""Scan the pinned Linux kernel checkout and build a contamination-aware manifest.

Split policy (never token-level random):
  - holdout_subsystem : entire top-level kernel domains held out of training
                        (used as in-domain-but-held-out eval)
  - holdout_driver    : entire driver families held out of training
  - val               : seeded file-level split (by content hash group, so exact
                        duplicates never straddle splits) from the rest
  - train             : everything else

Outputs:
  data/manifest.jsonl   one record per file: path, sha256, bytes, domain, split, ...
  data/splits_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

# Top-level kernel domains held out entirely (in-domain held-out evaluation).
HOLDOUT_SUBSYSTEMS = {"mm", "fs", "net", "crypto", "security", "sound", "virt"}
# Driver families held out entirely (held-out driver families).
HOLDOUT_DRIVER_FAMILIES = {"drivers/gpu", "drivers/net", "drivers/media", "drivers/staging"}
VAL_FRACTION = 0.02
SEED = 42


def sha256_file(path: Path, buf: bytes = b"") -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), buf):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-root", default=str(Path(__file__).resolve().parent.parent / "data" / "linux-v6.6"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    args = ap.parse_args()

    root = Path(args.kernel_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    git_rev = os.popen(f"git -C {root} rev-parse HEAD").read().strip()
    git_desc = os.popen(f"git -C {root} describe --tags --always 2>/dev/null").read().strip()

    files = []
    for p in sorted(root.rglob("*")):
        if ".git" in p.parts:
            continue
        if p.is_file() and p.suffix in (".c", ".h"):
            files.append(p)

    records = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        parts = rel.split("/")
        domain = parts[0]
        sub = "/".join(parts[:2]) if len(parts) > 1 else domain
        size = p.stat().st_size
        records.append({
            "path": rel,
            "sha256": sha256_file(p),
            "bytes": size,
            "domain": domain,
            "subdomain": sub,
        })

    # --- assign holdout splits by path rules ---
    for r in records:
        if r["domain"] in HOLDOUT_SUBSYSTEMS:
            r["split"] = "holdout_subsystem"
        elif r["subdomain"] in HOLDOUT_DRIVER_FAMILIES:
            r["split"] = "holdout_driver"
        else:
            r["split"] = "train"  # provisional; file-level val split below

    # --- file-level val split, grouped by content hash so exact duplicates
    #     always land in the same split (no leakage) ---
    rng = random.Random(SEED)
    by_hash: dict[str, list[dict]] = {}
    for r in records:
        if r["split"] == "train":
            by_hash.setdefault(r["sha256"], []).append(r)
    hash_groups = list(by_hash.items())
    rng.shuffle(hash_groups)
    n_val_groups = max(1, int(len(hash_groups) * VAL_FRACTION))
    val_hashes = {h for h, _ in hash_groups[:n_val_groups]}
    for h, group in hash_groups:
        if h in val_hashes:
            for r in group:
                r["split"] = "val"

    # exact duplicate accounting
    dup_count = 0
    for h, group in by_hash.items():
        if len(group) > 1:
            dup_count += len(group) - 1
            for r in group[1:]:
                r["dup_of"] = group[0]["path"]

    # --- write manifest ---
    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for r in records:
            r["repo"] = "torvalds/linux"
            r["revision"] = git_rev
            r["tag"] = git_desc
            f.write(json.dumps(r, sort_keys=True) + "\n")

    summary = {
        "repo": "torvalds/linux",
        "revision": git_rev,
        "tag": git_desc,
        "total_files": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
        "exact_duplicate_files_removed_from_unique_count": dup_count,
        "holdout_subsystems": sorted(HOLDOUT_SUBSYSTEMS),
        "holdout_driver_families": sorted(HOLDOUT_DRIVER_FAMILIES),
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "splits": {},
    }
    for split in ("train", "val", "holdout_subsystem", "holdout_driver"):
        rs = [r for r in records if r["split"] == split]
        summary["splits"][split] = {
            "files": len(rs),
            "bytes": sum(r["bytes"] for r in rs),
        }
    with open(out / "splits_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
