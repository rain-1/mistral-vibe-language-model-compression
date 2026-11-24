#!/usr/bin/env python3
"""Near-duplicate audit of the kernel corpus (spec section 5).

Computes 64-bit minhash signatures over shingles (8-grams) of normalized
whitespace-collapsed source for every manifest file, then reports pairs with
estimated Jaccard similarity >= threshold, flagging pairs that straddle
train/eval splits (contamination risk).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
WS = re.compile(rb"\s+")


def shingles(text: bytes, k: int = 8) -> set[int]:
    t = WS.sub(b" ", text)
    if len(t) < k:
        return {hash(t)}
    return {hash(t[i:i + k]) for i in range(len(t) - k + 1)}


def minhash(sh: set[int], n: int = 64, seed: int = 0) -> tuple[int, ...]:
    import random
    rng = random.Random(seed)
    perms = [rng.getrandbits(64) for _ in range(n)]
    return tuple(min((p * 0x9E3779B97F4A7C15) ^ s for s in sh) if sh else 0 for p in perms)


def jaccard_est(sig_a, sig_b) -> float:
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-root", default=str(DATA / "linux-v6.6"))
    ap.add_argument("--manifest", default=str(DATA / "manifest.jsonl"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=str(DATA / "near_dup_report.json"))
    ap.add_argument("--max-bytes", type=int, default=2_000_000, help="skip huge files")
    args = ap.parse_args()

    with open(args.manifest) as f:
        recs = [json.loads(l) for l in f]
    root = Path(args.kernel_root)

    # LSH banding: 64 perms in 16 bands of 4
    sigs = {}
    shingle_sets = {}
    for r in recs:
        if r["bytes"] > args.max_bytes:
            continue
        text = (root / r["path"]).read_bytes()
        sh = shingles(text)
        shingle_sets[r["path"]] = sh
        sigs[r["path"]] = minhash(sh)

    bands = defaultdict(list)
    for path, sig in sigs.items():
        for b in range(16):
            bands[(b, sig[b * 4:(b + 1) * 4])].append(path)

    candidates = set()
    for key, paths in bands.items():
        if len(paths) > 1:
            for i in range(len(paths)):
                for j in range(i + 1, min(i + 50, len(paths))):
                    candidates.add((paths[i], paths[j]) if paths[i] < paths[j] else (paths[j], paths[i]))

    split_of = {r["path"]: r["split"] for r in recs}
    report = {"threshold": args.threshold, "pairs": [], "cross_split_pairs": []}
    for a, b in candidates:
        j = jaccard_est(sigs[a], sigs[b])
        if j >= args.threshold:
            entry = {"a": a, "b": b, "est_jaccard": round(j, 3),
                     "split_a": split_of[a], "split_b": split_of[b]}
            report["pairs"].append(entry)
            if split_of[a] != split_of[b]:
                report["cross_split_pairs"].append(entry)
    report["n_files"] = len(sigs)
    report["n_candidate_pairs"] = len(candidates)
    report["n_pairs"] = len(report["pairs"])
    report["n_cross_split"] = len(report["cross_split_pairs"])
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"files={report['n_files']} candidate_pairs={report['n_candidate_pairs']} "
          f"pairs>={args.threshold}: {report['n_pairs']} cross-split: {report['n_cross_split']}")
    print(f"saved -> {args.out}")
    for e in report["cross_split_pairs"][:20]:
        print(f"  CROSS {e['split_a']} vs {e['split_b']}  J={e['est_jaccard']}  {e['a']} <-> {e['b']}")


if __name__ == "__main__":
    main()
