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


def shingles(text: bytes, k: int = 8):
    """Vectorized rolling base-257 hash (mod 2^64) of whitespace-normalized k-grams."""
    import numpy as np
    t = WS.sub(b" ", text)
    n = len(t) - k + 1
    if n <= 0:
        return np.array([np.uint64(hash(t) & 0xFFFFFFFFFFFFFFFF)], dtype=np.uint64)
    w = np.frombuffer(t, dtype=np.uint8).astype(np.uint64)
    base = 257
    MOD = 1 << 64
    inv_base = pow(base, -1, MOD)
    # p_i = base^i mod 2^64, vectorized (uint64 wraps naturally)
    p = np.multiply.accumulate(np.full(len(w), np.uint64(base), dtype=np.uint64))
    p[0] = np.uint64(1)
    t2 = w * p  # uint64 multiply wraps mod 2^64
    S = np.cumsum(t2, dtype=np.uint64)  # wraps mod 2^64
    # h_i = (S_{i+k-1} - S_{i-1}) * inv_base^i  (mod 2^64)
    Spre = np.concatenate([np.zeros(1, dtype=np.uint64), S[:-1]])
    inv_p = np.multiply.accumulate(np.full(n, np.uint64(inv_base % MOD), dtype=np.uint64))
    inv_p[0] = np.uint64(1)
    hi = (S[k - 1:] - Spre[:n]) * inv_p
    return np.unique(hi)


def minhash(sh, n: int = 64, seed: int = 0):
    """Vectorized minhash: returns uint64 array of length n."""
    import numpy as np
    rng = np.random.default_rng(seed)
    perms = rng.integers(0, 2**63, size=n, dtype=np.uint64) | np.uint64(1)
    if len(sh) == 0:
        return np.zeros(n, dtype=np.uint64)
    # (n,1) * (1,m) mod 2^64 -> take min over m for each perm
    prod = (perms[:, None] * sh[None, :])
    return prod.min(axis=1)


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

    import numpy as np
    paths_list = list(sigs.keys())
    sigmat = np.stack([sigs[p] for p in paths_list])  # (N, 64) uint64
    NB = 16
    B = 4
    bands = defaultdict(list)
    for b in range(NB):
        keys = sigmat[:, b * B:(b + 1) * B]
        seen = {}
        for i, row in enumerate(map(tuple, keys)):
            bands[(b, row)].append(i)

    candidates = set()
    for key, idxs in bands.items():
        if len(idxs) > 1:
            for x in range(len(idxs)):
                for y in range(x + 1, min(x + 1 + 50, len(idxs))):
                    a, b = idxs[x], idxs[y]
                    candidates.add((a, b) if a < b else (b, a))

    split_of = {r["path"]: r["split"] for r in recs}
    report = {"threshold": args.threshold, "pairs": [], "cross_split_pairs": []}
    for a, b in candidates:
        j = float((sigmat[a] == sigmat[b]).mean())
        if j >= args.threshold:
            entry = {"a": paths_list[a], "b": paths_list[b], "est_jaccard": round(j, 3),
                     "split_a": split_of[paths_list[a]], "split_b": split_of[paths_list[b]]}
            report["pairs"].append(entry)
            if entry["split_a"] != entry["split_b"]:
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
