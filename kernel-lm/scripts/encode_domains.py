#!/usr/bin/env python3
"""Encode each top-level kernel domain as its own eval corpus (for the
cross-subsystem generalization matrix), sampling from val+holdout files."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-root", default=str(DATA / "linux-v6.6"))
    ap.add_argument("--manifest", default=str(DATA / "manifest.jsonl"))
    ap.add_argument("--tokenizer", default=str(DATA / "tokenizer.json"))
    ap.add_argument("--out-dir", default=str(DATA / "domain_tokens"))
    ap.add_argument("--max-files-per-domain", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    root = Path(args.kernel_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with open(args.manifest) as f:
        recs = [json.loads(l) for l in f]
    by_domain: dict[str, list[dict]] = {}
    for r in recs:
        if r["split"] in ("val", "holdout_subsystem"):
            by_domain.setdefault(r["domain"], []).append(r)
    for dom, rs in sorted(by_domain.items()):
        rng.shuffle(rs)
        rs = rs[: args.max_files_per_domain]
        all_ids, offsets, meta = [], [], []
        for r in rs:
            raw = (root / r["path"]).read_bytes()
            ids = tok.encode(raw.decode("utf-8", errors="surrogateescape")).ids
            offsets.append([len(all_ids), len(all_ids) + len(ids)])
            all_ids.extend(ids)
            meta.append({"path": r["path"], "sha256": r["sha256"], "bytes": r["bytes"],
                         "token_count": len(ids)})
        arr = np.array(all_ids, dtype=np.uint16)
        arr.tofile(out_dir / f"{dom}.bin")
        with open(out_dir / f"{dom}.offsets.json", "w") as f:
            json.dump({"sep_id": None, "offsets": offsets}, f)
        with open(out_dir / f"{dom}.meta.json", "w") as f:
            json.dump({"root": str(root), "revision": recs[0]["revision"], "files": meta}, f)
        print(f"{dom:12s} files={len(rs):4d} tokens={len(arr):8d} bytes={sum(m['bytes'] for m in meta)}")


if __name__ == "__main__":
    main()
