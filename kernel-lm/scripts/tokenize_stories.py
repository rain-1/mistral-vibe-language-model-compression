#!/usr/bin/env python3
"""Tokenize every manifest file with the trained tokenizer and write
memory-mapped uint16 token arrays + per-file offsets.

Outputs per split S in {train, val, holdout_subsystem, holdout_driver}:
  data/tokens/S.bin       concatenated token ids (uint16), memmap-able
  data/tokens/S.offsets.json  list of [start, end) token ranges, one per file,
                          in manifest order. For train, files are joined by the
                          <|file|> separator token (present in the stream, inside
                          the *next* file's range). For eval splits, no separators:
                          each file is evaluated independently.
Also updates manifest.jsonl with token_count per file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, pre_tokenizers

_BYTE_ALPHABET = pre_tokenizers.ByteLevel.alphabet()


def encode_bytes(tok: Tokenizer, raw: bytes) -> list[int]:
    try:
        return tok.encode(raw.decode("utf-8", errors="surrogateescape")).ids
    except TypeError:
        bare = Tokenizer(tok.model)
        s = "".join(_BYTE_ALPHABET[b] for b in raw)
        return bare.encode(s).ids

DATA = Path(__file__).resolve().parent.parent / "data"
SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-root", default=str(DATA / "linux-v6.6"))
    ap.add_argument("--manifest", default=str(DATA / "manifest.jsonl"))
    ap.add_argument("--tokenizer", default=str(DATA / "tokenizer.json"))
    ap.add_argument("--out-dir", default=str(DATA / "tokens"))
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    sep_id = tok.token_to_id("<|file|>")
    assert sep_id is not None, "special <|file|> token missing"
    vocab = tok.get_vocab_size()
    assert vocab <= 65536
    print(f"tokenizer vocab={vocab}, sep_id={sep_id}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.kernel_root)

    with open(args.manifest) as f:
        records = [json.loads(l) for l in f]

    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for r in records:
        by_split[r["split"]].append(r)

    for split, recs in by_split.items():
        if not recs:
            continue
        if (out_dir / f"{split}.bin").exists() and (out_dir / f"{split}.offsets.json").exists():
            print(f"{split:20s} already tokenized, skipping")
            continue
        # stream tokens to disk in chunks: a full train split can be >1B
        # tokens, far too large to hold as a python list on a 16 GB box
        n_tokens = 0
        offsets = []
        meta_files = []
        buf: list[int] = []
        with open(out_dir / f"{split}.bin", "wb") as bf:
            def flush():
                nonlocal buf, n_tokens
                if buf:
                    np.array(buf, dtype=np.uint16).tofile(bf)
                    n_tokens += len(buf)
                    buf = []

            def _emit(raws, rs):
                nonlocal buf, n_tokens
                strs = [b.decode("utf-8", errors="surrogateescape") for b in raws]
                encs = tok.encode_batch(strs)
                for r, e in zip(rs, encs):
                    ids = e.ids
                    if split == "train" and n_tokens + len(buf) > 0:
                        buf.append(sep_id)
                    start = n_tokens + len(buf)
                    buf.extend(ids)
                    end = n_tokens + len(buf)
                    offsets.append([start, end])
                    r["token_count"] = len(ids)
                    meta_files.append({"path": r["path"], "sha256": r["sha256"],
                                       "bytes": r["bytes"], "token_count": len(ids)})
                    if len(buf) >= 1 << 20:
                        flush()

            pending_raw: list[bytes] = []
            pending_rec: list[dict] = []
            for r in recs:
                pending_raw.append((root / r["path"]).read_bytes())
                pending_rec.append(r)
                if len(pending_raw) >= 2000:
                    _emit(pending_raw, pending_rec)
                    pending_raw, pending_rec = [], []
            if pending_raw:
                _emit(pending_raw, pending_rec)
            flush()
        with open(out_dir / f"{split}.offsets.json", "w") as f:
            json.dump({"sep_id": sep_id if split == "train" else None, "offsets": offsets}, f)
        with open(out_dir / f"{split}.meta.json", "w") as f:
            json.dump({"root": str(root), "revision": recs[0]["revision"], "files": meta_files}, f)
        print(f"{split:20s} files={len(recs):6d} tokens={n_tokens:10d} "
              f"bytes={sum(r['bytes'] for r in recs):10d} -> {out_dir / (split + '.bin')}")

    # rewrite manifest with token counts
    with open(args.manifest, "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print("manifest updated with token_count")


if __name__ == "__main__":
    main()
