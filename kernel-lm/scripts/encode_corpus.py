#!/usr/bin/env python3
"""Encode a tree of C/H files into memmap-able token arrays for evaluation.

Usage:
  encode_corpus.py --root <repo> --out-dir data/eval_tokens --name musl \
      [--rev <git sha>] [--subdir src] [--max-files N]

Writes:
  <out-dir>/<name>.bin            uint16 token ids, files concatenated (no separators)
  <out-dir>/<name>.offsets.json   {"sep_id": null, "offsets": [[s,e],...]} per file
  <out-dir>/<name>.meta.json      {"files": [{path, sha256, bytes, token_count}], "revision": ...}
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

_BYTE_ALPHABET = pre_tokenizers.ByteLevel.alphabet()


def encode_bytes(tok: Tokenizer, raw: bytes) -> list[int]:
    """Lossless encode of arbitrary bytes. Falls back to a manual byte->char
    mapping (pre-tokenizer disabled) for the rare file whose surrogateescape
    decoding the Rust core rejects."""
    try:
        return tok.encode(raw.decode("utf-8", errors="surrogateescape")).ids
    except TypeError:
        bare = Tokenizer(tok.model)
        s = "".join(_BYTE_ALPHABET[b] for b in raw)
        return bare.encode(s).ids


def encode_files(tok, files: list[Path], root: Path):
    all_ids: list[int] = []
    offsets = []
    meta = []
    for p in files:
        raw = p.read_bytes()
        ids = encode_bytes(tok, raw)
        start = len(all_ids)
        all_ids.extend(ids)
        offsets.append([start, len(all_ids)])
        meta.append({
            "path": p.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "token_count": len(ids),
        })
    return np.array(all_ids, dtype=np.uint16), offsets, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", default=str(Path(__file__).resolve().parent.parent / "data" / "tokenizer.json"))
    ap.add_argument("--rev", default=None, help="git revision string to record")
    ap.add_argument("--include", nargs="+", default=["*.c", "*.h"])
    ap.add_argument("--exclude", nargs="+", default=[".git"])
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--min-bytes", type=int, default=64, help="skip tiny files")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)

    files = []
    for pat in args.include:
        for p in sorted(root.rglob(pat)):
            if any(x in p.parts for x in args.exclude):
                continue
            if p.stat().st_size < args.min_bytes:
                continue
            files.append(p)
    if args.max_files:
        files = files[: args.max_files]
    print(f"encoding {len(files)} files from {root} as '{args.name}'")

    arr, offsets, meta = encode_files(tok, files, root)
    arr.tofile(out_dir / f"{args.name}.bin")
    with open(out_dir / f"{args.name}.offsets.json", "w") as f:
        json.dump({"sep_id": None, "offsets": offsets}, f)
    with open(out_dir / f"{args.name}.meta.json", "w") as f:
        json.dump({"root": str(root), "revision": args.rev, "files": meta}, f)
    print(f"  tokens={len(arr)} bytes={sum(m['bytes'] for m in meta)} -> {out_dir / (args.name + '.bin')}")


if __name__ == "__main__":
    main()
