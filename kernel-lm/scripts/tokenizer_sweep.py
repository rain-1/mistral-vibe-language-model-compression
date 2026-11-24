#!/usr/bin/env python3
"""Train byte-level BPE tokenizers at several vocab sizes on the train split
and report tokenization statistics (bytes/token, vocab utilization) on val.

Full BPB comparison requires retraining a model per tokenizer (see EXPERIMENTS);
here we measure the tokenizer-only cost: bits/byte of the *tokenization*
itself is not defined, so we report tokens/byte and mean token id entropy
proxy. Real comparison: train tiny models per tokenizer (planned).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

DATA = Path(__file__).resolve().parent.parent / "data"


def train_bpe(vocab_size: int, files: list[str], out: Path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<|file|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train(files, trainer)
    tok.save(str(out))
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-root", default=str(DATA / "linux-v6.6"))
    ap.add_argument("--manifest", default=str(DATA / "manifest.jsonl"))
    ap.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--out-dir", default=str(DATA / "tokenizers"))
    ap.add_argument("--sample-files", type=int, default=300, help="val sample for stats")
    args = ap.parse_args()

    with open(args.manifest) as f:
        recs = [json.loads(l) for l in f]
    train_files = [str(Path(args.kernel_root) / r["path"]) for r in recs if r["split"] == "train"]
    val_sample = [r for r in recs if r["split"] == "val"][: args.sample_files]
    val_paths = [str(Path(args.kernel_root) / r["path"]) for r in val_sample]
    val_bytes = sum(r["bytes"] for r in val_sample)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for size in args.sizes:
        name = f"bpe-{size}"
        out = out_dir / f"{name}.json"
        if out.exists():
            tok = Tokenizer.from_file(str(out))
        else:
            print(f"training {name} ...", flush=True)
            tok = train_bpe(size, train_files, out)
        # stats on val sample
        n_tok = 0
        for p in val_paths:
            raw = Path(p).read_bytes()
            n_tok += len(tok.encode(raw.decode("utf-8", errors="surrogateescape")).ids)
        results[name] = {
            "vocab_size": tok.get_vocab_size(),
            "val_bytes": val_bytes, "val_tokens": n_tok,
            "bytes_per_token": val_bytes / n_tok,
            "tokens_per_byte": n_tok / val_bytes,
        }
        print(f"{name:10s} vocab={tok.get_vocab_size():6d} bytes/token={val_bytes/n_tok:.3f}")
    with open(out_dir / "sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_dir / 'sweep.json'}")


if __name__ == "__main__":
    main()
