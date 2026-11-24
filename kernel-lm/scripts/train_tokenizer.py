#!/usr/bin/env python3
"""Train a lossless byte-level BPE tokenizer (vocab 4096 incl. 1 special token)
exclusively on the *training* split files, then verify exact byte round-trip.

Byte-level BPE is lossless by construction: every byte maps to a base token.
The special token <|file|> (id 4095) separates files in the concatenated stream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

DATA = Path(__file__).resolve().parent.parent / "data"
VOCAB_SIZE = 4096
SPECIAL = "<|file|>"


def iter_train_files(manifest_path: Path):
    with open(manifest_path) as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "train":
                yield r["path"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-root", default=str(DATA / "linux-v6.6"))
    ap.add_argument("--manifest", default=str(DATA / "manifest.jsonl"))
    ap.add_argument("--out", default=str(DATA / "tokenizer.json"))
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()

    tok = Tokenizer(models.BPE(unk_token=None))
    # ByteLevel pre-tokenizer + decoder: maps bytes <-> unicode losslessly and
    # splits on the GPT-2 regex so merges respect word-ish boundaries.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,  # special token counts toward the vocab
        special_tokens=[SPECIAL],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    files = [str(Path(args.kernel_root) / p) for p in iter_train_files(Path(args.manifest))]
    print(f"training tokenizer on {len(files)} train files ...")
    tok.train(files, trainer)
    tok.save(args.out)
    print(f"saved -> {args.out}  vocab={tok.get_vocab_size()}")

    # --- verification: exact byte-for-byte reconstruction on a sample of files
    #     from *every* split (tokenizer was trained on train only) ---
    import random
    rng = random.Random(0)
    with open(args.manifest) as f:
        recs = [json.loads(l) for l in f]
    sample = rng.sample(recs, min(300, len(recs)))
    bad = 0
    for r in sample:
        raw = (Path(args.kernel_root) / r["path"]).read_bytes()
        enc = tok.encode(raw.decode("utf-8", errors="surrogateescape"))
        dec = tok.decode(enc.ids).encode("utf-8", errors="surrogateescape")
        if dec != raw:
            bad += 1
            print("MISMATCH:", r["path"])
    print(f"round-trip check: {len(sample) - bad}/{len(sample)} files exact")
    assert bad == 0, "tokenizer is not lossless!"
    print("LOSSLESS OK")


if __name__ == "__main__":
    main()
