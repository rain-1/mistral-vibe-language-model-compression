#!/usr/bin/env python3
"""Generate a batch of free samples from a checkpoint with varied prompts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent

PROMPTS = [
    "// SPDX-License-Identifier: GPL-2.0\n",
    "/*\n * foo_bar - handle the foo thing\n *\n * @f: the foo\n */\n",
    "static int __init foo_init(void)\n{\n\t",
    "static void __exit foo_exit(void)\n{\n\t",
    "struct foo {\n",
    "static const struct file_operations foo_fops = {\n",
    "static int foo_open(struct inode *inode, struct file *file)\n{\n",
    "#define FOO_MAX_",
    "if (ret < 0) {\n",
    "list_for_each_entry(pos, &foo->list, node) {\n",
    "static int foo_probe(struct platform_device *pdev)\n{\n\t",
    "MODULE_DESCRIPTION(\"",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/small-16k/ckpt_016000.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tokenizer.json"))
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=str(ROOT / "results/samples/batch_small16k.txt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, p in enumerate(PROMPTS):
            ids = tok.encode(p).ids
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
            text = tok.decode(y[0, len(ids):].tolist(), skip_special_tokens=False)
            f.write(f"===== sample {i} (prompt {p!r}, temp={args.temperature}) =====\n")
            f.write(p + text + "\n\n")
            print(f"--- sample {i}: {p[:50]!r}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
