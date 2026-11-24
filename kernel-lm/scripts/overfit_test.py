#!/usr/bin/env python3
"""Correctness test: a Tiny model must be able to overfit a small fixed batch
to near-zero loss. If it cannot, something is wrong with the model/loop.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=4096, n_layer=4, n_embd=256, n_head=4,
                      intermediate_size=683, context_length=args.ctx)
    model = Transformer(cfg).to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M device={device}")

    data = np.memmap(ROOT / "data" / "tokens" / "train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(data) - args.ctx - 1, size=args.batch)
    idx = torch.from_numpy(np.stack([np.asarray(data[s:s + args.ctx + 1]) for s in starts]).astype(np.int64)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
            _, loss = model(idx[:, :-1], idx[:, 1:])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step {step:4d} loss {loss.item():.4f}  ({time.time()-t0:.1f}s)")
    final = loss.item()
    print(f"final loss: {final:.4f}")
    assert final < 0.5, f"FAILED to overfit (loss {final:.3f} >= 0.5)"
    print("OVERFIT TEST PASSED")


if __name__ == "__main__":
    main()
