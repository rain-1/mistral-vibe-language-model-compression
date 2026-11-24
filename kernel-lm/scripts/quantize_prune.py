#!/usr/bin/env python3
"""Prune + INT8-quantize a checkpoint, save a compact artifact, and export a
dequantized .pt for evaluation.

- Magnitude pruning of all Linear weights (global, given sparsity).
- Per-output-channel INT8 quantization of Linear weights; per-row INT8 for the
  embedding; LN weights kept fp16.
- Artifact: torch.save of {int8 weights, fp32 scales} -> compact .pt
  Dequantized model: full fp32 Transformer state_dict for evaluation.

Usage:
  quantize_prune.py --checkpoint ... --sparsity 0.4 --out-quant ... --out-dequant ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/small-16k/ckpt_016000.pt"))
    ap.add_argument("--sparsity", type=float, default=0.4)
    ap.add_argument("--out-quant", default=str(ROOT / "checkpoints/small-16k-quant.pt"))
    ap.add_argument("--out-dequant", default=str(ROOT / "checkpoints/small-16k-dequant.pt"))
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc)
    model.load_state_dict(ck["model"])
    sd = model.state_dict()

    # ---- global magnitude pruning over all 2D Linear weights (except embedding) ----
    lin_keys = [k for k, v in sd.items() if k.endswith(".weight") and v.ndim == 2 and "tok_emb" not in k]
    all_w = torch.cat([sd[k].flatten() for k in lin_keys])
    thresh = torch.quantile(all_w.abs(), args.sparsity)
    n_pruned = 0
    for k in lin_keys:
        w = sd[k]
        mask = w.abs() <= thresh
        sd[k] = w * (~mask)
        n_pruned += int(mask.sum())
    print(f"pruned {n_pruned}/{all_w.numel()} linear weights ({args.sparsity:.0%})")

    # ---- INT8 per-channel quantization ----
    q_sd, scales = {}, {}
    for k, v in sd.items():
        if k.endswith(".weight") and v.ndim == 2:
            scale = v.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
            q = torch.round(v / scale).clamp(-128, 127).to(torch.int8)
            q_sd[k] = q
            scales[k] = scale.squeeze(1).float()
        elif k == "tok_emb.weight":
            scale = v.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
            q_sd[k] = torch.round(v / scale).clamp(-128, 127).to(torch.int8)
            scales[k] = scale.squeeze(1).float()
        else:  # LN weights: fp16
            q_sd[k] = v.half()
    torch.save({"config": ck["config"], "step": ck["step"], "sparsity": args.sparsity,
                "q": q_sd, "scales": scales}, args.out_quant)

    # ---- dequantized fp32 model for eval ----
    deq = {}
    for k, v in sd.items():
        if k in q_sd and q_sd[k].dtype == torch.int8:
            deq[k] = (q_sd[k].float() * scales[k].unsqueeze(1)).to(v.dtype)
        else:
            deq[k] = q_sd[k].float() if q_sd[k].dtype == torch.float16 else q_sd[k]
    torch.save({"step": ck["step"], "model": deq, "optimizer": None, "scaler": None,
                "config": ck["config"], "run_name": ck.get("run_name", "") + "-dequant",
                "n_params": sum(p.numel() for p in model.parameters())}, args.out_dequant)

    import os
    qb = os.path.getsize(args.out_quant)
    db = os.path.getsize(args.out_dequant)
    n = sum(p.numel() for p in model.parameters())
    print(f"params={n/1e6:.2f}M")
    print(f"dense fp32 model: {n*4/1e6:.1f} MB | fp16: {n*2/1e6:.1f} MB | pruned int8 artifact: {qb/1e6:.2f} MB")
    print(f"saved quant {args.out_quant} ({qb/1e6:.2f} MB), dequant {args.out_dequant} ({db/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
