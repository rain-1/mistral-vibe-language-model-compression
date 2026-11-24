#!/usr/bin/env python3
"""Benchmark VRAM and throughput across (model, micro_batch, context) configs
to find what fits comfortably in 6 GB on the RTX 2060.

Writes results/benchmarks/gpu_benchmark.json + prints a table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent


def bench_one(cfg: ModelConfig, micro_bsz: int, steps: int = 8, warmup: int = 2):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = Transformer(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    ctx = cfg.context_length
    results = []
    for step in range(steps + warmup):
        idx = torch.randint(0, cfg.vocab_size, (micro_bsz, ctx + 1), device="cuda")
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, loss = model(idx[:, :-1], idx[:, 1:])
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        if step >= warmup:
            results.append(dt)
    peak = torch.cuda.max_memory_allocated() / 1e9
    alloc = torch.cuda.memory_allocated() / 1e9
    dt = sum(results) / len(results)
    toks = micro_bsz * ctx / dt
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_tok = 6 * n_params  # fwd+bwd approx
    del model, opt, scaler
    torch.cuda.empty_cache()
    return {
        "micro_batch": micro_bsz, "context": ctx, "sec_per_step": round(dt, 4),
        "tokens_per_sec": round(toks, 1), "peak_vram_gb": round(peak, 3),
        "alloc_vram_gb": round(alloc, 3),
        "tflops": round(flops_per_tok * toks / 1e12, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["micro", "tiny", "small", "medium"])
    ap.add_argument("--batch-ctx", nargs="+", default=["1x256", "2x512", "4x512", "4x1024", "8x1024", "2x2048", "4x2048", "8x2048", "16x1024"])
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmarks" / "gpu_benchmark.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''}")
    print(f"torch {torch.__version__} cuda {torch.version.cuda} "
          f"bf16_supported={torch.cuda.is_bf16_supported() if device == 'cuda' else 'n/a'}")

    rows = []
    for cname in args.configs:
        with open(ROOT / "configs" / f"{cname}.yaml") as f:
            cfgd = yaml.safe_load(f)
        base = ModelConfig.from_dict(cfgd["model"])
        for bc in args.batch_ctx:
            bsz, ctx = (int(x) for x in bc.split("x"))
            cfg = ModelConfig.from_dict({**base.to_dict(), "context_length": ctx})
            try:
                r = bench_one(cfg, bsz)
                r.update(model=cname, n_params=sum(p.numel() for p in Transformer(cfg).parameters()))
                rows.append(r)
                print(f"{cname:7s} bsz={bsz:2d} ctx={ctx:4d} | {r['tokens_per_sec']:8.0f} tok/s | "
                      f"peak {r['peak_vram_gb']:.2f} GB | {r['tflops']:.1f} TFLOP/s")
            except torch.cuda.OutOfMemoryError:
                print(f"{cname:7s} bsz={bsz:2d} ctx={ctx:4d} | OOM")
                rows.append({"model": cname, "micro_batch": bsz, "context": ctx, "oom": True})
                torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                   "torch": torch.__version__, "cuda": torch.version.cuda, "rows": rows}, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
