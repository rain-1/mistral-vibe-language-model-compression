#!/usr/bin/env python3
"""Training loop for the kernel-lm baseline transformer.

- FP16 mixed precision via autocast + GradScaler (RTX 2060 has no native BF16).
- AdamW, cosine LR schedule with warmup, gradient clipping, grad accumulation.
- Checkpointing every ckpt_every steps; --resume continues a run.
- Logs JSONL locally (always) and to Weights & Biases if wandb is installed
  and --wandb is passed.
- Records: tokens/s, GPU mem, wall time, train/val loss, LR, grad norm,
  estimated FLOPs, total tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_lr(step: int, t: dict) -> float:
    lr, min_lr, warmup, max_steps = t["learning_rate"], t["min_learning_rate"], t["warmup_steps"], t["max_steps"]
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


class TokenStream:
    """Random-window sampler over a memmap'd token array.

    For train: windows may straddle file boundaries (files are joined by the
    <|file|> separator, which the model learns as a boundary signal).
    """

    def __init__(self, path: Path, context_length: int, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.ctx = context_length
        self.n = len(self.data)
        self.rng = np.random.default_rng(seed)

    def batch(self, bsz: int) -> torch.Tensor:
        max_start = self.n - self.ctx - 1
        starts = self.rng.integers(0, max_start, size=bsz)
        idx = np.stack([np.asarray(self.data[s:s + self.ctx + 1]) for s in starts])
        return torch.from_numpy(idx.astype(np.int64))


@torch.no_grad()
def evaluate(model, stream: TokenStream, steps: int, ctx: int, device) -> float:
    """Mean next-token cross-entropy over random windows (val split)."""
    model.eval()
    losses = []
    for _ in range(steps):
        idx = stream.batch(4).to(device)
        _, loss = model(idx[:, :-1], idx[:, 1:])
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "tiny.yaml"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    ap.add_argument("--max-steps", type=int, default=None, help="override max_steps")
    ap.add_argument("--micro-batch", type=int, default=None, help="override micro_batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="override gradient_accumulation_steps")
    ap.add_argument("--context", type=int, default=None, help="override context_length")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="kernel-lm")
    ap.add_argument("--tokens-dir", default=str(DATA / "tokens"),
                    help="directory with train.bin/val.bin (allows tokenizer experiments)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mc = ModelConfig.from_dict(cfg["model"])
    tc = cfg["training"]
    if args.context:
        mc.context_length = args.context
    if args.max_steps:
        tc["max_steps"] = args.max_steps
    if args.micro_batch:
        tc["micro_batch_size"] = args.micro_batch
    if args.grad_accum:
        tc["gradient_accumulation_steps"] = args.grad_accum

    run_name = args.run_name or f"{Path(args.config).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = ROOT / "results" / run_name
    ckpt_dir = ROOT / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  run={run_name}")

    model = Transformer(mc).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    tdir = Path(args.tokens_dir)
    train_stream = TokenStream(tdir / "train.bin", mc.context_length, tc["seed"])
    val_stream = TokenStream(tdir / "val.bin", mc.context_length, tc["seed"] + 1)

    # AdamW with weight decay only on >=2D params (standard)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tc["learning_rate"], betas=(tc["beta1"], tc["beta2"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(tc["dtype"] == "float16" and device == "cuda"))

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_step = ck["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}")

    if args.grad_checkpointing:
        for block in model.blocks:
            block.attn = torch.utils.checkpoint.checkpoint_wrapper(block.attn)
            block.mlp = torch.utils.checkpoint.checkpoint_wrapper(block.mlp)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name=run_name,
                                   config={"model": mc.to_dict(), "training": tc}, resume="allow")
        except ImportError:
            print("wandb not installed; logging to JSONL only")

    micro_bsz = tc["micro_batch_size"]
    accum = tc["gradient_accumulation_steps"]
    ctx = mc.context_length
    tokens_per_step = micro_bsz * accum * ctx
    # rough training FLOPs estimate: 6 * N * tokens (fwd+bwd) per step
    flops_per_step = 6 * n_params * tokens_per_step

    log_file = open(log_path, "a")
    t0 = time.time()
    total_tokens = start_step * tokens_per_step

    for step in range(start_step, tc["max_steps"]):
        lr = get_lr(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_t0 = time.time()
        for _ in range(accum):
            idx = train_stream.batch(micro_bsz).to(device)
            with torch.amp.autocast("cuda", enabled=(tc["dtype"] == "float16" and device == "cuda"),
                                    dtype=torch.float16):
                _, loss = model(idx[:, :-1], idx[:, 1:])
            step_loss += loss.item()
            scaler.scale(loss / accum).backward()
        grad_norm = None
        if tc["grad_clip"]:
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()

        total_tokens += tokens_per_step
        if step % tc["log_every"] == 0 or step == tc["max_steps"] - 1:
            dt = time.time() - step_t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            rec = {
                "step": step, "lr": lr, "loss": step_loss / accum,
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "tokens": total_tokens, "tokens_per_sec": tokens_per_step / dt,
                "sec_per_step": dt, "gpu_mem_gb": mem,
                "est_flops_total": flops_per_step * (step + 1),
                "wall_time_s": time.time() - t0,
            }
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            print(f"step {step:5d} | loss {rec['loss']:.4f} | lr {lr:.2e} | "
                  f"gnorm {rec['grad_norm']:.2f} | {rec['tokens_per_sec']:.0f} tok/s | "
                  f"{mem:.2f} GB | {dt:.2f}s")
            if wandb_run:
                wandb_run.log(rec, step=step)

        if (step + 1) % tc["eval_every"] == 0 or step == tc["max_steps"] - 1:
            val_loss = evaluate(model, val_stream, args.eval_steps, ctx, device)
            rec = {"step": step, "val_loss": val_loss, "val_ppl": math.exp(val_loss),
                   "wall_time_s": time.time() - t0, "tokens": total_tokens}
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            print(f"  >> val loss {val_loss:.4f}  ppl {math.exp(val_loss):.2f}")
            if wandb_run:
                wandb_run.log(rec, step=step)

        if (step + 1) % tc["ckpt_every"] == 0 or step == tc["max_steps"] - 1:
            ck = {
                "step": step, "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(), "config": cfg, "run_name": run_name,
                "n_params": n_params,
            }
            path = ckpt_dir / f"ckpt_{step+1:06d}.pt"
            torch.save(ck, path)
            torch.save(ck, ckpt_dir / "latest.pt")
            print(f"  saved {path}")

    log_file.close()
    if wandb_run:
        wandb_run.finish()
    print("done.")


if __name__ == "__main__":
    main()
