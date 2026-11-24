#!/usr/bin/env python3
"""Produce charts from results/ JSONL logs and eval JSONs.

Charts:
  - training loss vs step and vs tokens
  - val loss vs tokens
  - BPB bar chart across corpora for one or more checkpoints
  - gpu benchmark table chart (tok/s and VRAM vs config)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def plot_run_log(log_path: Path, out_dir: Path):
    steps, loss, toks, lr, gn, tps = [], [], [], [], [], []
    val = []
    with open(log_path) as f:
        for line in f:
            r = json.loads(line)
            if "loss" in r and r["loss"] is not None:
                steps.append(r["step"]); loss.append(r["loss"]); toks.append(r["tokens"])
                lr.append(r["lr"]); gn.append(r.get("grad_norm")); tps.append(r.get("tokens_per_sec"))
            if "val_loss" in r:
                val.append((r["step"], r["val_loss"], r.get("tokens", 0)))
    if not steps:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, loss); axes[0, 0].set_title("train loss vs step"); axes[0, 0].set_xlabel("step")
    axes[0, 1].plot(toks, loss); axes[0, 1].set_title("train loss vs tokens"); axes[0, 1].set_xlabel("tokens")
    axes[1, 0].plot(steps, lr); axes[1, 0].set_title("lr"); axes[1, 0].set_yscale("log")
    if val:
        axes[1, 1].plot([v[0] for v in val], [v[1] for v in val], "o-")
        axes[1, 1].set_title("val loss vs step")
    fig.tight_layout()
    fig.savefig(out_dir / f"{log_path.parent.name}_curves.png", dpi=110)
    print(f"saved {out_dir / (log_path.parent.name + '_curves.png')}")


def plot_bpb(eval_json: Path, out_dir: Path):
    with open(eval_json) as f:
        payload = json.load(f)
    res = payload["results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for corpus, corpora in res.items():
        names = list(corpora.keys())
        bpbs = [corpora[n]["bits_per_byte"] for n in names]
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.7), 4))
        ax.bar(names, bpbs)
        ax.set_ylabel("bits per byte")
        ax.set_title(f"BPB by corpus — {Path(eval_json).name}")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        out = out_dir / f"bpb_{Path(eval_json).stem}.png"
        fig.savefig(out, dpi=110)
        print(f"saved {out}")


def plot_bench(bench_json: Path, out_dir: Path):
    with open(bench_json) as f:
        rows = json.load(f)["rows"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows if not r.get("oom")]
    labels = [f"{r['model']}\n{r['micro_batch']}x{r['context']}" for r in ok]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(labels, [r["tokens_per_sec"] for r in ok])
    axes[0].set_title("tokens/s"); plt.setp(axes[0].get_xticklabels(), rotation=60, ha="right", fontsize=7)
    axes[1].bar(labels, [r["peak_vram_gb"] for r in ok])
    axes[1].axhline(6.0, color="r", ls="--", label="6 GB")
    axes[1].set_title("peak VRAM (GB)"); axes[1].legend()
    plt.setp(axes[1].get_xticklabels(), rotation=60, ha="right", fontsize=7)
    fig.tight_layout()
    out = out_dir / "gpu_benchmark.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    out_dir = RESULTS / "charts"
    if args.all:
        for log in RESULTS.glob("*/log.jsonl"):
            plot_run_log(log, out_dir)
        for ej in (RESULTS / "eval").glob("*.json"):
            plot_bpb(ej, out_dir)
        b = RESULTS / "benchmarks" / "gpu_benchmark.json"
        if b.exists():
            plot_bench(b, out_dir)
    print("done")


if __name__ == "__main__":
    main()
