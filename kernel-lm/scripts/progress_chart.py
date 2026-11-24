#!/usr/bin/env python3
"""Autoresearch-style progress chart: best validation BPB over successive
experiments (by wall-clock time of the eval file), plus BPB vs params/tokens.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "results" / "eval"


def collect():
    rows = []
    for ej in sorted(EVAL.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            with open(ej) as f:
                payload = json.load(f)
        except Exception:
            continue
        step = payload.get("step")
        for corpus, corpora in payload.get("results", {}).items():
            for name, r in corpora.items():
                if name == "val" and r.get("bits_per_byte"):
                    rows.append({
                        "file": ej.name, "mtime": ej.stat().st_mtime,
                        "step": step, "val_bpb": r["bits_per_byte"],
                        "tokens": r.get("tokens"), "ppl": r.get("perplexity"),
                    })
    return rows


def main():
    rows = collect()
    if not rows:
        print("no eval results yet")
        return
    rows.sort(key=lambda r: r["mtime"])
    best = []
    b = float("inf")
    for r in rows:
        b = min(b, r["val_bpb"])
        best.append(b)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xs = [r["file"] for r in rows]
    axes[0].plot(range(len(rows)), [r["val_bpb"] for r in rows], "o-", label="val BPB")
    axes[0].plot(range(len(rows)), best, "-", color="red", label="best so far")
    axes[0].set_xticks(range(len(rows)))
    axes[0].set_xticklabels(xs, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("val bits/byte")
    axes[0].set_title("Progress: validation BPB over experiments")
    axes[0].legend()
    if any(r.get("step") for r in rows):
        axes[1].plot([r["step"] for r in rows], [r["val_bpb"] for r in rows], "o-")
        axes[1].set_xlabel("training step")
        axes[1].set_ylabel("val BPB")
        axes[1].set_title("val BPB vs training step")
    fig.tight_layout()
    out = ROOT / "results/charts/progress.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    for r, b in zip(rows, best):
        print(f"  {r['file']:40s} step={r['step']} val_bpb={r['val_bpb']:.4f} best={b:.4f}")


if __name__ == "__main__":
    main()
