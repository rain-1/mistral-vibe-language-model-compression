#!/usr/bin/env python3
"""Story-model charts: training loss curve + neural vs conventional BPB.

Panel 1: train/val loss vs training tokens (from results/story-*/log.jsonl).
Panel 2: per corpus — gray original bar (8 BPB), gzip-6/zstd-6/xz-6 bars, and
         a stacked neural bar (conditional BPB + model overhead in BPB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "results/story-medium-16k/log.jsonl"))
    ap.add_argument("--eval", default=str(ROOT / "results/eval/story-ckpt_030000.json"))
    ap.add_argument("--comp", nargs="+",
                    default=[str(ROOT / "results/benchmarks/compressors_stories.json"),
                             str(ROOT / "results/benchmarks/compressors_story_eval.json")])
    ap.add_argument("--model-bytes", type=int, default=27_267_584 * 4 + 1_200_000)
    ap.add_argument("--out", default=str(ROOT / "results/charts/story_neural_vs_conventional.png"))
    args = ap.parse_args()

    # ---- panel 1: loss curve
    steps, loss, val_s, val_l, toks = [], [], [], [], []
    with open(args.log) as f:
        for line in f:
            r = json.loads(line)
            if "loss" in r and r.get("step", 0) > 0:
                steps.append(r["step"]); loss.append(r["loss"]); toks.append(r.get("tokens", 0) / 1e6)
            if "val_loss" in r:
                val_s.append(r["step"]); val_l.append(r["val_loss"])

    # ---- panel 2: BPB bars
    with open(args.eval) as f:
        ev = json.load(f)
    neural = {}
    for _c, corpora in ev["results"].items():
        for name, r in corpora.items():
            if r.get("bits_per_byte"):
                neural[name] = r
    comp = {}
    comp_bytes = {}
    for p in args.comp:
        with open(p) as f:
            d = json.load(f)
        for corpus, res in d.items():
            comp[corpus] = {k: v["bits_per_byte"] for k, v in res.items()}
            comp_bytes[corpus] = res["gzip-6"]["orig_bytes"]

    names = [n for n in ("val", "test", "fimfiction", "pg19-test", "pg19-val") if n in neural]
    labels = [f"{n}\n({comp_bytes.get(n, neural[n].get('bytes', 0))/1e6:.0f} MB)" for n in names]
    cond = [neural[n]["bits_per_byte"] for n in names]
    ovh = [8 * args.model_bytes / comp_bytes.get(n, neural[n].get("bytes", 1)) for n in names]
    gz = [comp.get(n, {}).get("gzip-6", np.nan) for n in names]
    zs = [comp.get(n, {}).get("zstd-6", np.nan) for n in names]
    xz = [comp.get(n, {}).get("xz-6", np.nan) for n in names]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.0), gridspec_kw={"width_ratios": [1, 1.7]})
    ax.plot(toks, loss, lw=1.2, label="train loss", color="#ff8205")
    if val_s:
        ax.plot([t / 1e6 for t in [steps[min(len(steps) - 1, 0)]] * len(val_s)] if False else
                [toks[-1] * s / steps[-1] for s in val_s], val_l, "o-", ms=3, label="val loss", color="crimson")
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("cross-entropy (nats)")
    ax.set_title("story-medium-16k (27.3M) on AO3 fanfiction — training loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    x = np.arange(len(names))
    w = 0.16
    ax2.bar(x - 2 * w, [8.0] * len(names), w, color="#3a3a44", label="original (8 BPB)")
    ax2.bar(x - w, cond, w, color="crimson", label="neural stream (conditional BPB)")
    ax2.bar(x - w, ovh, w, bottom=cond, color="darkred", hatch="//", edgecolor="white",
            label=f"+ model ({args.model_bytes/1e6:.0f} MB)")
    ax2.bar(x, gz, w, color="gray", label="gzip-6")
    ax2.bar(x + w, zs, w, color="steelblue", label="zstd-6")
    ax2.bar(x + 2 * w, xz, w, color="seagreen", label="xz-6")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_yscale("log")
    ax2.set_ylim(0.05, 40)
    for xi, c, o in zip(x, cond, ovh):
        ax2.text(xi - w, c + o * 1.2, f"{c:.2f}\n(+{o:.1f})", ha="center", fontsize=6.5, color="darkred")
    ax2.set_ylabel("bits per byte (log, lower = better)")
    ax2.set_title("AO3 story model vs conventional compressors — neural bar stacked with model size")
    ax2.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
