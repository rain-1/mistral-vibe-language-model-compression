#!/usr/bin/env python3
"""Quantized story-model comparison chart: artifact size vs BPB, and xz crossover."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CH = ROOT / "results/charts"


def flat(p):
    d = json.load(open(p))["results"]
    return {k: v for _c, corpora in d.items() for k, v in corpora.items()}


dense = flat(ROOT / "results/eval/story-ckpt_030000.json")
s03 = flat(ROOT / "results/eval/story-s03-dequant.json")
s05 = flat(ROOT / "results/eval/story-s05-dequant.json")

models = ["dense fp32\n169 MB", "pruned 30% + INT8\n43.4 MB", "pruned 50% + INT8\n43.4 MB"]
arts = [169.0, 43.4, 43.4]
val_bpb = [dense["val"]["bits_per_byte"], s03["val"]["bits_per_byte"], s05["val"]["bits_per_byte"]]
test_bpb = [dense["test"]["bits_per_byte"], s03["test"]["bits_per_byte"], s05["test"]["bits_per_byte"]]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(3); w = 0.35
a1.bar(x - w / 2, val_bpb, w, color="crimson", label="val BPB")
a1.bar(x + w / 2, test_bpb, w, color="#ff8205", label="test BPB")
a1.axhline(2.202, color="seagreen", ls="--", label="xz-6 (2.20)")
for xi, (v, t) in enumerate(zip(val_bpb, test_bpb)):
    a1.text(xi - w / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)
    a1.text(xi + w / 2, t + 0.03, f"{t:.3f}", ha="center", fontsize=8)
a1.set_xticks(x); a1.set_xticklabels(models, fontsize=8)
a1.set_ylabel("BPB (lower = better)"); a1.set_title("story model: dense vs pruned+INT8 quantized")
a1.legend(fontsize=8); a1.grid(alpha=0.3, axis="y")

sizes = np.logspace(5, 11, 300)
for bpb0, art, c, lab in ((val_bpb[0], arts[0], "darkred", "dense 169 MB"),
                          (val_bpb[1], arts[1], "#ff8205", "pruned30+INT8 43.4 MB")):
    a2.plot(sizes / 1e6, bpb0 + 8 * art * 1e6 / sizes, color=c, lw=2, label=f"neural total, {lab}")
    cross = 8 * art * 1e6 / (2.202 - bpb0)
    a2.axvline(cross / 1e6, color=c, ls=":", lw=1.5)
    a2.text(cross / 1e6 * 1.12, 2.6, f"xz crossover ≈ {cross/1e6:.0f} MB", fontsize=8, color=c)
a2.axhline(2.202, color="seagreen", ls="--", label="xz-6 (2.20 BPB)")
a2.set_xscale("log"); a2.set_xlabel("corpus size (MB)"); a2.set_ylabel("total BPB incl. model")
a2.set_title("quantized model pays for itself much sooner")
a2.legend(fontsize=8); a2.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(CH / "story_quant.png", dpi=110, bbox_inches="tight")
print("saved", CH / "story_quant.png")
print("val BPB:", [round(v, 4) for v in val_bpb], "test:", [round(v, 4) for v in test_bpb])
