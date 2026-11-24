#!/usr/bin/env python3
"""Neural vs conventional compression chart.

Panel 1: per corpus — a gray "original" bar at 8 BPB (= the uncompressed
corpus size, labelled with its MB), gzip-6 / zstd-6 / xz-6 bars, and TWO
stacked neural bars: dense fp32 model and pruned+INT8 quantized model.
Stacked = conditional stream BPB + model overhead (8 * model_bytes / bytes).

Panel 2: total BPB vs corpus size for both model sizes, with xz/gzip lines
and crossover points.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_model_bpb(paths: list[Path]) -> dict[str, dict]:
    out = {}
    for p in paths:
        with open(p) as f:
            payload = json.load(f)
        for _c, corpora in payload["results"].items():
            for name, r in corpora.items():
                if r.get("bits_per_byte"):
                    out[name] = r
    return out


def load_compressor_bpb(paths: list[Path]) -> dict[str, dict[str, float]]:
    out = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        for corpus, res in d.items():
            out[corpus] = {k: v["bits_per_byte"] for k, v in res.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/small-16k/ckpt_016000.pt"))
    ap.add_argument("--model-bytes", type=int, default=12_190_000 * 4, help="dense fp32 model bytes")
    ap.add_argument("--quant-checkpoint", default=str(ROOT / "checkpoints/small-16k-s30-dequant.pt"))
    ap.add_argument("--quant-bytes", type=int, default=13_926_614, help="pruned+INT8 artifact bytes")
    ap.add_argument("--quant-eval", nargs="+",
                    default=[str(ROOT / "results/eval/quant_s30_kernel.json"),
                             str(ROOT / "results/eval/quant_s30_external.json")])
    ap.add_argument("--tokenizer-bytes", type=int,
                    default=(ROOT / "data/tokenizer.json").stat().st_size)
    ap.add_argument("--out", default=str(ROOT / "results/charts/neural_vs_conventional.png"))
    args = ap.parse_args()

    dense_ovh = args.model_bytes + args.tokenizer_bytes
    quant_ovh = args.quant_bytes + args.tokenizer_bytes
    print(f"dense model+tok: {dense_ovh/1e6:.1f} MB | quantized(int8,30% pruned)+tok: {quant_ovh/1e6:.1f} MB")

    neural = load_model_bpb([ROOT / "results/eval/small16k_kernel.json",
                             ROOT / "results/eval/small16k_external.json"])
    neural_q = load_model_bpb([Path(p) for p in args.quant_eval])
    comp = load_compressor_bpb([ROOT / "results/benchmarks/compressors_kernel.json",
                                ROOT / "results/benchmarks/compressors_ext.json"])
    names = [n for n in neural if n in comp and n in neural_q]
    names.sort(key=lambda n: neural[n]["bits_per_byte"])

    cond = [neural[n]["bits_per_byte"] for n in names]
    cond_q = [neural_q[n]["bits_per_byte"] for n in names]
    ovh = [8 * dense_ovh / neural[n]["bytes"] for n in names]
    ovh_q = [8 * quant_ovh / neural[n]["bytes"] for n in names]
    gz = [comp[n].get("gzip-6", np.nan) for n in names]
    zs = [comp[n].get("zstd-6", np.nan) for n in names]
    xz = [comp[n].get("xz-6", np.nan) for n in names]
    labels = [f"{n}\n({neural[n]['bytes']/1e6:.1f} MB)" for n in names]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.2),
                                  gridspec_kw={"width_ratios": [2.4, 1]})
    x = np.arange(len(names))
    w = 0.13
    ax.bar(x - 2.5 * w, [8.0] * len(names), w, color="#3a3a44", label="original (8 BPB = uncompressed)")
    ax.bar(x - 1.5 * w, cond, w, color="crimson", label="neural stream (dense model)")
    ax.bar(x - 1.5 * w, ovh, w, bottom=cond, color="darkred", hatch="//", edgecolor="white",
           label=f"+ dense model ({dense_ovh/1e6:.0f} MB)")
    ax.bar(x - 0.5 * w, cond_q, w, color="#ff8205", label="neural stream (pruned+INT8)")
    ax.bar(x - 0.5 * w, ovh_q, w, bottom=cond_q, color="#b34700", hatch="..", edgecolor="white",
           label=f"+ quantized model ({quant_ovh/1e6:.1f} MB)")
    ax.bar(x + 0.5 * w, gz, w, color="gray", label="gzip-6")
    ax.bar(x + 1.5 * w, zs, w, color="steelblue", label="zstd-6")
    ax.bar(x + 2.5 * w, xz, w, color="seagreen", label="xz-6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=7)
    ax.set_yscale("log")
    max_total = max(max(c + o for c, o in zip(cond, ovh)),
                    max(c + o for c, o in zip(cond_q, ovh_q)))
    ax.set_ylim(0.1, max_total * 2.5)
    for xi, c, o, cq, oq in zip(x, cond, ovh, cond_q, ovh_q):
        ax.text(xi - 1.5 * w, c + o * 1.15, f"{c+o:.1f}", ha="center", fontsize=6.5, color="darkred")
        ax.text(xi - 0.5 * w, cq + oq * 1.15, f"{cq+oq:.1f}", ha="center", fontsize=6.5, color="#b34700")
    ax.set_ylabel("bits per byte (log scale, lower = better)")
    ax.set_title("Neural vs conventional compression — neural bars STACKED with model size;\n"
                 "gray bar = original corpus size (8 BPB); corpus MB in x labels")
    ax.legend(fontsize=7.5, loc="upper right")

    # panel 2: total BPB vs corpus size
    sizes = np.logspace(6, 10.5, 200)
    ax2.plot(sizes / 1e6, neural["val"]["bits_per_byte"] + 8 * dense_ovh / sizes,
             color="darkred", lw=2, label=f"neural total, dense ({dense_ovh/1e6:.0f} MB)")
    ax2.plot(sizes / 1e6, neural_q["val"]["bits_per_byte"] + 8 * quant_ovh / sizes,
             color="#ff8205", lw=2, ls="-", label=f"neural total, quantized ({quant_ovh/1e6:.1f} MB)")
    ax2.axhline(1.946, color="gray", ls="--", label="gzip-6 (val)")
    ax2.axhline(1.511, color="seagreen", ls="--", label="xz-6 (val)")
    for bpb0, ov, c in ((neural["val"]["bits_per_byte"], dense_ovh, "darkred"),
                        (neural_q["val"]["bits_per_byte"], quant_ovh, "#ff8205")):
        cross = 8 * ov / (1.511 - bpb0)
        ax2.axvline(cross / 1e6, color=c, ls=":", lw=1)
        ax2.text(cross / 1e6 * 1.1, 1.6, f"xz crossover ≈{cross/1e6:.0f} MB", fontsize=7.5, color=c)
    ax2.set_xscale("log")
    ax2.set_xlabel("corpus size (MB)")
    ax2.set_ylabel("total BPB (incl. model)")
    ax2.set_title("Model cost amortization")
    ax2.legend(fontsize=7.5)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
