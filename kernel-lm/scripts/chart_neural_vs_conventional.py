#!/usr/bin/env python3
"""Neural vs conventional compression chart.

Panel 1: per-corpus bars for gzip-6 / zstd-6 / xz-6, plus the neural model as a
STACKED bar = conditional BPB (range-coded stream) + model-weights overhead
(8 * model_bytes / corpus_bytes), i.e. TOTAL compression including the .pt.

Panel 2: total neural BPB vs corpus size (log scale) with the model cost
amortizing, against flat gzip/xz lines — shows the crossover point.
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
        for _corpus, corpora in payload["results"].items():
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


def model_bytes_from_checkpoint(ckpt: Path) -> int:
    import torch
    sys.path.insert(0, str(ROOT))
    from model import Transformer, ModelConfig
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    m = Transformer(mc)
    n = sum(p.numel() for p in m.parameters())  # tied embeddings counted once
    return n * 4  # fp32 state_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/small-16k/ckpt_016000.pt"))
    ap.add_argument("--model-bytes", type=int, default=None,
                    help="override model weight bytes (default: fp32 state_dict of checkpoint)")
    ap.add_argument("--tokenizer-bytes", type=int,
                    default=(ROOT / "data/tokenizer.json").stat().st_size)
    ap.add_argument("--out", default=str(ROOT / "results/charts/neural_vs_conventional.png"))
    args = ap.parse_args()

    model_bytes = args.model_bytes or model_bytes_from_checkpoint(Path(args.checkpoint))
    overhead_bytes = model_bytes + args.tokenizer_bytes
    print(f"model weights (fp32 state_dict): {model_bytes/1e6:.1f} MB + tokenizer {args.tokenizer_bytes/1e3:.0f} KB")

    neural = load_model_bpb([ROOT / "results/eval/small16k_kernel.json",
                             ROOT / "results/eval/small16k_external.json"])
    comp = load_compressor_bpb([ROOT / "results/benchmarks/compressors_kernel.json",
                                ROOT / "results/benchmarks/compressors_ext.json"])
    names = [n for n in neural if n in comp]
    names.sort(key=lambda n: neural[n]["bits_per_byte"])

    cond = [neural[n]["bits_per_byte"] for n in names]
    # model overhead in BPB for each corpus's byte count (from eval meta)
    ovh = [8 * overhead_bytes / neural[n]["bytes"] if neural[n].get("bytes") else 0.0 for n in names]
    gz = [comp[n].get("gzip-6", np.nan) for n in names]
    zs = [comp[n].get("zstd-6", np.nan) for n in names]
    xz = [comp[n].get("xz-6", np.nan) for n in names]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(max(14, len(names) * 0.85), 6),
                                  gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - 1.5 * w, cond, w, color="crimson", label="neural: stream (conditional)")
    ax.bar(x - 1.5 * w, ovh, w, bottom=cond, color="darkred", hatch="//",
           edgecolor="white", label=f"neural: + model ({model_bytes/1e6:.0f} MB) & tokenizer")
    ax.bar(x - 0.5 * w, gz, w, color="gray", label="gzip-6")
    ax.bar(x + 0.5 * w, zs, w, color="steelblue", label="zstd-6")
    ax.bar(x + 1.5 * w, xz, w, color="seagreen", label="xz-6")
    for xi, c, o in zip(x, cond, ovh):
        ax.text(xi - 1.5 * w, c + o + 0.05, f"{c+o:.2f}", ha="center", fontsize=7, color="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.1, max(cond[i] + ovh[i] for i in range(len(names))) * 3)
    ax.set_ylabel("bits per byte (log scale, lower = better)")
    ax.set_title(f"Neural vs conventional — neural bar is STACKED with model size\n"
                 f"(small-16k, 12.2M params; total = stream + {overhead_bytes/1e6:.1f} MB model+tokenizer)")
    ax.legend(fontsize=8)

    # panel 2: total BPB vs corpus size
    sizes = np.logspace(6, 10.5, 200)  # 1 MB .. ~3 GB
    total = 0.7075 + 8 * overhead_bytes / sizes
    ax2.plot(sizes / 1e6, total, color="darkred", lw=2, label="neural total (incl. model)")
    ax2.axhline(1.946, color="gray", ls="--", label="gzip-6 (val)")
    ax2.axhline(1.802, color="steelblue", ls="--", label="zstd-6 (val)")
    ax2.axhline(1.511, color="seagreen", ls="--", label="xz-6 (val)")
    cross = 8 * overhead_bytes / (1.511 - 0.7075)
    ax2.axvline(cross / 1e6, color="black", ls=":", lw=1)
    ax2.text(cross / 1e6 * 1.15, 0.9, f"xz crossover\n≈{cross/1e6:.0f} MB", fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("corpus size (MB)")
    ax2.set_ylabel("total BPB")
    ax2.set_title("Model cost amortization")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
