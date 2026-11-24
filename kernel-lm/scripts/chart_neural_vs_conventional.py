#!/usr/bin/env python3
"""Bar chart: neural model BPB vs conventional compressors per corpus."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def load_model_bpb(eval_jsons: list[Path]) -> dict[str, float]:
    out = {}
    for ej in eval_jsons:
        with open(ej) as f:
            payload = json.load(f)
        for corpus, corpora in payload["results"].items():
            for name, r in corpora.items():
                if r.get("bits_per_byte"):
                    out[name] = r["bits_per_byte"]
    return out


def load_compressor_bpb(jsons: list[Path]) -> dict[str, dict[str, float]]:
    out = {}
    for cj in jsons:
        with open(cj) as f:
            d = json.load(f)
        for corpus, res in d.items():
            out[corpus] = {k: v["bits_per_byte"] for k, v in res.items()}
    return out


def main():
    model = load_model_bpb([ROOT / "results/eval/eval_kernel_latest.json",
                            ROOT / "results/eval/eval_eval_tokens_latest.json"])
    # external eval may be saved under a different name; glob
    for ej in (ROOT / "results/eval").glob("*.json"):
        if ej.name in ("eval_kernel_latest.json",):
            continue
        try:
            model.update(load_model_bpb([ej]))
        except Exception:
            pass
    comp = load_compressor_bpb([ROOT / "results/benchmarks/compressors_kernel.json",
                                ROOT / "results/benchmarks/compressors_ext.json"])
    # pick corpora present in both
    names = [n for n in model if n in comp]
    names.sort()
    if not names:
        print("no overlapping corpora"); return
    neural = [model[n] for n in names]
    gz = [comp[n].get("gzip-6", float("nan")) for n in names]
    zs = [comp[n].get("zstd-6", float("nan")) for n in names]
    xz = [comp[n].get("xz-6", float("nan")) for n in names]
    import numpy as np
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
    ax.bar(x - 1.5 * w, neural, w, label="neural (tiny-pilot)", color="crimson")
    ax.bar(x - 0.5 * w, gz, w, label="gzip-6")
    ax.bar(x + 0.5 * w, zs, w, label="zstd-6")
    ax.bar(x + 1.5 * w, xz, w, label="xz-6")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("bits per byte")
    ax.set_title("Neural vs conventional compression (BPB, lower is better)")
    ax.legend()
    fig.tight_layout()
    out = ROOT / "results/charts/neural_vs_conventional.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
