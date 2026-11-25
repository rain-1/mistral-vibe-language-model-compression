#!/usr/bin/env python3
"""Full chart pack for the story (AO3) pretraining run.

Outputs into results/charts/:
  story_full_loss.png            train/val loss + val PPL vs training tokens
  story_full_progress.png        val BPB over training tokens
  story_full_bpb_corpora.png     neural BPB vs gzip/zstd/xz per held-out corpus
  story_full_amortization.png    total BPB (incl. 169MB model) vs corpus size + xz crossover
  story_full_rep.png             repeated-5gram fraction vs repetition penalty (samples)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CH = ROOT / "results" / "charts"
LOG = ROOT / "results/story-medium-16k/log.jsonl"
EV = ROOT / "results/eval/story-ckpt_030000.json"
COMP = [ROOT / "results/benchmarks/compressors_stories.json",
        ROOT / "results/benchmarks/compressors_story_eval.json"]
MODEL_BYTES = 168_930_081  # fp32 weights + tokenizer
BPT = 3.938  # bytes per token on AO3 prose (measured)


def load_log():
    steps, toks, loss, vs, vl = [], [], [], [], []
    for line in open(LOG):
        r = json.loads(line)
        if "loss" in r and r.get("step", 0) > 0:
            steps.append(r["step"]); loss.append(r["loss"]); toks.append(r["tokens"] / 1e6)
        if "val_loss" in r:
            vs.append(r["step"]); vl.append(r["val_loss"])
    vtoks = [toks[-1] * s / steps[-1] for s in vs]
    return toks, loss, vtoks, vl


def load_eval():
    d = json.load(open(EV))["results"]
    out = {}
    for _c, corpora in d.items():
        for k, v in corpora.items():
            out[k] = v
    return out


def load_comp():
    comp, nbytes = {}, {}
    for p in COMP:
        for corpus, res in json.load(open(p)).items():
            comp[corpus] = {k: v["bits_per_byte"] for k, v in res.items()}
            nbytes[corpus] = res["gzip-6"]["orig_bytes"]
    return comp, nbytes


def main():
    CH.mkdir(parents=True, exist_ok=True)
    toks, loss, vtoks, vl = load_log()
    ev = load_eval()
    comp, nbytes = load_comp()
    bpb = lambda ce: ce / 0.6931 / BPT

    # ---- 1: loss + ppl
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.plot(toks, loss, lw=1, color="#ff8205", label="train")
    a1.plot(vtoks, vl, "o-", ms=3, color="crimson", label="val (AO3 held-out)")
    a1.set_xlabel("training tokens (M)"); a1.set_ylabel("cross-entropy (nats)")
    a1.set_title("story-medium-16k loss"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a2.plot(vtoks, [np.exp(x) for x in vl], "o-", ms=3, color="darkred")
    a2.set_xlabel("training tokens (M)"); a2.set_ylabel("val perplexity")
    a2.set_title(f"val PPL (final {np.exp(vl[-1]):.1f})"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(CH / "story_full_loss.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---- 2: val BPB progress
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(vtoks, [bpb(x) for x in vl], "o-", ms=4, color="#ff8205")
    ax.axhline(2.202, color="seagreen", ls="--", label="xz-6 on AO3 val (2.20 BPB)")
    ax.axhline(2.983, color="gray", ls="--", label="gzip-6 (2.98)")
    ax.set_xlabel("training tokens (M)"); ax.set_ylabel("val BPB")
    ax.set_title(f"story-medium-16k val BPB over training (final {bpb(vl[-1]):.3f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(CH / "story_full_progress.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---- 3: BPB per corpus vs conventional
    names = ["val", "test", "fimfiction", "pg19-test", "pg19-val"]
    labels = [f"{n}\n({nbytes.get(n, ev[n].get('bytes', 0))/1e6:.0f} MB)" for n in names]
    neural = [ev[n]["bits_per_byte"] for n in names]
    x = np.arange(len(names)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * w, [8.0] * len(names), w, color="#3a3a44", label="original (8 BPB)")
    ax.bar(x - 0.5 * w, neural, w, color="crimson", label="neural (conditional)")
    ax.bar(x + 0.5 * w, [comp[n]["gzip-6"] for n in names], w, color="gray", label="gzip-6")
    ax.bar(x + 1.5 * w, [comp[n]["zstd-6"] for n in names], w, color="steelblue", label="zstd-6")
    ax.bar(x + 2.5 * w, [comp[n]["xz-6"] for n in names], w, color="seagreen", label="xz-6")
    for xi, v in zip(x, neural):
        ax.text(xi - 0.5 * w, v + 0.06, f"{v:.2f}", ha="center", fontsize=8, color="darkred")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("bits per byte (lower = better)")
    ax.set_title("story model BPB by held-out corpus vs conventional compressors")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(CH / "story_full_bpb_corpora.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---- 4: amortization / crossover
    sizes = np.logspace(5, 11, 300)
    cond = ev["val"]["bits_per_byte"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(sizes / 1e6, cond + 8 * MODEL_BYTES / sizes, color="darkred", lw=2,
            label=f"neural total (cond {cond:.2f} + 169 MB model)")
    ax.axhline(2.202, color="seagreen", ls="--", label="xz-6 (2.20 BPB)")
    ax.axhline(2.983, color="gray", ls="--", label="gzip-6 (2.98)")
    cross = 8 * MODEL_BYTES / (2.202 - cond)
    ax.axvline(cross / 1e6, color="darkred", ls=":", lw=1.5)
    ax.text(cross / 1e6 * 1.15, 2.35, f"xz crossover ≈ {cross/1e6:.0f} MB", fontsize=8, color="darkred")
    ax.set_xscale("log"); ax.set_xlabel("corpus size (MB)"); ax.set_ylabel("total BPB incl. model")
    ax.set_title("story model: model cost amortization vs corpus size")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(CH / "story_full_amortization.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---- 5: repetition penalty analysis on samples
    def rep_ngram(path, n=4):
        txt = open(path).read()
        samples = re.split(r"===== sample \d+ .*?=====\n", txt)[1:]
        fracs = []
        for s in samples:
            words = s.split()
            grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
            if grams:
                fracs.append(1 - len(set(grams)) / len(grams))
        return np.mean(fracs) * 100 if fracs else 0
    rps, vals = [], []
    for rp in ("1.0", "1.2", "1.5"):
        p = ROOT / f"results/samples/story_rp{rp}.txt"
        if p.exists():
            rps.append(float(rp)); vals.append(rep_ngram(p, 4))
    if rps:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(rps, vals, "o-", color="#ff8205", lw=2, ms=7)
        for r, v in zip(rps, vals):
            ax.annotate(f"{v:.1f}%", (r, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
        ax.set_xlabel("repetition penalty"); ax.set_ylabel("repeated word-4-grams (%)")
        ax.set_title("sampling: repetition penalty vs repeated 4-grams (story model)")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(CH / "story_full_rep.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        print("rep-penalty:", dict(zip(rps, [round(v, 2) for v in vals])))

    print("charts written to", CH)


if __name__ == "__main__":
    main()
