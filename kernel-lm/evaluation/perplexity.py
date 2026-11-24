#!/usr/bin/env python3
"""Evaluate a checkpoint on tokenized corpora with proper metric accounting.

Metrics per corpus:
  - total NLL in nats over all predictable tokens (every token except the
    first of each file)
  - cross-entropy (nats/token), perplexity, bits/token
  - bits/byte (BPB) = NLL / (ln2 * original_bytes)
  - ideal compressed size = 8/BPB ratio vs original

Documents are evaluated independently: no context crosses file boundaries,
no padding tokens, first token of each file is excluded from the loss.
Windows slide within a file with stride = context (no overlap).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@torch.no_grad()
def eval_token_array(model, bin_path: Path, offsets: list, ctx: int, device: str,
                     max_files: int | None = None, batch: int = 4,
                     meta_files: list | None = None) -> dict:
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    model.eval()
    total_nll = 0.0
    total_toks = 0
    total_bytes = 0
    files = offsets[:max_files] if max_files else offsets
    if meta_files is not None:
        total_bytes = sum(m["bytes"] for m in meta_files[:len(files)])
    # group file segments into batches of windows
    windows = []  # (file_idx, start, end)
    for fi, (s, e) in enumerate(files):
        n = e - s
        if n < 2:
            continue
        for wstart in range(0, n - 1, ctx):
            wend = min(wstart + ctx + 1, n)  # +1 for target shift
            if wend - wstart < 2:
                break
            windows.append((fi, s + wstart, s + wend))
    i = 0
    while i < len(windows):
        chunk = windows[i:i + batch]
        i += batch
        # pad windows to same length with -100 targets (ignored in loss)
        L = max(w[2] - w[1] - 1 for w in chunk)  # input length = seg-1
        x = torch.full((len(chunk), L), 0, dtype=torch.long)  # pad id 0, masked below
        y = torch.full((len(chunk), L), -100, dtype=torch.long)
        for r, (fi, a, b) in enumerate(chunk):
            seg = np.asarray(data[a:b], dtype=np.int64)
            x[r, :len(seg) - 1] = torch.from_numpy(seg[:-1])
            y[r, :len(seg) - 1] = torch.from_numpy(seg[1:])
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        logp = torch.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        mask = (y != -100)
        total_nll += float((nll * mask).sum())
        total_toks += int(mask.sum())
    ce = total_nll / max(total_toks, 1)
    return {
        "files": len(files), "tokens": total_toks, "nll_nats": total_nll,
        "bytes": total_bytes,
        "cross_entropy_nats": ce, "perplexity": math.exp(ce),
        "bits_per_token": ce / math.log(2),
    }


def eval_corpus(model, corpus_dir: Path, ctx: int, device: str,
                max_files: int | None = None, batch: int = 4) -> dict:
    """corpus_dir contains <name>.bin + <name>.offsets.json + <name>.meta.json(bytes)"""
    results = {}
    for bin_path in sorted(corpus_dir.glob("*.bin")):
        name = bin_path.stem
        off_path = corpus_dir / f"{name}.offsets.json"
        meta_path = corpus_dir / f"{name}.meta.json"
        with open(off_path) as f:
            offsets = json.load(f)["offsets"]
        meta_files = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta_files = json.load(f)["files"]
        r = eval_token_array(model, bin_path, offsets, ctx, device, max_files, batch, meta_files)
        bpb = r["nll_nats"] / (math.log(2) * r["bytes"]) if r["bytes"] else None
        r["bits_per_byte"] = bpb
        r["ideal_ratio"] = 8.0 / bpb if bpb else None
        results[name] = r
        print(f"  {name:24s} files={r['files']:5d} toks={r['tokens']:9d} "
              f"ce={r['cross_entropy_nats']:.4f} ppl={r['perplexity']:8.2f} "
              f"bpt={r['bits_per_token']:.3f} BPB={bpb:.4f} ratio={r['ideal_ratio']:.2f}x")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpora", nargs="+", default=[str(DATA / "tokens")])
    ap.add_argument("--corpus-names", nargs="+", default=None,
                    help="only these corpus stems (e.g. val holdout_subsystem)")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    print(f"loaded {args.checkpoint}  params={ck.get('n_params', sum(p.numel() for p in model.parameters()))/1e6:.2f}M "
          f"step={ck['step']} ctx={mc.context_length}")

    all_results = {}
    for corpus in args.corpora:
        cdir = Path(corpus)
        names = args.corpus_names
        if names:
            # build a temp view: evaluate only requested stems
            res = {}
            for n in names:
                bin_path = cdir / f"{n}.bin"
                if not bin_path.exists():
                    continue
                with open(cdir / f"{n}.offsets.json") as f:
                    offsets = json.load(f)["offsets"]
                meta = cdir / f"{n}.meta.json"
                meta_files = None
                if meta.exists():
                    with open(meta) as f:
                        meta_files = json.load(f)["files"]
                r = eval_token_array(model, bin_path, offsets, mc.context_length, device,
                                     args.max_files, args.batch, meta_files)
                bpb = r["nll_nats"] / (math.log(2) * r["bytes"]) if r["bytes"] else None
                r["bits_per_byte"] = bpb
                r["ideal_ratio"] = 8.0 / bpb if bpb else None
                res[n] = r
                print(f"  {n:24s} files={r['files']:5d} toks={r['tokens']:9d} "
                      f"ce={r['cross_entropy_nats']:.4f} ppl={r['perplexity']:8.2f} "
                      f"bpt={r['bits_per_token']:.3f} BPB={bpb:.4f} ratio={r['ideal_ratio']:.2f}x")
            all_results[str(cdir)] = res
        else:
            all_results[str(cdir)] = eval_corpus(model, cdir, mc.context_length, device,
                                                 args.max_files, args.batch)

    out = Path(args.out) if args.out else ROOT / "results" / "eval" / f"eval_{Path(args.checkpoint).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": str(args.checkpoint), "step": ck["step"], "results": all_results}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
