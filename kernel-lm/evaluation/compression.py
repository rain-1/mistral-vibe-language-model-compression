#!/usr/bin/env python3
"""Lossless range coder driven by model probabilities + exact-reconstruction test.

Implements a 32-bit range coder with 16-bit cumulative frequency tables
(Subbotin-style carry handling). Encodes a token stream to a byte string and
decodes it back; assert byte-for-byte equality with the original file bytes.

Reports:
  - ideal compressed size = NLL / 8 bytes (theoretical)
  - actual compressed size = range-coder output bytes
  - bits/byte, compression ratio, and model+tokenizer overhead accounting
    (conditional vs total compression).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent
TOP = 1 << 32
BOT = 1 << 16  # cumulative precision


class RangeCoder:
    """Carry-less 32-bit range coder (Subbotin) with 16-bit cumulative tables."""

    def __init__(self):
        self.low = 0
        self.range = TOP - 1
        self.out = bytearray()
        self.dlow = 0
        self.drange = TOP - 1
        self.dpos = 0
        self.ddata = b""
        self.dval = 0

    # ---------- encoder ----------
    def encode(self, cum_lo: int, cum_hi: int):
        assert 0 <= cum_lo < cum_hi <= BOT
        r = self.range // BOT
        self.low += cum_lo * r
        self.range = (cum_hi - cum_lo) * r
        self._norm_enc()

    def _norm_enc(self):
        while True:
            if (self.low ^ (self.low + self.range)) < (1 << 24):
                pass
            elif self.range < (1 << 16):
                self.range = (-self.low) & 0xFFFF
            else:
                break
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & 0xFFFFFFFF
            self.range = (self.range << 8) & 0xFFFFFFFF

    def finish_encode(self) -> bytes:
        for _ in range(4):
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & 0xFFFFFFFF
        return bytes(self.out)

    # ---------- decoder ----------
    def start_decode(self, data: bytes):
        self.ddata = data
        self.dpos = 0
        self.dval = 0
        for _ in range(4):
            self.dval = ((self.dval << 8) | self._next_byte()) & 0xFFFFFFFF

    def _next_byte(self) -> int:
        if self.dpos < len(self.ddata):
            b = self.ddata[self.dpos]
            self.dpos += 1
            return b
        return 0

    def decode(self, cum_freq: np.ndarray) -> int:
        """cum_freq: cumulative table, len = vocab+1, strictly increasing, last == BOT"""
        r = self.drange // BOT
        target = (self.dval - self.dlow) // r
        idx = int(np.searchsorted(cum_freq, target, side="right")) - 1
        if idx < 0:
            idx = 0
        if idx >= len(cum_freq) - 1:
            idx = len(cum_freq) - 2
        lo, hi = int(cum_freq[idx]), int(cum_freq[idx + 1])
        self.dlow += lo * r
        self.drange = (hi - lo) * r
        self._norm_dec()
        return idx

    def _norm_dec(self):
        while True:
            if (self.dlow ^ (self.dlow + self.drange)) < (1 << 24):
                pass
            elif self.drange < (1 << 16):
                self.drange = (-self.dlow) & 0xFFFF
            else:
                break
            self.dval = ((self.dval << 8) | self._next_byte()) & 0xFFFFFFFF
            self.dlow = (self.dlow << 8) & 0xFFFFFFFF
            self.drange = (self.drange << 8) & 0xFFFFFFFF


def probs_to_cumfreq(probs: np.ndarray, vocab: int) -> np.ndarray:
    """Convert a probability vector to a 16-bit cumulative table (sum == BOT).

    Reserve 1 count per symbol so every symbol stays encodable:
    freq = floor(p * (BOT - vocab)) + 1, then fix float-rounding residue.
    """
    p = np.asarray(probs, dtype=np.float64)
    p = np.maximum(p, 0.0)
    s = p.sum()
    if s <= 0:
        p = np.full(vocab, 1.0 / vocab)
    else:
        p /= s
    scale = BOT - vocab
    freq = (p * scale).astype(np.int64) + 1
    diff = BOT - int(freq.sum())
    if diff > 0:
        freq[int(np.argmax(p))] += diff
    elif diff < 0:
        for j in np.argsort(-freq):
            if diff == 0:
                break
            take = min(int(freq[j]) - 1, -diff)
            freq[j] -= take
            diff += take
    cum = np.zeros(vocab + 1, dtype=np.int64)
    np.cumsum(freq, out=cum[1:])
    cum[-1] = BOT
    return cum


@torch.no_grad()
def model_cumfreq_stream(model, ids: list[int], ctx: int, device: str):
    """Yield (cumfreq, true_id) per position, sliding a context window."""
    n = len(ids)
    for start in range(0, n - 1, ctx):
        end = min(start + ctx + 1, n)
        seg = ids[start:end]
        x = torch.tensor([seg[:-1]], dtype=torch.long, device=device)
        logits, _ = model(x)
        logp = torch.log_softmax(logits[0].float(), dim=-1).cpu().numpy()
        for i in range(len(seg) - 1):
            yield probs_to_cumfreq(np.exp(logp[i]), logp.shape[-1]), seg[i + 1]


def compress_file(model, tok: Tokenizer, raw: bytes, ctx: int, device: str) -> dict:
    ids = tok.encode(raw.decode("utf-8", errors="surrogateescape")).ids
    coder = RangeCoder()
    nll = 0.0
    for cum, true_id in model_cumfreq_stream(model, ids, ctx, device):
        lo, hi = int(cum[true_id]), int(cum[true_id + 1])
        coder.encode(lo, hi)
        nll += -math.log(max((hi - lo) / BOT, 1e-12))
    comp = coder.finish_encode()
    # decode and verify
    dec = RangeCoder()
    dec.start_decode(comp)
    # first token is given (not predicted); decode the remaining len(ids)-1
    out_ids = [ids[0]]
    cum_stream = model_cumfreq_stream(model, ids, ctx, device)
    for _ in range(len(ids) - 1):
        cum, true_id = next(cum_stream)
        out_ids.append(dec.decode(cum))
    dec_raw = tok.decode(out_ids, skip_special_tokens=False).encode("utf-8", errors="surrogateescape")
    exact = (dec_raw == raw)
    return {
        "orig_bytes": len(raw), "tokens": len(ids), "ideal_bytes": nll / (8 * math.log(2)),
        "actual_bytes": len(comp), "bits_per_byte": len(comp) * 8 / len(raw),
        "exact_reconstruction": exact,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    ap.add_argument("--model-bytes", type=int, default=0, help="model+tokenizer storage bytes for total-compression accounting")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)

    results = []
    for f in args.files:
        raw = Path(f).read_bytes()
        r = compress_file(model, tok, raw, mc.context_length, device)
        r["file"] = f
        results.append(r)
        print(f"{f}: orig={r['orig_bytes']}B tokens={r['tokens']} "
              f"ideal={r['ideal_bytes']:.0f}B actual={r['actual_bytes']}B "
              f"BPB={r['bits_per_byte']:.4f} exact={r['exact_reconstruction']}")
        assert r["exact_reconstruction"], "RECONSTRUCTION FAILED"
    print("ALL EXACT")
    tot_orig = sum(r["orig_bytes"] for r in results)
    tot_act = sum(r["actual_bytes"] for r in results)
    tot_ideal = sum(r["ideal_bytes"] for r in results)
    print(f"TOTAL orig={tot_orig} ideal={tot_ideal:.0f} actual={tot_act} "
          f"BPB={tot_act*8/tot_orig:.4f} ratio={tot_orig/tot_act:.3f}x")
    if args.model_bytes:
        print(f"total incl. model+tokenizer ({args.model_bytes} B): "
              f"{(tot_act + args.model_bytes)*8/tot_orig:.4f} BPB, "
              f"ratio={tot_orig/(tot_act+args.model_bytes):.3f}x")


if __name__ == "__main__":
    main()
