#!/usr/bin/env python3
"""Benchmark conventional compressors (gzip, zstd, xz, bzip2) on evaluation
corpora: report bits/byte, ratio, compress/decompress speed.

Operates on the raw bytes of each corpus (same files as the tokenized eval
corpora, from <name>.meta.json), concatenated per corpus with a 4-byte
little-endian length header per file so boundaries are consistent and
reversible.
"""
from __future__ import annotations

import argparse
import json
import lzma
import bz2
import gzip
import shutil
import struct
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def corpus_bytes(meta_path: Path) -> bytes:
    with open(meta_path) as f:
        meta = json.load(f)
    root = Path(meta["root"])
    buf = bytearray()
    for m in meta["files"]:
        raw = (root / m["path"]).read_bytes()
        buf += struct.pack("<I", len(raw)) + raw
    return bytes(buf)


def bench(name: str, data: bytes, out: dict):
    n = len(data)
    # gzip
    t0 = time.time(); gz = gzip.compress(data, 6); t_c = time.time() - t0
    t0 = time.time(); gzip.decompress(gz); t_d = time.time() - t0
    out["gzip-6"] = _rec(n, len(gz), t_c, t_d)
    # xz
    t0 = time.time(); xz = lzma.compress(data, preset=6); t_c = time.time() - t0
    t0 = time.time(); lzma.decompress(xz); t_d = time.time() - t0
    out["xz-6"] = _rec(n, len(xz), t_c, t_d)
    # bzip2
    t0 = time.time(); bz = bz2.compress(data, 9); t_c = time.time() - t0
    t0 = time.time(); bz2.decompress(bz); t_d = time.time() - t0
    out["bzip2-9"] = _rec(n, len(bz), t_c, t_d)
    # zstd via CLI if available
    zstd = shutil.which("zstd")
    if zstd:
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(data); tmp_in = tf.name
        tmp_out = tmp_in + ".zst"
        t0 = time.time()
        subprocess.run([zstd, "-6", "-q", "-f", "-o", tmp_out, tmp_in], check=True)
        t_c = time.time() - t0
        size = os.path.getsize(tmp_out)
        t0 = time.time()
        subprocess.run([zstd, "-d", "-q", "-f", "-o", tmp_in + ".dec", tmp_out], check=True)
        t_d = time.time() - t0
        assert open(tmp_in + ".dec", "rb").read() == data
        out["zstd-6"] = _rec(n, size, t_c, t_d)
        os.unlink(tmp_in); os.unlink(tmp_out); os.unlink(tmp_in + ".dec")


def _rec(n: int, csize: int, t_c: float, t_d: float) -> dict:
    bpb = csize * 8 / n
    return {"orig_bytes": n, "compressed_bytes": csize, "bits_per_byte": round(bpb, 4),
            "ratio": round(n / csize, 3), "comp_s": round(t_c, 2), "decomp_s": round(t_d, 2),
            "comp_MBps": round(n / 1e6 / t_c, 1), "decomp_MBps": round(n / 1e6 / t_d, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(ROOT / "data" / "eval_tokens"))
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--max-mb", type=float, default=200.0, help="truncate corpus to this many MB")
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmarks" / "compressors.json"))
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    results = {}
    for meta_path in sorted(eval_dir.glob("*.meta.json")):
        name = meta_path.stem.replace(".meta", "")
        if args.names and name not in args.names:
            continue
        print(f"corpus {name} ...")
        data = corpus_bytes(meta_path)
        if len(data) > args.max_mb * 1e6:
            data = data[: int(args.max_mb * 1e6)]
            print(f"  truncated to {len(data)/1e6:.0f} MB")
        out = {}
        bench(name, data, out)
        results[name] = out
        for k, v in out.items():
            print(f"  {k:9s} BPB={v['bits_per_byte']:.4f} ratio={v['ratio']:.3f}x "
                  f"comp={v['comp_MBps']}MB/s decomp={v['decomp_MBps']}MB/s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
