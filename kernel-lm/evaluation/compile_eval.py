#!/usr/bin/env python3
"""Compilation-success evaluation (spec section 11).

For sampled held-out val functions: keep the signature line as prompt,
greedily generate a body, wrap it in a minimal harness with common kernel-ish
stub types, and run `gcc -fsyntax-only`. Reports syntax-valid rate.
This is deliberately approximate — kernel code needs the full kernel headers,
so we only measure *syntactic* plausibility of generated bodies, not full
compilation against the kernel tree.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent

HARNESS = """typedef unsigned char u8; typedef unsigned short u16; typedef unsigned int u32;
typedef unsigned long long u64; typedef signed char s8; typedef short s16; typedef int s32; typedef long long s64;
typedef unsigned long size_t; typedef long ssize_t; typedef _Bool bool;
#define NULL ((void*)0)
struct list_head { struct list_head *next, *prev; };
static inline void INIT_LIST_HEAD(struct list_head *l){l->next=l;l->prev=l;}
{body}
int main(void){return 0;}
"""

FUNC = re.compile(r"^[A-Za-z_][A-Za-z0-9_ \*]*\([^;]*$", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tokenizer.json"))
    ap.add_argument("--max-files", type=int, default=40)
    ap.add_argument("--funcs-per-file", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not shutil.which("gcc"):
        print("gcc not available"); return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)

    # sample val files (raw text)
    with open(ROOT / "data/manifest.jsonl") as f:
        recs = [json.loads(l) for l in f if json.loads(l)["split"] == "val"]
    import random
    rng = random.Random(0)
    rng.shuffle(recs)
    n_ok = n_tot = 0
    examples = []
    for r in recs[: args.max_files]:
        text = (ROOT / "data/linux-v6.6" / r["path"]).read_text(errors="replace")
        # find function starts: a line ending with '{' preceded by signature
        lines = text.split("\n")
        cand = []
        for i in range(1, len(lines) - 2):
            if lines[i].rstrip().endswith("{") and "(" in lines[i - 1] and not lines[i - 1].startswith(("if", "for", "while", "switch", "else", "do", "#")):
                cand.append(i)
        rng.shuffle(cand)
        for cut in cand[: args.funcs_per_file]:
            sig = lines[cut - 1].strip()
            if len(sig) < 8 or len(sig) > 120:
                continue
            prompt = "\n".join(lines[:cut]) + "\n"
            ids = tok.encode(prompt).ids
            x = torch.tensor([ids[-mc.context_length:]], dtype=torch.long, device=device)
            out = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=50)
            gen = tok.decode(out[0, x.shape[1]:].tolist(), skip_special_tokens=False)
            # cut at first complete-looking end: stop after a line that is just "}"
            body_lines = []
            for ln in gen.split("\n"):
                body_lines.append(ln)
                if ln.strip() == "}" and len(body_lines) > 1:
                    break
            body = sig + "\n" + "\n".join(body_lines)
            n_tot += 1
            with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as tf:
                tf.write(HARNESS.format(body=body))
                path = tf.name
            res = subprocess.run(["gcc", "-fsyntax-only", "-w", path],
                                 capture_output=True, timeout=20)
            ok = res.returncode == 0
            n_ok += ok
            if len(examples) < 5:
                examples.append({"file": r["path"], "ok": ok, "body": body[:400]})
            Path(path).unlink(missing_ok=True)
    res = {"checkpoint": str(args.checkpoint), "n": n_tot, "syntax_valid": n_ok,
           "rate": n_ok / max(n_tot, 1), "examples": examples}
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))
    out = Path(args.out) if args.out else ROOT / "results/eval" / f"compile_{Path(args.checkpoint).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
