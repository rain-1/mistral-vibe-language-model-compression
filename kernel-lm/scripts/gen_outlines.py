#!/usr/bin/env python3
"""Generate bullet-list outlines for Fimfiction stories using the story model
(few-shot: STORY excerpt -> OUTLINE bullets). Writes outline/story pairs for
outline->story fine-tuning.

Pairs: data/ft/outline_pairs.jsonl  {id, title, outline, story}
Skips stories used in the fimfiction eval corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer, ModelConfig

ROOT = Path(__file__).resolve().parent.parent

FEWSHOT = """STORY: {s1}
OUTLINE:
- {o1}

STORY: {s2}
OUTLINE:
- {o2}

STORY: {s}
OUTLINE:
- """

EX1_S = ("Harry Potter and the Philosopher's Stone, Chapter 1. Eleven-year-old Harry lives in a "
         "cupboard under the stairs with his cruel aunt, uncle and cousin Dudley. On his eleventh "
         "birthday a giant named Hagrid arrives and tells Harry he is a wizard, and that his "
         "parents were killed by the dark wizard Voldemort.")
EX1_O = ("Harry, an abused orphan, lives with the Dursleys in a cupboard under the stairs",
         "Hagrid arrives on Harry's 11th birthday and reveals he is a wizard",
         "Harry learns his parents were killed by Voldemort and he survived as a baby",
         "Hagrid takes Harry shopping in Diagon Alley and buys him school supplies",
         "Harry catches his first train to Hogwarts and meets Ron and Hermione")
EX2_S = ("Sherlock Holmes, Chapter 1. It was 1889 in London. A young woman in mourning visited "
         "221B Baker Street, begging Mr Holmes to help her. Her fiancé had vanished three days "
         "before their wedding, after a strange encounter in an empty house. Holmes examined the "
         "mystery with his friend Dr Watson and quickly deduced that the empty house held the key.")
EX2_O = ("A grieving woman hires Holmes because her fiancé disappeared before their wedding",
         "The fiancé was last seen entering an empty house on the street",
         "Holmes and Watson deduce the empty house is connected to the disappearance",
         "They investigate the house and find hidden clues",
         "Holmes prepares to confront the person responsible")


def parse_outline(text: str) -> str | None:
    lines = [l.strip() for l in text.splitlines()]
    bullets = []
    for l in lines:
        m = re.match(r"^[-*•]\s+(.*)", l)
        if m:
            bullets.append(m.group(1).strip())
        elif bullets and l and not l.startswith("STORY"):
            bullets[-1] += " " + l  # continuation line
        elif bullets:
            break
        if len(bullets) >= 8:
            break
    if not (3 <= len(bullets) <= 8):
        return None
    if any(len(b) < 15 or len(b) > 160 for b in bullets):
        return None
    return "\n".join(f"- {b}" for b in bullets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/story-medium-16k/ckpt_030000.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data/ao3/tokenizer-16k.json"))
    ap.add_argument("--n", type=int, default=12000, help="number of pairs to build")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=str(ROOT / "data/ft/outline_pairs.jsonl"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["config"]["model"])
    model = Transformer(mc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)

    # candidate stories from fimfiction, excluding eval files, 300..4000 bytes
    eval_paths = set()
    meta_p = ROOT / "data/story_eval_tokens/fimfiction.meta.json"
    if meta_p.exists():
        eval_paths = {m["path"] for m in json.load(open(meta_p))["files"]}
    cands = []
    for l in open(ROOT / "data/fimfiction/story_manifest.jsonl"):
        r = json.loads(l)
        if r["path"] in eval_paths:
            continue
        if 300 <= r["bytes"] <= 4000:
            cands.append(r)
    rng = random.Random(args.seed)
    rng.shuffle(cands)
    print(f"{len(cands)} candidate stories; need {args.n} pairs")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with open(out, "w") as f:
        i = 0
        while n_done < args.n and i < len(cands):
            batch = cands[i:i + args.batch]
            i += args.batch
            prompts = []
            for r in batch:
                story = (ROOT / "data/fimfiction/stories" / r["path"]).read_text(
                    encoding="utf-8", errors="surrogateescape")
                excerpt = story[:1200]
                prompts.append(FEWSHOT.format(s1=EX1_S, o1="\n- ".join(EX1_O),
                                             s2=EX2_S, o2="\n- ".join(EX2_O),
                                             s=excerpt))
            ids = tok.encode_batch(prompts)
            maxlen = max(len(e.ids) for e in ids)
            x = torch.full((len(batch), maxlen), tok.token_to_id("<|file|>"), dtype=torch.long)
            for b, e in enumerate(ids):
                x[b, :len(e.ids)] = torch.tensor(e.ids)
            x = x.to(device)
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature,
                               top_k=40, repetition_penalty=1.2)
            for b, r in enumerate(batch):
                gen = tok.decode(y[b, maxlen:].tolist(), skip_special_tokens=True)
                outline = parse_outline(gen)
                if outline is None:
                    continue
                story = (ROOT / "data/fimfiction/stories" / r["path"]).read_text(
                    encoding="utf-8", errors="surrogateescape")
                f.write(json.dumps({"id": r["story_id"], "title": r.get("title", ""),
                                    "outline": outline, "story": story}, ensure_ascii=False) + "\n")
                n_done += 1
            if n_done % 400 < args.batch:
                print(f"{n_done}/{args.n} pairs", flush=True)
    print(f"wrote {n_done} pairs -> {out}")


if __name__ == "__main__":
    main()
