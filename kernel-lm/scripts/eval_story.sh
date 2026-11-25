#!/usr/bin/env bash
# Evaluate a story-model checkpoint: perplexity/BPB on AO3 val+test and the
# external corpora (fimfiction, pg19), exact range-coder compression on a
# sample of raw files, and a batch of free prose samples.
set -x
cd "$(dirname "$0")/.."
PY=../.venv/bin/python3
CK=${1:-checkpoints/story-medium-16k/ckpt_030000.pt}
TOK=data/ao3/tokenizer-16k.json
TAG=$(basename "$CK" .pt)

$PY -u evaluation/perplexity.py --checkpoint "$CK" \
  --corpora data/ao3/story_tokens data/story_eval_tokens \
  --corpus-names val test fimfiction pg19-test pg19-val \
  --batch 4 --out results/eval/story-${TAG}.json

# exact arithmetic-coder compression on a sample of raw AO3 test + fimfiction files
$PY -u evaluation/compression.py --checkpoint "$CK" --tokenizer "$TOK" \
  --files $(python3 -c "
import json
recs=[json.loads(l) for l in open('data/ao3/story_manifest.jsonl') if json.loads(l)['split']=='test']
print(' '.join('data/ao3/stories/'+r['path'] for r in recs[:40]))") \
  --model-bytes $(python3 -c "import torch; ck=torch.load('$CK',map_location='cpu',weights_only=False); print(sum(p.numel() for p in ck['model'].values())*4)") \
  2>&1 | tee results/eval/story-${TAG}-compression.txt

$PY -u scripts/sample_stories.py --checkpoint "$CK" --tokenizer "$TOK" \
  --temperature 0.8 --top-k 40 --repetition-penalty 1.2 \
  --max-new-tokens 300 --real-prefixes 8 \
  --out results/samples/story-${TAG}.txt
