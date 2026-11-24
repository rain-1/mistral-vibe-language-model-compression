#!/usr/bin/env bash
# Evaluate a checkpoint on every corpus. Usage: eval_all.sh <checkpoint> <tag>
set -e
cd "$(dirname "$0")/.."
PY=../.venv/bin/python3
CK=$1
TAG=$2
$PY evaluation/perplexity.py --checkpoint "$CK" --corpora data/tokens \
    --corpus-names val holdout_subsystem holdout_driver --max-files 2500 --batch 8 \
    --out "results/eval/${TAG}_kernel.json"
$PY evaluation/perplexity.py --checkpoint "$CK" --corpora data/eval_tokens --batch 8 \
    --out "results/eval/${TAG}_external.json"
$PY evaluation/perplexity.py --checkpoint "$CK" --corpora data/domain_tokens --batch 8 \
    --out "results/eval/${TAG}_domains.json"
echo "EVAL_ALL_DONE $TAG"
