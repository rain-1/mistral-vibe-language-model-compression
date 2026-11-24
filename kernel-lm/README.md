# kernel-lm

Small decoder-only transformer trained from scratch on Linux kernel C source,
studying **neural compression** (bits/byte) and code generation under a hard
hardware budget: **NVIDIA RTX 2060 (6 GB VRAM), Xeon E3-1245 v3, 16 GB RAM**.

Primary metric: **bits per byte (BPB)**, not perplexity alone.

## Layout

```
kernel-lm/
  configs/            YAML model+training configs (micro/tiny/small/medium/large)
  data/               (gitignored) kernel checkout, manifest, token arrays, eval corpora
  scripts/
    prepare_data.py   manifest + contamination-aware splits
    train_tokenizer.py  byte-level BPE (4096) trained on train split only
    tokenize_data.py    memmap uint16 token arrays + per-file offsets
    encode_corpus.py    encode arbitrary repo into eval token arrays
    benchmark_gpu.py    VRAM/throughput sweep for 6 GB fitting
    benchmark_compressors.py  gzip/zstd/xz/bzip2 BPB baselines
    overfit_test.py     correctness test (must overfit a small batch)
    make_charts.py      loss/BPB/benchmark charts
  model/transformer.py  RMSNorm + RoPE + causal SDPA + SwiGLU + tied embeddings
  train.py            FP16 AMP training loop, checkpoint/resume, JSONL (+optional wandb)
  generate.py         sampling from a checkpoint
  evaluation/perplexity.py  NLL/PPL/BPT/BPB with correct doc boundaries
  results/  checkpoints/  experiments/
```

## Quickstart

```bash
uv pip install --python ../.venv/bin/python3 -r requirements.txt
# data
python3 scripts/prepare_data.py
python3 scripts/train_tokenizer.py
python3 scripts/tokenize_data.py
# verify pipeline
python3 scripts/overfit_test.py
# benchmark hardware limits
python3 scripts/benchmark_gpu.py
# train
python3 train.py --config configs/tiny.yaml --run-name tiny-pilot
# evaluate BPB on all corpora
python3 evaluation/perplexity.py --checkpoint checkpoints/<run>/latest.pt \
    --corpora data/tokens data/eval_tokens
```

## Hardware notes (measured)

See `results/benchmarks/gpu_benchmark.json`. RTX 2060 = Turing (sm_75): no
native BF16, so training uses **FP16 autocast + GradScaler**. FlashAttention is
not used; attention is PyTorch SDPA (memory-efficient kernel on Ampere+; on
Turing it falls back to the math/efficient path — benchmarked, still OK).

## Data & contamination policy

- Kernel: torvalds/linux pinned at `ffc253263a1375a65fa6c9f62a893e9767fbebfa` (v6.6),
  shallow sparse checkout of `*.c`/`*.h` only (56,946 files, 1.22 GB).
- Splits are **never token-level random**: held-out subsystems (`mm fs net
  crypto security sound virt`), held-out driver families (`drivers/{gpu,net,
  media,staging}`), plus a seeded **file-level** val split grouped by content
  hash so exact duplicates never straddle splits (166 exact dup files found).
- Tokenizer trained on the train split only; verified lossless byte round-trip.
- External repos (musl, busybox, sqlite, nginx, redis, lua, xv6, doom, git,
  cpython) pinned to explicit revisions in `data/eval/*/REVISION.txt`, used for
  evaluation only.

## Metrics

`evaluation/perplexity.py` computes, per corpus: NLL (nats), cross-entropy,
perplexity, bits/token, **bits/byte**, ideal compressed size and ratio.
Documents are evaluated independently (no cross-file context, no padding in the
loss, first token of each file excluded).
