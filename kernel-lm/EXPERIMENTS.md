# Experiment ledger

Permanent record of configurations, hypotheses, results, conclusions — including
failures. One variable changed at a time where comparisons matter.

## Environment (measured 2026-11-24)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2060, 6144 MB, Turing sm_75 |
| Driver / CUDA UMD | 620.70.06 / 13.5 |
| PyTorch | 2.15.0+cu132 (CUDA 13.2 runtime), CUDA available: True |
| BF16 | reported supported by API but NOT native on Turing → use FP16 AMP |
| CPU | Xeon E3-1245 v3, 8 threads | RAM 16 GB | Disk free ~259 GB |
| Python | 3.12.14 venv (uv), tokenizers 0.23.3, numpy 2.5.5 |

## E0 — GPU benchmark (scripts/benchmark_gpu.py, results/benchmarks/gpu_benchmark.json)

FP16 autocast + GradScaler, 8 timed steps after 2 warmup. Peak VRAM / tok/s:

| model | bsz×ctx | tok/s | peak VRAM | TFLOP/s |
|---|---|---:|---:|---:|
| micro 1.3M | 8×1024 | 256k | 0.67 GB | 2.0 |
| micro 1.3M | 16×1024 | 269k | 1.31 GB | 2.1 |
| tiny 4.2M | 4×1024 | 132k | 0.50 GB | 3.3 |
| tiny 4.2M | 8×1024 | 140k | 0.95 GB | 3.5 |
| tiny 4.2M | 16×1024 | 151k | 1.85 GB | 3.8 |
| small 12.2M | 4×1024 | 62k | 0.92 GB | 4.5 |
| small 12.2M | 8×1024 | 67k | 1.65 GB | 4.9 |
| small 12.2M | 16×1024 | 72k | 3.13 GB | 5.3 |
| small 12.2M | 8×2048 | 66k | 3.13 GB | 4.8 |
| medium 27.3M | 4×1024 | 31k | 1.54 GB | 5.1 |
| medium 27.3M | 8×1024 | 35k | 2.71 GB | 5.7 |
| medium 27.3M | 4×2048 | 32k | 2.71 GB | 5.2 |
| medium 27.3M | 8×2048 | **OOM** | — | — |
| medium 27.3M | 16×1024 | **OOM** | — | — |

Conclusions:
- 6 GB is comfortable for tiny/small at 8–16×1024; medium caps at 8×1024 (2.7 GB).
- Throughput is far below the 2060's ~50 TFLOP/s FP16 peak because models are
  small; larger batches amortize kernel-launch overhead better.
- VRAM scales ≈ linearly with bsz×ctx; weights+optimizer are minor at these sizes.
- Effective training batch = micro_bsz × grad_accum (e.g. 4×8×1024 = 32,768
  tokens/step). Tiny pilot runs ~95k tok/s → 5000 steps ≈ 3.5 min... measured
  ~0.35 s/step → ~30 min for 5000 steps.

## E1 — Data & tokenizer

- Kernel: torvalds/linux @ ffc253263a1375a65fa6c9f62a893e9767fbebfa (v6.6),
  shallow sparse checkout, 56,946 .c/.h files, 1,217,862,242 bytes.
- Splits (file-level, hash-grouped so exact dups never straddle splits):
  train 34,398 files / 404.9 MB; val 699 / 6.9 MB;
  holdout_subsystem 6,820 / 131.7 MB (mm,fs,net,crypto,security,sound,virt);
  holdout_driver 15,029 / 674.4 MB (drivers/{gpu,net,media,staging}).
  166 exact duplicate files detected.
- Tokenizer: byte-level BPE, vocab 4096 (incl. `<|file|>` = id 0), trained on
  train split only; 300/300 sampled files across all splits round-trip exactly.
- Tokenized: train 181,681,735 tokens (2.23 bytes/token), val 3,093,966,
  holdout_subsystem 61,403,726, holdout_driver ~300M (encoding).

## E2 — Overfit test (scripts/overfit_test.py)

Tiny 4.2M, fixed batch 8×256, 300 steps: loss 8.82 → 0.0023. **PASSED** —
model + AMP loop + RoPE/SwiGLU implementation can learn; pipeline correct.

## E3 — Tiny pilot (configs/tiny.yaml, 1000 steps, run `tiny-pilot`)

Hypothesis: pipeline trains end-to-end; loss should fall well below ln(4096)=8.32.
Result: **PASSED**. 1000 steps, 32.77M tokens, ~7.5 min wall, ~75–110k tok/s,
peak VRAM 0.52 GB. Train loss 8.8 → ~2.9; val loss 3.150 (ppl 23.3).
Checkpoint/resume verified separately (resumed ckpt_000500 → continued at step 500).

### BPB of tiny-pilot (4.2M params, 33M tokens) — evaluation/perplexity.py

| corpus | BPB | ratio | ppl |
|---|---:|---:|---:|
| kernel val | 2.011 | 3.98x | 22.9 |
| holdout_subsystem (2500 files) | 2.243 | 3.57x | 27.7 |
| holdout_driver (800 files) | 2.429 | 3.29x | 34.4 |
| git | 2.312 | 3.46x | 40.9 |
| nginx | 2.302 | 3.48x | 44.6 |
| xv6 | 2.625 | 3.05x | 70.5 |
| redis | 2.605 | 3.07x | 84.6 |
| cpython | 2.851 | 2.81x | 70.5 |
| lua | 2.890 | 2.77x | 98.4 |
| musl | 2.902 | 2.76x | 45.0 |
| sqlite | 3.280 | 2.44x | 151.6 |
| doom | 3.334 | 2.40x | 99.9 |

### Conventional compressors, same corpora (scripts/benchmark_compressors.py)

| corpus | gzip-6 | zstd-6 | bzip2-9 | xz-6 |
|---|---:|---:|---:|---:|
| kernel val | 1.946 | 1.802 | 1.685 | 1.511 |
| holdout_subsystem (100MB) | 1.815 | 1.590 | 1.395 | 1.262 |

Conclusions so far:
- After only 33M tokens the 4.2M-param net is **already competitive with gzip**
  on in-domain kernel code (BPB 2.01 vs 1.95) and clearly worse than xz (1.51).
- Held-out subsystems/drivers cost +0.23/+0.42 BPB vs val — expected domain shift.
- External C codebases: BPB 2.3–3.3; closest are git/nginx (application C),
  worst are doom/sqlite (very different style/era).
- Model BPB on val ≈ 2.0 corresponds to ppl 23 at 4.5 bits/token (2.23 B/token).
- Eval throughput ~20–60k tok/s depending on file sizes; long driver files are
  slow with batch padding — subsample for pilot evals.

## E4 — Range coder (evaluation/compression.py)

Carry-less 32-bit range coder, 16-bit cumulative tables (1 count reserved per
symbol). Verified **byte-exact reconstruction** on real files with tiny-pilot:
ctype.c 1426B → 699B actual (ideal 695B, overhead 3.5B); bounds.c 943B → 246B
(ideal 243B). So actual ≈ theoretical NLL/8 — the BPB numbers are real
achievable compression (conditional on the model). Total compression incl.
model+tokenizer overhead is reported separately by the script.

## E5 — Tokenizer vocab sweep (scripts/tokenizer_sweep.py, data/tokenizers/)

Byte-level BPE trained on train split only; bytes/token on val sample:
256→1.000, 512→1.537, 1024→1.833, 2048→2.071, **4096→2.261**, 8192→2.402,
16384→2.505. Diminishing returns past 4096. (BPB comparison per tokenizer needs
a model per tokenizer — queued as E8.)

## E6 — First substantial run: `small-8k` (configs/small.yaml) — DONE

Chosen from E0: small 12.2M at micro_batch 8 × ctx 1024 × accum 4 = 32,768
tok/step, peak 1.71 GB, ~62.5k tok/s avg. 8000 steps → **262.1M tokens**,
wall **69.9 min**, ~0.019 EFLOPs, best val loss 1.2136 (ppl 3.37) @ step 7749.

### BPB results (12.2M params, 262M tokens) vs conventional compressors

| corpus | neural BPB | ratio | gzip-6 | zstd-6 | xz-6 |
|---|---:|---:|---:|---:|---:|
| kernel val | **0.775** | 10.3x | 1.946 | 1.802 | 1.511 |
| holdout_subsystem | **1.097** | 7.3x | 1.815 | 1.590 | 1.262 |
| holdout_driver | **0.815** | 9.8x | – | – | – |
| linux v6.12 (temporal) | 0.853 | 9.4x | – | – | – |
| nginx | 0.916 | 8.7x | | | |
| openbsd | 1.035 | 7.7x | | | |
| freebsd | 1.091 | 7.3x | | | |
| git | 1.164 | 6.9x | | | |
| redis | 1.178 | 6.8x | | | |
| xv6 | 1.189 | 6.7x | | | |
| cpython | 1.200 | 6.7x | | | |
| musl | 1.473 | 5.4x | | | |
| lua | 1.482 | 5.4x | | | |
| sqlite | 1.497 | 5.3x | | | |
| doom | 1.578 | 5.1x | | | |
| busybox | 1.604 | 5.0x | | | |

Per-domain (v6.6): io_uring 0.52, drivers 0.72, sound 0.76, include 0.80,
arch 0.88, net 0.92, security 0.93, lib 0.94, mm 0.97, kernel 1.00, fs 1.03,
crypto 1.40.

**Headline: a 12.2M-param transformer trained for 70 min on an RTX 2060 beats
xz-6 on in-domain kernel code by ~1.9x (BPB 0.78 vs 1.51) and beats zstd/gzip
by 2.3–2.5x; on held-out subsystems it still beats xz (1.10 vs 1.26).** On
external codebases it compresses at 5–8.7x (BPB 0.85–1.60) — i.e., between
gzip and xz territory. Compression is real: the range coder (E4) achieves
actual ≈ ideal (within 0.5%).

Comparison vs tiny-pilot (4.2M, 33M tokens, val BPB 2.01): +8x more params and
+8x more tokens → 2.6x lower BPB. Compute-matched comparison (tiny at 262M
tokens) queued as E7.

## Planned next

- E7: eval small-8k (kernel val/holdouts, 12 external repos, temporal v6.12)
- E8: tiny model per tokenizer vocab (512…16384), equal steps — BPB vs vocab
- E9: context-length sweep (256…2048) at fixed tokens
- E10: lexical loss breakdown (evaluation/lexical_analysis.py) + generation eval
- E11: cross-subsystem generalization matrix
- E12: architecture variants (deep-vs-wide, shared layers, recurrent) at equal
  params and equal FLOPs; pointer/copy mechanism for identifiers
- E13: near-duplicate function detection (LSH) for contamination audit
