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

## Planned next

- E4: BPB eval of tiny-pilot on val / holdout / external repos (perplexity.py)
- E5: conventional compressor baselines (gzip/zstd/xz/bzip2) on same corpora
- E6: first substantial run — config chosen from E0/E3 measurements
- E7+: tokenizer vocab sweep, context sweep, architecture variants, arithmetic coder
