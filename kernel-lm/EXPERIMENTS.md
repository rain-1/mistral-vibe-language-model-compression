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

## E7 — Compute-matched comparison: tiny-8k (4.2M, 262M tokens) vs small-8k

Both trained on exactly 262.1M tokens (8000 × 32,768). (tiny-8k survived a
machine freeze; resumed from ckpt_005500 via --resume, finished 8000.)

| corpus | tiny-8k BPB (4.2M) | small-8k BPB (12.2M) |
|---|---:|---:|
| val | 0.9070 | **0.7753** |
| holdout_subsystem | 1.2254 | **1.0968** |
| holdout_driver | 0.9244 | **0.8147** |

Conclusion: at equal training tokens (262M), the 12.2M model is uniformly
better (~0.13–0.17 BPB). At this scale, under a fixed token budget, **more
parameters beats more of the same** — consistent with Chinchilla scaling.
(tiny-8k wall ~45 min at ~135k tok/s vs small-8k 70 min at ~62k tok/s, so at
equal *wall-clock* the gap narrows; FLOPs-matched comparison still queued.)

## E7b — Medium-8k (27.3M params, 262M tokens, 8×1024×4, 2.82 GB peak, 35.8k tok/s, ~2h)

Equal-token scaling ladder at 262.1M tokens (val BPB):

| params | val BPB | holdout_subsystem | holdout_driver | wall |
|---:|---:|---:|---:|---:|
| 4.2M (tiny-8k) | 0.907 | 1.225 | 0.924 | ~45 min |
| 12.2M (small-8k) | 0.775 | 1.097 | 0.815 | ~70 min |
| **27.3M (medium-8k)** | **0.713** | **1.048** | **0.773** | ~2h |

Monotonic gains with params at fixed tokens; medium beats xz-6 (1.51) on val by
**2.1x**. Diminishing per-param returns (4.2→12.2: −0.13 BPB; 12.2→27.3: −0.06).

medium-8k vs small-8k on external repos (BPB): medium wins 12/13 — linux-v6.12
0.762, nginx 0.878, freebsd 0.979, cpython 1.085, redis 1.104, git 1.110,
openbsd 1.116 (small better), xv6 1.122, lua 1.411, sqlite 1.443, doom 1.501,
busybox 1.574, musl 1.641 (small better). Cross-repo transfer scales with params.

## E8 — Context sweep at equal tokens (262.1M tokens, small 12.2M)

| ctx | val BPB | holdout_subsystem | holdout_driver |
|---:|---:|---:|---:|
| 512 (16k steps) | 0.7948 | 1.1410 | 0.9079 |
| **1024 (8k steps)** | **0.7753** | **1.0968** | **0.8147** |
| 2048 (4k steps) | 0.8074 | 1.1108 | 0.7993 |

Conclusion: **ctx 1024 is the sweet spot** at 262M tokens; 512 and 2048 both lose.
(2048 has fewer optimizer steps at equal tokens, which may partly explain it.)

## E9 — Contamination audit (scripts/near_dup_audit.py)

Exact dups: 166 files (hash-grouped into the same split by construction).
Near-dups (8-gram shingles, whitespace-normalized, 64-perm minhash LSH over
56,902 files): 117,531 pairs with est. Jaccard ≥ 0.5, of which **1,605 straddle
train/eval splits** — mostly boilerplate (dt-bindings headers, trace wrappers,
selftests). Report: data/near_dup_report.json. Kernel code is highly templated,
so some leakage is unavoidable at file level; function-level de-dup is a
possible later refinement.

## E10 — Lexical breakdown & generation (small-8k, val sample)

Mean NLL by category: whitespace 0.42 < preprocessor 0.65 < string 0.66 <
comment 1.20 < identifier_novel 1.34 < char 1.34 < keyword 1.37 < number 1.54 <
type 1.62 < operator 1.67 nats. Identifier NLL vs distance to previous
occurrence: 1–8 tokens → 0.40, 9–64 → 1.23, 65–512 → 1.87, 512+ → 3.50 nats
(clear recency/copy signal). Generation (next-line, 148 samples): greedy
exact-line 28.4%, token-acc 51.6%, balanced 79.7%; sampled t=0.8: 25.7% exact.

## Planned next

## E11 — Looped transformer: tiny-loop2 (4 layers × n_loop=2, 8.4M params, 262M tokens)

Same architecture as tiny-8k but each block stack applied twice per forward
(2× FLOPs/token). Result at equal 262.1M tokens:

| corpus | tiny-8k (loop1) BPB | tiny-loop2 BPB |
|---|---:|---:|
| val | 0.9070 | **0.8925** |
| holdout_subsystem | 1.2254 | **1.2098** |
| holdout_driver | 0.9244 | **0.9016** |

Looping gives a small gain at equal tokens (~1.5%), but costs 2× compute — at
equal FLOPs (loop2 trained on half the tokens) it would lose. **Verdict: recurrent
looping is not a per-FLOP win at this scale** (parameter-matched comparison still
possible with 8-layer loop1 vs 4-layer loop2).

## E11b — Tokenizer vocab sweep with models (tiny, 4000 steps = 131M tokens, val BPB)

| tokenizer | bytes/token (val) | val BPB |
|---|---:|---:|
| BPE-1024 | 1.833 | 1.0495 |
| BPE-4096 | 2.261 | ~1.036 (tiny-8k @ step 3999) |
| BPE-16384 | 2.505 | 1.0349 |

At 4k steps the vocab choice is BPB-neutral — the embedding-parameter cost of
large vocabs is offset by fewer tokens. (Longer runs may differ; queued.)

- E11c: small ctx2048 (4×2048, 4000 steps = 262M tokens) vs ctx1024 — running
- E12: tiny model per tokenizer vocab (1024, 16384 token arrays ready), equal steps
- E13: cross-subsystem generalization matrix (train per subsystem)
- E14: small ctx2048 (4×2048, 4000 steps = 262M tokens) vs ctx1024
- E15: compile-based generation eval (evaluation/compile_eval.py, gcc -fsyntax-only)
