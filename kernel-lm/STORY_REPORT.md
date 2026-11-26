# Story pretraining report — AO3 fanfiction language model

Companion to `EXPERIMENTS.md` entry **E16**. All raw data in `results/`, charts in
`results/charts/`, samples in `results/samples/`.

## 1. Data: getting "stories" when the AO3 torrent is banned

The requested Academic Torrents dump (`51c21fd1…`, "Archive Of Our Own [12.6 mil]",
163.8 GB) is **DMCA-banned** on academictorrents.com (`/download/<hash>.torrent` →
"This torrent is banned"). Substitutes, all individually selectable/unpackable:

| source | what | size used | role |
|---|---|---|---|
| HF `ray0rf1re/AO3-2020` config `full` | real AO3 2020 dump, 563 parquet shards; took shards 0–19 (~185 MB each, downloaded one by one) | 338,132 chapters / 128,554 stories / **5.07 GB** | **training corpus** |
| Academic Torrents `08d162a9…` "Fimfiction (2025-04-25)" | 20 `fimfiction_XX.jsonl.zst` shards (aria2c `--select-file` works; each shard unpacks alone with `zstd -d`) | 815,740 stories / 15.1 GB | external eval (3,000-file sample) |
| PG-19 (`deepmind/pg19`, GCS `deepmind-gutenberg`) | 4,800-book pack + official val/test, each book an individual `.txt` | 2.0 GB pack | external eval (books) |

Schema notes: AO3 parquet rows are `(storyId, idx, textRaw, text)` — one row per chapter;
we train on the cleaned `text`. Fimfiction jsonl rows are `(id, title, content)` with a few
null-padding bytes between records (skipped).

**Splits are story-level** (all chapters of a `storyId` share a split), seeded (42) and grouped
by content hash so duplicate stories never straddle splits:
`data/ao3/story_manifest.jsonl` — train 325,321 files / 4.87 GB, val 6,670 / 109 MB,
test 6,141 / 92 MB.

## 2. Tokenizer

Byte-level BPE, **vocab 16384** (+ `<|file|>` separator = id 0), trained on a 400 MB sample of
the AO3 train chapters only (`data/ao3/tokenizer-16k.json`). Verified **200/200 files
byte-exact roundtrip**. Prose tokenizes to **3.94 bytes/token** (kernel code @16k was 2.5) →
train split = **1.230B tokens** (`data/ao3/story_tokens/train.bin`, uint16 memmap).

## 3. Training

Best kernel recipe scaled to the budget: **medium-16k** (8 layers × 512, 33.6M params incl.
16k-vocab tied embedding), ctx 1024, RoPE/RMSNorm/SwiGLU, FP16 AMP + GradScaler, AdamW
lr 6e-4 → 6e-5 cosine, warmup 200, grad clip 1.0, micro-batch 8 × accum 4 (32,768 tok/step).

GPU benchmark first (`results/benchmarks/gpu_benchmark_story.json`): medium 8×1024 =
**34.6k tok/s @ 2.7 GB**; large 8×1024 = 23.6k tok/s @ 3.8 GB; 16×1024 OOMs on 6 GB.

Run `story-medium-16k`: **30,000 steps = 983M tokens in 8.7 h wall clock, 32.2k tok/s,
4.15 GB VRAM**. Final train loss 3.03, **val loss 3.065 nats**. Log:
`results/story-medium-16k/log.jsonl`; checkpoints `checkpoints/story-medium-16k/`.

## 4. Evaluation (yes — both done)

**Perplexity / BPB** (`evaluation/perplexity.py`, doc-boundary-correct,
`results/eval/story-ckpt_030000.json`):

| corpus (held out) | files | tokens | CE nats | PPL | BPB | ideal ratio | xz-6 BPB |
|---|---:|---:|---:|---:|---:|---:|---:|
| AO3 val | 6,670 | 27.7M | 3.058 | 21.3 | **1.118** | 7.16× | 2.20 |
| AO3 test | 6,141 | 23.2M | 3.047 | 21.1 | **1.102** | 7.26× | 2.21 |
| Fimfiction (other archive) | 3,000 | 15.0M | 3.429 | 30.8 | **1.301** | 6.15× | 2.15 |
| PG-19 test (old books) | 100 | 12.8M | 4.648 | 104 | 2.072 | 3.86× | 2.23 |
| PG-19 val | 50 | 5.5M | 4.676 | 107 | 2.111 | 3.79× | 2.31 |

**Compression** (`evaluation/compression.py`, carry-less 32-bit range coder, 16-bit cumfreq):
on 40 AO3 test files (715 KB) — ideal 103,751 B, **actual 103,975 B (0.2 % coder overhead),
BPB 1.163, every file byte-for-byte exact** (`results/eval/story-ckpt_030000-compression.txt`).
Conventional baselines on the same corpora (`results/benchmarks/compressors_*.json`):
gzip-6 ≈ 2.98, zstd-6 ≈ 2.75, bzip2-9 ≈ 2.26, xz-6 ≈ 2.20 BPB.

**Total compression incl. model** (169 MB = fp32 weights + tokenizer): total BPB(N) =
1.118 + 8·169 MB/N; **xz-6 crossover at N ≈ 1.25 GB** of corpus (chart
`story_full_amortization.png`). On small corpora the model cost dominates.

**Sampling**: 20 free samples at temp 0.8 / top-k 40 / rep-penalty 1.2
(`results/samples/story-ckpt_030000.txt`) — fluent fanfic-style prose, consistent character
names and dialogue. Repetition-penalty sweep (word-4-gram repetition in samples):
rp 1.0 → 1.56 %, **rp 1.2 → 0 %**, rp 1.5 → 0 % (`story_full_rep.png`).

## 5. Charts (`results/charts/`)

- `story_full_loss.png` — train/val loss and val PPL vs training tokens
- `story_full_progress.png` — val BPB over training vs xz/gzip lines
- `story_full_bpb_corpora.png` — neural vs gzip/zstd/xz BPB per held-out corpus (+ original 8 BPB bar)
- `story_neural_vs_conventional.png` — log-scale version with the 169 MB model **stacked** on the neural bar
- `story_full_amortization.png` — total BPB vs corpus size, xz crossover ≈ 1.25 GB
- `story_full_rep.png` — repetition penalty vs repeated 4-grams

## 6. Findings

- A **33.6M-param** model trained 8.7 h on a single RTX 2060 reaches **1.10 BPB (7.3×) on held-out
  AO3 fanfiction — ~2× better than xz-6 (2.21)** on the same bytes, and the range coder makes it
  real (byte-exact, within 0.2 % of ideal).
- Transfer to a *different* fanfic archive (Fimfiction): 1.30 BPB vs xz 2.15 — the model learned
  general fanfic/prose conventions, not just AO3 specifics. Transfer to **1820s–1910s books
  (PG-19) is much weaker** (2.07 BPB, only 1.08× over xz) — era/style domain shift is the main
  generalization gap for prose.
- Prose costs ~1.6× more bytes/token than code at the same vocab (3.94 vs 2.5), so BPB numbers
  are not directly comparable across domains; per-token CE (3.05 nats) is the cleaner metric.
- No overfitting: train ≈ val loss through 983M tokens (1.23B available).

## 7. Limitations / queued

- lr sweep not run (10 h budget consumed); config ready: `configs/story-medium-16k-lr12.yaml`.
- No large-16k story run (would need ~9.5 h for Chinchilla tokens), no ctx-2048 variant, no
  quantized story model yet.
- AO3 `full` config is the general corpus; the separate `adult_18plus_100K` config was
  deliberately excluded.
- Fimfiction/PG-19 used only for eval, never trained on.

## 8. Quantized + pruned story model

`scripts/quantize_prune.py` (global magnitude pruning of Linear weights + per-channel INT8,
embedding per-row INT8, LN fp16) applied to `ckpt_030000.pt`:

| artifact | size | AO3 val BPB | test BPB | fimfiction | pg19-test |
|---|---:|---:|---:|---:|---:|
| dense fp32 (+tokenizer) | 169.0 MB | 1.118 | 1.102 | 1.301 | 2.072 |
| **pruned 30% + INT8** | **43.4 MB (3.9× smaller)** | **1.154** | 1.139 | 1.332 | 2.124 |
| pruned 50% + INT8 | 43.4 MB | 1.477 | 1.456 | 1.711 | 2.494 |

- 30% pruning costs only **+0.037 BPB** (same delta as the kernel small-16k model); 50% is too
  lossy (+0.36). Range coder on the 30% model: still **byte-exact**, actual 107,065 B vs ideal
  106,840 B, BPB 1.198 on the 715 KB sample.
- **xz-6 total-compression crossover drops from ≈1.25 GB (dense) to ≈330 MB** with the 43 MB
  artifact (`results/charts/story_quant.png`).
- Artifacts: `checkpoints/story-medium-16k-s03-quant.pt` (42.3 MB), `…-s03-dequant.pt` (eval),
  eval JSONs `results/eval/story-s03-dequant.json`, `story-s05-dequant.json`.

## 9. Fine-tune: bullet-outline -> story (E17)

**Data**: 12,000 Fimfiction stories (never used in pretraining; eval split excluded) turned into
(OUTLINE -> STORY) pairs. Outlines are extractive skeletons (up to 6 evenly-spaced sentences as
bullets) — chosen because the 33M base model does not follow few-shot outline instructions
reliably (tested: generates prose or empty bullets). Format: `OUTLINE:\n- ...\n\nSTORY:\n<story>`.
11,700 train / 300 val pairs = 9.57M tokens (`data/ft_tokens/`).

**Run** `story-ft-outline`: resumed weights from `ckpt_030000.pt` (new `--resume-weights-only`
flag in train.py), lr 1e-4 -> 1e-5 cosine, 2,500 steps x 32,768 tok = 82M tokens, ~43 min,
32k tok/s, 4.15 GB. FT val loss 2.74. (`results/story-ft-outline/log.jsonl`)

**Conditional eval** (`scripts/cond_eval.py`, 200 held-out fimfiction pairs + 150 AO3-test
stories with extractive outlines):

| setting | CE nats | PPL | BPB |
|---|---:|---:|---:|
| story alone (unconditional) | 3.139 | 23.1 | 1.192 |
| story given its bullet outline | **2.676** | **14.5** | **1.019** |

-> the outline is worth **0.17 BPB (1.17x)**; conditioned, the model beats xz-6 (2.15) by >2x.

**Regression**: unconditional AO3 val BPB 1.118 -> 1.278 after FT (small model, 82M FT tokens:
format shift costs some general prose). Net: use the base model for compression, the FT model
for outline->story generation.

**Samples** (`results/samples/outline_ft.txt`): given 5-bullet outlines, the FT model writes
coherent fanfic-style scenes that follow the outline's beats; it sometimes re-emits the
`OUTLINE:` marker mid-story (no stop-sequence support in generate()). Chart: `story_ft.png`.
