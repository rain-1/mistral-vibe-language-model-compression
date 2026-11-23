# Linux Kernel Char-Level LM Benchmark — LSTM vs Transformer + Neural Compression

Date: 2025-11-23 · Machine: 8-core CPU laptop, 15 GB RAM, PyTorch 2.15 CPU

## Data
Official kernel.org tarballs; all source text (.c/.h/.S, Kconfig, Makefiles, docs, scripts)
concatenated into char-level corpora:

| version | chars |
|---|---|
| 2.0.40 | 27.5M |
| 3.0 (train) | 403M |
| 4.0 | 531M |
| 5.0 | 801M |
| 6.0 | 1.27B |

## Models (both trained 50 min on the 3.0 corpus, CPU)
- **LSTM**: 3 layers, hidden 1024, embed 256, vocab 1750, seq 100, batch 48. Final train PPL ~40s and falling; ~1k chars/s.
- **Transformer (GPT)**: 6 layers, d_model 384, 8 heads, seq 256, batch 32, AdamW 3e-4. 299 steps (~2.5M chars seen), train PPL 17.3 at last log.

## Perplexity (300k-char slice of each eval corpus)

| eval set | LSTM PPL | Transformer PPL | Transformer (matched, 50 min on 3M slice) |
|---|---|---|---|
| linux-2.0.40 | **8.64** | 21.33 | 10.00 |
| linux-4.0 | 16.30 | 40.76 | **9.61** |
| linux-5.0 | 16.34 | 40.45 | **10.01** |
| linux-6.0 | 16.31 | 39.98 | **10.42** |

Training curves: `training_ppl.png` (LSTM vs both transformer runs). The matched run
(train PPL 9.76) clearly beats the streamed transformer and the LSTM on 4.0/5.0/6.0 —
training longer on a matched slice was the right call.

## Generated samples (prompt `static int __init `, temps 0.5/0.8/1.0)
LSTM at temp 0.5 produces well-formed kernel C with plausible identifiers (`sk_proc_dev`,
`-EINVAL`, `spin_lock(...)`); quality degrades with temperature. Transformer samples are
rougher at this training amount (temp 0.8 still yields recognizable fragments like
`spin_lock`, `return -EINVAL`-style structure). Full text in `eval.log` / `eval_gpt.log`.

## Compression with system tools (tar + compressor, whole tree)

| tree | raw | gzip -9 | xz -9 | zstd -19 |
|---|---|---|---|---|
| 2.0.40 | 29.6 MB | 7.3 MB | **5.0 MB** | 5.2 MB |
| 3.0 | 421 MB | 94.1 MB | **61.5 MB** | 65.6 MB |
| 4.0 | 551 MB | 121.1 MB | **78.9 MB** | 83.8 MB |
| 5.0 | 813 MB | 158.7 MB | **100.5 MB** | 106.9 MB |
| 6.0 | 1.28 GB | 208.9 MB | **123.7 MB** | 132.9 MB |

xz -9 wins on ratio everywhere (~3.4–5.9x); zstd -19 close behind and much faster.

## Neural compression: LSTM + arithmetic coding (100k-char slice per kernel)

| corpus | bits/char | ratio |
|---|---|---|
| linux-2.0.40 | 3.446 | 2.32x |
| linux-3.0 | 4.113 | 1.95x |
| linux-4.0 | 4.136 | 1.93x |
| linux-5.0 | 4.171 | 1.92x |
| linux-6.0 | 4.172 | 1.92x |

## Neural compression: Transformer (GPT) + arithmetic coding (same 100k slices)

| corpus | bits/char | ratio |
|---|---|---|
| linux-2.0.40 | 4.705 | 1.70x |
| linux-3.0 | 5.146 | 1.55x |
| linux-4.0 | 5.163 | 1.55x |
| linux-5.0 | 5.194 | 1.54x |
| linux-6.0 | 5.195 | 1.54x |

Both neural coders compress to ~1.5–2.3x, consistent with their PPL, but well behind
xz -9 (~1.2 bits/char) — expected for small char-level models trained one hour on CPU.
The LSTM compresses better here because it saw more training text in the same budget. Artifacts: `model.pt`, `model_gpt.pt`, logs in `/home/riv/kernels/`.
