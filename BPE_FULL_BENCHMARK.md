# Full-tree Linux 3.0 BPE Transformer benchmarks

The final checkpoint completed one epoch at step 45,984. Evaluation uses the same leading text slices as the original benchmark. Linux 3.0 served as the training version. Other kernel versions can share source with it.

[Main experiment report](REPORT.md) · [Matched BPE completions](samples_bpe_full_comparison.md) · [Additional full-tree completions](samples_bpe_full_extra.md)

## Perplexity on the first 300,000 characters

| Linux | Previous BPE | Full-tree BPE |
| --- | ---: | ---: |
| 2.0.40 | 127.69 | 64.77 |
| 3.0 | 28.26 | 26.22 |
| 4.0 | 30.23 | 25.36 |
| 5.0 | 36.81 | 26.92 |
| 6.0 | 43.58 | 28.54 |

Perplexity is per BPE token. Character-level perplexities from the original report use a different unit. The separate Linux 3.0 held-out-window test measured 6.04 for the full-tree model and 131.31 for the previous BPE model. That evaluation samples different parts of the corpus.

## Compression on the first 100,000 characters

Rates are measured in bits per UTF-8 byte. Each system compressor ran on the same slice as the full-tree BPE Transformer.

| Linux | Full-tree BPE | gzip -9 | xz -9 | zstd -19 | Full-tree BPE bytes | Raw bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0.40 | 2.509 | 3.234 | 2.986 | 3.083 | 31,368 | 100,000 |
| 3.0 | 3.035 | 3.566 | 3.238 | 3.368 | 37,946 | 100,030 |
| 4.0 | 3.049 | 3.574 | 3.241 | 3.372 | 38,124 | 100,030 |
| 5.0 | 3.056 | 3.561 | 3.221 | 3.352 | 38,210 | 100,032 |
| 6.0 | 3.055 | 3.555 | 3.216 | 3.344 | 38,202 | 100,033 |

The BPE benchmark arithmetic-codes all tokens, including the first token of each context window, and counts the packed payload, framing, and repair bytes. The Linux 5.0 and 6.0 slices each needed a two-byte repair for text the tokenizer could not represent. All five slices reconstructed byte-for-byte using the encoder's probability tables. The reported rates assume the model and tokenizer are shared. The benchmark does not independently regenerate the probability tables during decoding.

## Whole-tree projection

Sizes are in MiB. gzip, xz, and zstd sizes are measured from the existing tar archives. The Transformer projection multiplies each tree's raw tar size by its 100,000-character compressed-to-raw ratio. It is a projection from the prefix and excludes the model and tokenizer.

| Linux | Raw tar | gzip -9 | xz -9 | zstd -19 | Full-tree BPE estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2.0.40 | 30.0 | 7.3 | 5.0 | 5.2 | 9.4 |
| 3.0 | 430.1 | 94.1 | 61.5 | 65.6 | 163.2 |
| 4.0 | 562.7 | 121.1 | 78.9 | 83.8 | 214.4 |
| 5.0 | 823.3 | 158.7 | 100.5 | 106.9 | 314.5 |
| 6.0 | 1278.4 | 208.9 | 123.7 | 132.9 | 488.2 |

Run the [benchmark script](eval_bpe_full.py) on the training machine with both checkpoints and the original corpus prefixes available.

```bash
python eval_bpe_full.py /path/to/corpora model_bpe_full.pt model_bpe.pt
```
