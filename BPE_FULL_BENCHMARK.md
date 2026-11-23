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

Every method processed the same UTF-8 bytes. Rates are in bits per UTF-8 byte.

| Linux | gzip -9 | xz -9 | zstd -19 | LSTM | Full-tree BPE Transformer |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2.0.40 | 3.234 | 2.986 | 3.083 | 3.578 | **2.509** |
| 3.0 | 3.566 | 3.238 | 3.368 | 4.244 | **3.035** |
| 4.0 | 3.574 | 3.241 | 3.372 | 4.267 | **3.049** |
| 5.0 | 3.561 | 3.221 | 3.352 | 4.302 | **3.056** |
| 6.0 | 3.555 | 3.216 | 3.344 | 4.302 | **3.055** |

### Exact compressed sizes in bytes

| Linux | Raw bytes | gzip -9 | xz -9 | zstd -19 | LSTM | Full-tree BPE Transformer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0.40 | 100,000 | 40,426 | 37,320 | 38,539 | 44,726 | **31,368** |
| 3.0 | 100,030 | 44,590 | 40,488 | 42,109 | 53,067 | **37,946** |
| 4.0 | 100,030 | 44,689 | 40,528 | 42,158 | 53,352 | **38,124** |
| 5.0 | 100,032 | 44,522 | 40,280 | 41,916 | 53,793 | **38,210** |
| 6.0 | 100,033 | 44,448 | 40,208 | 41,817 | 53,796 | **38,202** |

The benchmark uses the same arithmetic coder for both models. It encodes every symbol, including the first symbol of each context window, and counts the packed payload, framing, and repair bytes. The LSTM uses its original 100-character windows. The Transformer uses 256-token windows. The Linux 5.0 and 6.0 Transformer slices each required a two-byte repair for text the BPE vocabulary could not represent. Both models reconstructed all five slices byte-for-byte using the encoder's probability tables. Sizes assume the model weights and vocabulary are shared.

## Whole-tree archives and neural projections

Sizes are in MiB. gzip, xz, and zstd sizes come from the existing tar archives. The neural columns multiply each tree's raw tar size by the model's measured slice compression ratio. The projections exclude model weights and vocabulary.

| Linux | Raw tar | gzip -9 | xz -9 | zstd -19 | LSTM est. | Full-tree BPE Transformer est. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0.40 | 30.0 | 7.3 | **5.0** | 5.2 | 13.4 | 9.4 |
| 3.0 | 430.1 | 94.1 | **61.5** | 65.6 | 228.2 | 163.2 |
| 4.0 | 562.7 | 121.1 | **78.9** | 83.8 | 300.1 | 214.4 |
| 5.0 | 823.3 | 158.7 | **100.5** | 106.9 | 442.7 | 314.5 |
| 6.0 | 1278.4 | 208.9 | **123.7** | 132.9 | 687.5 | 488.2 |

Run the [benchmark script](eval_bpe_full.py) with the two model checkpoints and the original corpus prefixes.

```bash
python eval_bpe_full.py /path/to/corpora model_bpe_full.pt --lstm model.pt
```
