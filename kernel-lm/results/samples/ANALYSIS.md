# Free-generation samples — small-16k (12.2M params, 524M tokens)

Files: `batch_small16k.txt` (12 prompts, temp 0.8, top-k 40, seed 1234),
`batch_small16k_greedy.txt` (temp 0.0, seed 7). Prompts are kernel-style openers
(SPDX line, doc comment, `__init`/`__exit` functions, struct, file_operations,
probe, macros, `if (ret < 0)`, list iterator, MODULE_DESCRIPTION).

## What the model does well

- **Macro-structure is essentially perfect**: SPDX → doc comment → include
  guard → includes; probe/init functions with `int ret;` declarations,
  `devm_kzalloc(..., GFP_KERNEL)`, `if (!priv) return -ENOMEM;`,
  `platform_set_drvdata`, `module_init/module_exit` + `MODULE_LICENSE("GPL")`.
- **Real kernel idioms, not gibberish**: `.owner = THIS_MODULE,
  .unlocked_ioctl = foo_ioctl, .compat_ioctl = compat_ptr_ioctl,
  .llseek = noop_llseek` (greedy sample 5) is a textbook file_operations block.
  Error paths `if (ret < 0) { pr_err(...); return ret; }` cascade correctly
  (sampled sample 8).
- **Formatting**: tabs, aligned `=` in initializers, `/* * ... @param */` doc
  style — matches kernel style closely (this is where most of the BPB win lives).
- It even learns **document boundaries**: emits `<|file|>` and starts a fresh
  SPDX header mid-sample (samples 3, 11).

## What it gets wrong

- **Greedy decoding loops hard**: `ret = foo_init(); if (ret) return ret;` ×6,
  `struct foo *foo;` ×12 — classic argmax repetition; sampling (t=0.8) avoids it.
- **Semantically unsound under the surface**: `struct foo_bar(f)` (invalid C),
  `fflush(stdout)` / `exit(0)` in kernel context, tautological conditions
  (`pos->target == pos->target`, `foo->mode != FOO_MODE_WRITE && foo->mode != FOO_MODE_WRITE`).
  No type/semantic checker — output is syntactically and idiomatically plausible
  but not compilable novel code (0/45 under the gcc stub harness).
- **Copy behavior**: repeats identifiers/phrases from context (author listed
  twice, `FOO_MAX_DEV_TABLES 256/3/4/7/8` counting up) — consistent with the
  lexical finding that seen identifiers are cheap (0.9 nats) and it exploits that.
- Coherence decays after ~50–60 tokens; struct sample drifts into duplicated
  nested structs. `<|file|>` occasionally leaks into visible output.

## Verdict

A 12M-param model trained 2.3 h on a 2060 is a strong *kernel style/idiom*
model — boilerplate, formatting, error paths, API usage — and completes lines
well (30% exact next-line, 52% token accuracy). It is not a semantic reasoner:
free samples need sampling (not greedy) and human review; compilation of novel
generated functions is not yet achievable.

## Long samples (batch_long_sampled.txt: 18 prompts × 768 tokens, temp 0.8, incl. 6 real val-file prefixes)

- Real prefixes are reproduced **verbatim** (memorization), then continued
  with plausible field/include-level content (sample 13: `struct pci_controller`
  with real-looking members; sample 16: correct alpha pci.c include list).
- After ~30–60 tokens of novel continuation, **local repetition drift** sets in
  (duplicate struct members, `#include <linux/pci_hotplug.h>` repeated 8×,
  `ret = foo_init(); if (ret) return ret;` loops in greedy mode).
- Comments become semantically empty filler ("The bridge is a bridge of the
  busses... not supported in the LLD") while *syntactic* form stays perfect.
- No global coherence: functions repeat with slight variations instead of
  building one coherent file; `<|file|>` can appear mid-sample.

## Repetition penalty (HF-style logit division, penalty=1.3, temp 0.8)

Batches: `batch_reppen13.txt` (12 short, seed 1234), `batch_long_reppen13.txt`
(18 long × 768 tokens, seed 99) — same prompts/seeds as the unpenalized batches.

Quantitative effect (repeated 5-grams / duplicate lines):
- short batch: 2.3% / 9.9%  →  **0.0% / 1.3%**
- long batch:  21.4% / 32.4% → **0.1% / 7.9%**

Qualitatively: kills mechanical loops (`ret = foo_init(); if (ret)...` ×6,
repeated `#include <linux/pci_hotplug.h>`), produces more structurally varied
code (goto err paths, pr_warn, platform_device_add) — but the novelty is
hallucinated identifiers and word-salad comments ("FastRam and BIOS on U16
8bpp..."), not better semantics. Penalty trades repetition for invention.
