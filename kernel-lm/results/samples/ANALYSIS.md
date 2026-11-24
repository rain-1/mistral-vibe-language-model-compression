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
