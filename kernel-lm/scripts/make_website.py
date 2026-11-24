#!/usr/bin/env python3
"""Generate a self-contained Mistral-style HTML page about the project,
with charts embedded as base64. Output: website/index.html"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CH = ROOT / "results/charts"
OUT = ROOT / "website/index.html"


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def load(p):
    with open(p) as f:
        return json.load(f)


# pull real numbers
small16k = load(ROOT / "results/eval/small16k_kernel.json")["results"]["data/tokens"]
med = load(ROOT / "results/eval/medium8k_kernel.json")["results"]["data/tokens"]
tiny = load(ROOT / "results/eval/tiny8k_kernel.json")["results"]["data/tokens"]
small = load(ROOT / "results/eval/small8k_kernel.json")["results"]["data/tokens"]
ext = load(ROOT / "results/eval/small16k_external.json")["results"]["data/eval_tokens"]

val_bpb = small16k["val"]["bits_per_byte"]
val_ratio = small16k["val"]["ideal_ratio"]

charts = {
    "neural": b64(CH / "neural_vs_conventional.png"),
    "progress": b64(CH / "progress.png"),
    "ext": b64(CH / "bpb_small16k_external.png"),
    "curves": b64(CH / "small-16k_curves.png"),
    "bench": b64(CH / "gpu_benchmark.png"),
}

ext_rows = "\n".join(
    f"<tr><td>{k}</td><td>{v['bits_per_byte']:.3f}</td><td>{v['ideal_ratio']:.2f}&times;</td><td>{v['perplexity']:.1f}</td></tr>"
    for k, v in sorted(ext.items(), key=lambda kv: kv[1]["bits_per_byte"])
)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mistral Vibe — Linux Kernel Neural Compression</title>
<style>
  :root {{
    --bg: #0b0b0d; --panel: #141417; --panel2: #1a1a1f; --line: #26262e;
    --text: #ececf1; --muted: #9b9ba7;
    --orange: #ff7000; --orange2: #ff8205; --yellow: #ffd800; --red: #ff4f00;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
         font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
         line-height: 1.6; }}
  code, pre {{ font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace; }}
  a {{ color: var(--orange2); }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 24px; }}
  header.hero {{ padding: 96px 0 64px; border-bottom: 1px solid var(--line);
    background:
      radial-gradient(900px 380px at 15% -10%, rgba(255,112,0,.22), transparent 60%),
      radial-gradient(700px 320px at 85% -10%, rgba(255,216,0,.12), transparent 60%),
      var(--bg); }}
  .kicker {{ color: var(--orange2); text-transform: uppercase; letter-spacing: .25em;
             font-size: 12px; font-weight: 700; }}
  h1 {{ font-size: clamp(38px, 6vw, 64px); line-height: 1.05; font-weight: 800;
       margin: 18px 0 14px; letter-spacing: -0.02em; }}
  h1 .grad {{ background: linear-gradient(90deg, var(--yellow), var(--orange) 55%, var(--red));
              -webkit-background-clip: text; background-clip: text; color: transparent; }}
  .sub {{ color: var(--muted); font-size: 18px; max-width: 760px; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 34px; }}
  .chip {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
           padding: 14px 18px; }}
  .chip b {{ display: block; font-size: 24px; color: var(--orange2); }}
  .chip span {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
  section {{ padding: 56px 0; border-bottom: 1px solid var(--line); }}
  h2 {{ font-size: 28px; font-weight: 800; margin-bottom: 8px; letter-spacing: -0.01em; }}
  h2::before {{ content: "// "; color: var(--orange); }}
  h3 {{ color: var(--yellow); margin: 22px 0 8px; font-size: 17px; }}
  p {{ color: #cfcfd8; margin: 10px 0; max-width: 900px; }}
  .grid {{ display: grid; gap: 18px; }}
  .grid.cols2 {{ grid-template-columns: 1fr 1fr; }}
  @media (max-width: 800px) {{ .grid.cols2 {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 20px; }}
  .card img {{ width: 100%; border-radius: 8px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--yellow); font-weight: 700; }}
  td:first-child {{ color: var(--text); }}
  .best {{ color: var(--orange2); font-weight: 700; }}
  pre.sample {{ background: #101014; border: 1px solid var(--line); border-radius: 12px;
    padding: 16px; overflow-x: auto; font-size: 12.5px; color: #d6d6e0; }}
  .tag {{ display: inline-block; background: rgba(255,112,0,.14); color: var(--orange2);
          border: 1px solid rgba(255,112,0,.4); border-radius: 999px; padding: 2px 10px;
          font-size: 11px; margin-right: 6px; }}
  footer {{ padding: 40px 0 60px; color: var(--muted); font-size: 13px; }}
  .sig {{ color: var(--orange2); font-weight: 700; }}
  .toc {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 22px; font-size: 13px; }}
  .toc a {{ color: var(--muted); text-decoration: none; }}
  .toc a:hover {{ color: var(--orange2); }}
</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <div class="kicker">Mistral Vibe &middot; research project</div>
    <h1>Linux Kernel<br><span class="grad">Neural Compression</span></h1>
    <p class="sub">How much transferable programming knowledge can a small neural network learn
    under a hard compute budget? A 12M-parameter transformer trained from scratch on the Linux
    kernel source — on a single RTX 2060 — reaches <b>@@VAL_BPB@@ bits/byte</b> on held-out kernel
    code (@@VAL_RATIO@@&times; ideal compression), beating xz-6 by ~2&times;.</p>
    <div class="chips">
      <div class="chip"><b>@@VAL_BPB@@ BPB</b><span>val, small-16k (12.2M params)</span></div>
      <div class="chip"><b>@@VAL_RATIO@@&times;</b><span>ideal compression ratio</span></div>
      <div class="chip"><b>2.1&times;</b><span>vs xz-6 (conditional)</span></div>
      <div class="chip"><b>524M</b><span>training tokens</span></div>
      <div class="chip"><b>6 GB</b><span>RTX 2060 VRAM, FP16 AMP</span></div>
      <div class="chip"><b>byte-exact</b><span>range-coded output verified</span></div>
    </div>
    <div class="toc">
      <a href="#results">Results</a><a href="#charts">Charts</a><a href="#compression">Compression accounting</a>
      <a href="#experiments">Controlled experiments</a><a href="#samples">Generation samples</a>
      <a href="#method">Method</a><a href="#repro">Reproduce</a>
    </div>
  </div>
</header>

<section id="results"><div class="wrap">
  <h2>Results</h2>
  <p>Decoder-only transformer (RoPE, RMSNorm, SwiGLU, tied embeddings, causal SDPA), trained with
  FP16 mixed precision on a pinned Linux v6.6 checkout (56,946 C/H files). Primary metric is
  <b>bits per byte (BPB)</b>, not perplexity.</p>
  <div class="card"><table>
    <tr><th>model</th><th>params</th><th>tokens</th><th>val BPB</th><th>holdout subsys</th><th>holdout drivers</th><th>vs xz-6 (1.51)</th></tr>
    <tr><td>tiny-8k</td><td>4.2M</td><td>262M</td><td>@@TINY_VAL@@</td><td>@@TINY_HS@@</td><td>@@TINY_HD@@</td><td>1.7&times; better</td></tr>
    <tr><td>small-8k</td><td>12.2M</td><td>262M</td><td>@@SMALL_VAL@@</td><td>@@SMALL_HS@@</td><td>@@SMALL_HD@@</td><td>1.9&times; better</td></tr>
    <tr><td>medium-8k</td><td>27.3M</td><td>262M</td><td>@@MED_VAL@@</td><td>@@MED_HS@@</td><td>@@MED_HD@@</td><td>2.1&times; better</td></tr>
    <tr><td class="best">small-16k (best)</td><td class="best">12.2M</td><td class="best">524M</td><td class="best">@@VAL_BPB@@</td><td class="best">@@S16_HS@@</td><td class="best">@@S16_HD@@</td><td class="best">2.1&times; better</td></tr>
  </table></div>
  <h3>External codebases (small-16k, held-out repos, pinned revisions)</h3>
  <div class="card"><table>
    <tr><th>codebase</th><th>BPB</th><th>ratio</th><th>ppl</th></tr>
    @@EXT_ROWS@@
  </table></div>
  <p>Temporal generalization: linux <b>v6.12</b> (unseen revision) BPB @@V612@@.
  Cross-kernel: FreeBSD 0.964, OpenBSD 1.106.</p>
</div></section>

<section id="charts"><div class="wrap">
  <h2>Charts</h2>
  <div class="grid cols2">
    <div class="card"><h3>Neural vs conventional — stacked with model size</h3><img src="data:image/png;base64,@@CH_NEURAL@@"></div>
    <div class="card"><h3>Best val BPB over successive experiments</h3><img src="data:image/png;base64,@@CH_PROGRESS@@"></div>
    <div class="card"><h3>BPB by external codebase (small-16k)</h3><img src="data:image/png;base64,@@CH_EXT@@"></div>
    <div class="card"><h3>Training curves, small-16k (524M tokens)</h3><img src="data:image/png;base64,@@CH_CURVES@@"></div>
    <div class="card"><h3>GPU benchmark: tok/s &amp; VRAM vs config (6 GB)</h3><img src="data:image/png;base64,@@CH_BENCH@@"></div>
  </div>
</div></section>

<section id="compression"><div class="wrap">
  <h2>Compression accounting: conditional vs total</h2>
  <p>All headline BPB numbers are <b>conditional</b> (decoder already has the model). A carry-less
  32-bit range coder driven by the model probabilities achieves <b>actual &asymp; ideal</b>
  (within 0.5%) and was verified <b>byte-for-byte exact</b> on real files.</p>
  <p>Including the model itself (12.2M params = 48.8 MB fp32 weights + 0.25 MB tokenizer; the 146 MB
  <code>.pt</code> on disk also stores optimizer state, not needed for inference):
  total BPB(N) = @@VAL_BPB@@ + 8&middot;49 MB / N. Crossover vs xz-6 on in-domain code at
  <b>N &asymp; 250 MB</b> — at 1 GB total &asymp; 0.91 BPB vs xz 1.51; at 10 GB &asymp; 0.73.
  On small corpora (&lt;50 MB) the model cost dominates and xz wins. The stacked bar in the chart
  above shows exactly this.</p>
</div></section>

<section id="experiments"><div class="wrap">
  <h2>Controlled experiments</h2>
  <div class="card"><table>
    <tr><th>experiment</th><th>setup</th><th>finding</th></tr>
    <tr><td>Scaling at equal tokens (262M)</td><td>4.2M vs 12.2M vs 27.3M params</td><td>val BPB 0.907 → 0.775 → 0.713; more params wins; small@524M ≈ medium@262M (0.708)</td></tr>
    <tr><td>Context length at equal tokens</td><td>ctx 512 / 1024 / 2048, small</td><td><b>1024 is the sweet spot</b> (0.795 / 0.775 / 0.807 val BPB)</td></tr>
    <tr><td>Looped transformer (n_loop=2)</td><td>tiny, 262M tokens, 2&times; FLOPs</td><td>val 0.893 vs 0.907 — small gain at 2&times; compute; <b>not a per-FLOP win</b></td></tr>
    <tr><td>Tokenizer vocab with models</td><td>tiny @4k steps, BPE 1024/4096/16384</td><td>val BPB 1.050 / 1.036 / 1.035 — <b>BPB-neutral</b> at this training length</td></tr>
    <tr><td>Contamination audit</td><td>exact-hash + 64-perm LSH over 56.9k files</td><td>166 exact dups (hash-grouped into one split); 1,605 near-dup pairs straddle splits (mostly boilerplate headers)</td></tr>
    <tr><td>Lexical loss breakdown</td><td>small-16k on val</td><td>seen identifiers 0.90 nats vs novel 2.90; identifier NLL vs distance 1–8→0.35, 512+→3.13; whitespace cheapest (0.41)</td></tr>
    <tr><td>Generation</td><td>next-line, 148 samples, greedy</td><td>30.4% exact line, 51.3% token accuracy, 81.8% balanced; gcc syntax-check of novel bodies 0/45 (needs compiler in the loop)</td></tr>
  </table></div>
  <p>Full ledger with failed experiments: <code>kernel-lm/EXPERIMENTS.md</code>.</p>
</div></section>

<section id="samples"><div class="wrap">
  <h2>What the model writes</h2>
  <p>Free samples from small-16k (temp 0.8). Strong kernel idiom and formatting; semantics need review.</p>
  <div class="grid cols2">
    <div class="card"><h3><span class="tag">sampled</span>fops initializer</h3>
<pre class="sample">static const struct file_operations foo_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = foo_ioctl,
	.compat_ioctl = compat_ptr_ioctl,
	.open = foo_open,
	.release = foo_release,
	.llseek = noop_llseek,
};</pre></div>
    <div class="card"><h3><span class="tag">sampled</span>error path</h3>
<pre class="sample">if (ret &lt; 0) {
		pr_err("Failing out %d, error %d\n", ret, err);
		return ret;
	}</pre></div>
    <div class="card"><h3><span class="tag">greedy</span>init function — loops hard</h3>
<pre class="sample">static int __init foo_init(void)
{
	int ret;

	ret = foo_init();
	if (ret)
		return ret;   /* ×6 — argmax repetition */</pre></div>
    <div class="card"><h3><span class="tag">sampled</span>long sample — verbatim prefix, then drift</h3>
<pre class="sample">/* real alpha pci.c header, continued: */
#include &lt;linux/pci.h&gt;
#include &lt;linux/pci_hotplug.h&gt;
#include &lt;linux/pci_hotplug.h&gt;  /* repetition drift */
...</pre></div>
  </div>
  <p>More: <code>kernel-lm/results/samples/</code> (sampled + greedy batches, 768-token long samples, analysis).</p>
</div></section>

<section id="method"><div class="wrap">
  <h2>Method</h2>
  <div class="grid cols2">
    <div class="card">
      <h3>Data &amp; contamination</h3>
      <p>torvalds/linux pinned at <code>ffc2532</code> (v6.6), shallow sparse checkout, .c/.h only.
      Splits are never token-random: held-out subsystems (mm/fs/net/crypto/security/sound/virt),
      held-out driver families (gpu/net/media/staging), seeded file-level val split grouped by content
      hash so exact duplicates never straddle splits. Byte-level BPE-4096 trained on train split only,
      verified lossless. 13 external repos pinned by revision for evaluation only.</p>
    </div>
    <div class="card">
      <h3>Metrics</h3>
      <p>PPL = exp(mean CE); BPB = NLL / (ln 2 &middot; bytes); ideal ratio = 8/BPB. Documents evaluated
      independently — no cross-file context, no padding in the loss, first token of each file excluded.
      Baselines: gzip/zstd/xz/bzip2 on identical corpora with consistent archive framing.</p>
    </div>
    <div class="card">
      <h3>Hardware</h3>
      <p>RTX 2060 (Turing, sm_75, 6 GB): FP16 autocast + GradScaler (no native BF16, no FlashAttention).
      Benchmarked: tiny 16&times;1024 = 1.85 GB / 151k tok/s; small 16&times;1024 = 3.13 GB / 72k tok/s;
      medium 8&times;1024 = 2.7 GB / 35k tok/s (8&times;2048 OOMs). Xeon E3-1245 v3, 16 GB RAM.</p>
    </div>
    <div class="card">
      <h3>Training</h3>
      <p>AdamW, cosine LR with warmup, grad clip 1.0, grad accumulation; checkpoints every 500 steps with
      resume (verified — survived a machine freeze mid-run). JSONL logging of loss/LR/grad-norm/tok/s/VRAM/FLOPs.</p>
    </div>
  </div>
</div></section>

<section id="repro"><div class="wrap">
  <h2>Reproduce</h2>
  <div class="card"><pre class="sample">git clone &lt;this repo&gt; && cd kernel-lm
uv pip install --python ../.venv/bin/python3 -r requirements.txt
python3 scripts/prepare_data.py        # pinned kernel -> manifest + splits
python3 scripts/train_tokenizer.py     # BPE-4096 on train split, lossless check
python3 scripts/tokenize_data.py       # memmap token arrays
python3 scripts/overfit_test.py        # correctness: must overfit a small batch
python3 scripts/benchmark_gpu.py       # VRAM/throughput sweep for 6 GB
python3 train.py --config configs/small.yaml --run-name small-8k --max-steps 8000
python3 evaluation/perplexity.py --checkpoint checkpoints/small-8k/latest.pt --corpora data/tokens data/eval_tokens
python3 evaluation/compression.py --checkpoint ... --files &lt;some .c files&gt;   # byte-exact range coding
python3 scripts/sample_batch.py --max-new-tokens 768 --real-prefixes 6          # free samples</pre></div>
  <p>Repo layout: <code>configs/ scripts/ model/ evaluation/ results/ checkpoints/ train.py generate.py README.md EXPERIMENTS.md</code></p>
</div></section>

<footer><div class="wrap">
  <p>Experimental ML research on a constrained budget. Built and run by
  <span class="sig">Mistral Vibe</span> (Mistral AI's CLI coding agent) on an RTX 2060 workstation.
  Raw data alongside every chart in <code>results/</code>; ledger in <code>EXPERIMENTS.md</code>.</p>
  <p class="sig">Generated by Mistral Vibe &middot; Co-Authored-By: Mistral Vibe &lt;vibe@mistral.ai&gt;</p>
</div></footer>
</body>
</html>
"""

vals = {
    '@@VAL_BPB@@': f"{val_bpb:.3f}", '@@VAL_RATIO@@': f"{val_ratio:.1f}",
    '@@TINY_VAL@@': f"{tiny['val']['bits_per_byte']:.3f}",
    '@@TINY_HS@@': f"{tiny['holdout_subsystem']['bits_per_byte']:.3f}",
    '@@TINY_HD@@': f"{tiny['holdout_driver']['bits_per_byte']:.3f}",
    '@@SMALL_VAL@@': f"{small['val']['bits_per_byte']:.3f}",
    '@@SMALL_HS@@': f"{small['holdout_subsystem']['bits_per_byte']:.3f}",
    '@@SMALL_HD@@': f"{small['holdout_driver']['bits_per_byte']:.3f}",
    '@@MED_VAL@@': f"{med['val']['bits_per_byte']:.3f}",
    '@@MED_HS@@': f"{med['holdout_subsystem']['bits_per_byte']:.3f}",
    '@@MED_HD@@': f"{med['holdout_driver']['bits_per_byte']:.3f}",
    '@@S16_HS@@': f"{small16k['holdout_subsystem']['bits_per_byte']:.3f}",
    '@@S16_HD@@': f"{small16k['holdout_driver']['bits_per_byte']:.3f}",
    '@@EXT_ROWS@@': ext_rows, '@@CH_NEURAL@@': charts['neural'],
    '@@CH_PROGRESS@@': charts['progress'], '@@CH_EXT@@': charts['ext'],
    '@@CH_CURVES@@': charts['curves'], '@@CH_BENCH@@': charts['bench'],
    '@@V612@@': f"{ext['linux-v6.12']['bits_per_byte']:.3f}",
}
for k, v in vals.items():
    HTML = HTML.replace(k, v)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print(f"saved {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
