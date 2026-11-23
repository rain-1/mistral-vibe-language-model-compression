# 🐧 Linux Kernel Language Models — LSTM vs Transformer
### *Character-level and BPE models trained on Linux 3.0 and evaluated across five kernel generations.*

**Author:** Mistral Vibe · **Date:** 2025-11-23 · **Hardware:** 8-core CPU, 15 GB RAM, PyTorch 2.15 (CPU)

The follow-up BPE experiment ran on an RTX 2060. [Full benchmark tables and methods](BPE_FULL_BENCHMARK.md)

---

## 1. Dataset

Official tarballs from kernel.org, all source text (`.c/.h/.S`, Kconfig, Makefiles, docs, scripts) concatenated into char-level corpora:

| version | role | chars |
|---|---|---|
| 2.0.40 | eval | 27.5M |
| **3.0** | **train** | **403M** |
| 4.0 | eval | 531M |
| 5.0 | eval | 801M |
| 6.0 | eval | 1.27B |

The full-tree BPE run used 34,675 Linux 3.0 source files (399.6 MB, 188.35 million BPE tokens) and reserved 358 files by path for validation. The previous BPE run used the first 3 million characters of the corpus.

## 2. Models

| | LSTM | Transformer (streamed) | Transformer (matched) |
|---|---|---|---|
| architecture | 3-layer LSTM, h=1024 | 6-layer GPT, d=384, 8 heads | same |
| context | 100 chars | 256 chars | 256 chars |
| data seen in 50 min | full 3.0 stream (~2.8M chars) | full 3.0 stream (~2.5M chars) | first 3M chars, many epochs |
| final train PPL | ~40s, still falling | 17.3 | **9.76** |

The full-tree Transformer trained from scratch with the existing 4,000-token BPE vocabulary, six layers, width 384, eight heads, and a 256-token context. It completed one pass over the training split in 45,984 optimizer steps, about 31 minutes on the RTX 2060.

![Training PPL](training_ppl.png)

## 3. Perplexity across kernel generations (300k-char slices)

| eval set | LSTM | Transformer (streamed) | Transformer (matched) |
|---|---|---|---|
| linux-2.0.40 | **8.64** | 21.33 | 10.00 |
| linux-4.0 | 16.30 | 40.76 | **9.61** |
| linux-5.0 | 16.34 | 40.45 | **10.01** |
| linux-6.0 | 16.31 | 39.98 | **10.42** |

**Takeaway:** with matched training data, the transformer generalizes far better across kernel
versions (flat ~10 PPL) than the LSTM, which overfits to 3.0-era style and degrades on newer
kernels. Temporal proximity helps the LSTM only on 2.0.

### BPE Transformer perplexity

| Linux | Previous BPE | Full-tree BPE |
| --- | ---: | ---: |
| 2.0.40 | 127.69 | **64.77** |
| 3.0 | 28.26 | **26.22** |
| 4.0 | 30.23 | **25.36** |
| 5.0 | 36.81 | **26.92** |
| 6.0 | 43.58 | **28.54** |

These results use the first 300,000 characters of each corpus and are measured per BPE token. The character-level results above measure perplexity per character. Linux 3.0 is the training version, and kernel releases can share source. On a separate sample of 256 windows from the Linux 3.0 validation files, perplexity fell from 131.31 to 6.04. Four validation files also appeared in the old BPE training slice. The full-tree run changed both the amount of data and the number of training steps.

## 4. Generated samples (prompt: `static int __init ` / `static void `)

Two 1200-char samples per model at temp 0.5 + top-k 40 are appended at the end of this
report (`samples.md`). The LSTM produces the most code-like output; the transformer — even
after continued training (train PPL 7.99) — tends to collapse into repetitive token loops,
a classic small-char-LM failure mode where held-out PPL (~10) overstates sample coherence.

The first BPE Transformer used a 4,000-token vocabulary and trained for 50 minutes. Its samples are appended below.

### Full-tree BPE samples

The full-tree model produced more recognizable C on both original prompts. With the same seed, temperature 0.7, and top-k 40, the init sample emitted an init/exit pair and `module_init`, with seven identical registration calls. The `static void` sample began with a function body and degenerated into a chain of `port->port` accesses. The archived LSTM samples also repeated, with a long whitespace run on one prompt and a sequence of `#include` directives on the other. The LSTM samples used temperature 0.5.

Ten further completions used seeds 17 and 42 across five prompts. On the platform probe prompt, seed 42 generated this excerpt before repeating `.dma_addr[i]`:

```c
static int device_probe(struct platform_device *pdev)
{
	int retval;
	int i;

	for (i = 0; i < ARRAY_SIZE(dev->name); i++) {
		if (dev->dev) {
			dev_err(&pdev->dev, "can't register resource\n");
			return -EINVAL;
		}
	}
```

The same seed on `static int __init` repeated `__init` for the full completion. A descriptive interrupt-registration comment still led to repetition. [Matched old and new BPE samples](samples_bpe_full_comparison.md) · [Ten additional full-tree samples](samples_bpe_full_extra.md) · [Archived LSTM samples](samples.md)

## 5. Compression

### System tools (whole tree, tar + compressor)

| tree | raw | gzip -9 | **xz -9** | zstd -19 | LSTM neural, whole tree (est.) | GPT neural, whole tree (est.) |
|---|---|---|---|---|---|---|
| 2.0.40 | 29.6 MB | 7.3 MB | **5.0 MB** | 5.2 MB | 11.9 MB | 16.2 MB |
| 3.0 | 421 MB | 94.1 MB | **61.5 MB** | 65.6 MB | 207 MB | 260 MB |
| 4.0 | 551 MB | 121.1 MB | **78.9 MB** | 83.8 MB | 275 MB | 343 MB |
| 5.0 | 813 MB | 158.7 MB | **100.5 MB** | 106.9 MB | 417 MB | 520 MB |
| 6.0 | 1.28 GB | 208.9 MB | **123.7 MB** | 132.9 MB | 664 MB | 824 MB |

Neural columns are extrapolated from a 100k-char slice (bits/char × tree size): LSTM
3.45–4.17 bits/char, GPT 4.71–5.20 bits/char. xz -9 (~1.2 bits/char) beats both by ~3–5x.

### Neural compression (arithmetic coding with model probabilities, 100k chars each)

| corpus | LSTM bits/char | GPT bits/char |
|---|---|---|
| 2.0.40 | **3.45** | 4.71 |
| 3.0 | **4.11** | 5.15 |
| 4.0 | **4.14** | 5.16 |
| 5.0 | **4.17** | 5.19 |
| 6.0 | **4.17** | 5.20 |

Neural coders land at ~1.5–2.3x — consistent with their perplexities — but xz -9 (~1.2
bits/char) remains the king. A one-hour CPU-trained char model is a baseline, not a contender.

### Full-tree BPE compression

| Linux | Previous BPE bits/byte | Full-tree BPE bits/byte | xz -9 bits/byte on same slice |
| --- | ---: | ---: | ---: |
| 2.0.40 | 2.682 | **2.509** | 2.986 |
| 3.0 | **2.973** | 3.035 | 3.238 |
| 4.0 | **2.987** | 3.049 | 3.241 |
| 5.0 | **3.010** | 3.056 | 3.221 |
| 6.0 | **3.025** | 3.055 | 3.216 |

The 100,000-character slices reconstructed byte-for-byte using the encoder's probability tables. The measurements include arithmetic coding, framing, and any bytes needed to repair tokenizer losses. They assume shared model weights and tokenizer. The full-tree model used about 26–27% fewer bits than the historical LSTM on these slices. The existing whole-tree xz archives were 1.9–3.9× smaller than the full-tree BPE projections from the same prefixes. [Compression details, gzip and zstd results, and whole-tree projections](BPE_FULL_BENCHMARK.md#compression-on-the-first-100000-characters)

## 6. Reproduce

```bash
python3 build_corpus.py      # needs the five kernel trees
python3 train.py             # LSTM, 50-min budget
python3 train_gpt_matched.py # transformer, matched slice
python3 eval.py / eval_gpt_matched.py
python3 compress_lstm.py / compress_gpt.py
python3 plot.py              # training_ppl.png
```

For the full-tree experiment, run these commands with CUDA PyTorch, NumPy, and Tokenizers, using the original Linux 3.0 source archive and corpus text files.

```bash
python3 train_bpe_full.py --prepare linux-3.0.tar.xz
python3 train_bpe_full.py --baseline model_bpe.pt
python3 eval_bpe_full.py . model_bpe_full.pt model_bpe.pt
```

---
*Generated by Mistral Vibe · models: `model.pt`, `model_gpt.pt`, `model_gpt_matched.pt`*
### LSTM — example 1 (prompt: `static int __init `, temp 0.5 + top-k 40, 1200 chars)

```c
static int __init *id = size = irq;

	if (status & TX_INT) && (page & IWL_RIVER_SHIFT) && !(skb->port != 0x00) {
		if ((ret <= 0)
						return const struct tty_table *pdev;
		int err;
	}

	if (!port->io_mutex)
		return 0;
}

static void __iomem *mask = {
	.start = 0x00000000000000000000000000000000000000000008, .rx_priv = 0x000000000000000000, -1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
```

### LSTM — example 2 (prompt: `static void `, temp 0.5 + top-k 40, 1200 chars)

```c
static void saa_start_to_netdev(struct drm_device *dev, int offset, long flags)
{
	int ret;

	if (err < 0)
		return 0;

	/* FIXME: mask device */
	if (ret < 1) {
		printk(KERN_ERR "%s: endpoint is port from is device of a signal: Register and a reset the device in allocation.
 */

#include <linux/pci.h>
#include <asm/mach/arm/module.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/start.h>
#include <linux/slab.h>
#include <linux/mach/dma.h>
#include <linux/irq.h>
#include <linux/start.h>
#include <linux/string.h>
#include <linux/io.h>
#include <asm/irq.h>
#include <linux/module.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/parameter.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/skb.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/slab.h>
#include <asm/mmc.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/slab.h>
#include <asm/irq.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/size.h>
#include <linux/processor.h>
#include <linux/module.h>
#include <linux/irq.h>
#include <lin
```

### Transformer (matched, continued) — example 1 (prompt: `static int __init `, temp 0.5 + top-k 40, 1200 chars)

```c
static int __init struct treatturn the set ass the can of that is and a can be the function to for
a coure an the not the can for the on the should be and a the stand to the
the driver to muations struct to in the set as the be are the firer that ling the contatins
for the sare in the set the seect probeess the device to the stare freed the
the first the the seving the the can be device the on of the sight the mation in in the dever
in the lock not the call waiting the device sime stllem in the free or the
stage in the the sto is pare the of the the that ime if the card parter
is in the a s are and some shave art wo the ware conting of the synce filloce the befoore the prout the compic, in the
the are lock to is the can the chare the can ince in the contem.


The comple is the semincess in the rett the nater prober to the
instamenter.  This the the sime ame sot can in the to conterfaction the
contration is and in the cand in the the vall be of the use the capper in the
device in the card the set the firstame and the to the for driver the can whe the sat shat
or not the rester not imemater boot in to the can be ast the tat pare ally to the driver the device the
lock the seret but the of the seation o
```

### Transformer (matched, continued) — example 2 (prompt: `static void `, temp 0.5 + top-k 40, 1200 chars)

```c
static void und in the acking contard be so and on the whe
requerese the with a can pare set in thre the pagen set in the kernel with the
acke the sincall the to can the provent the lock is the kernel file the
withere a not stame the kernel sot the all to the chardwaing the file the is the
the can onstre the device the set are the change sen in the stemems of the the is in the
conter at be chares ore the constans of the can the defirest it the
the the list and the cher the to the coment is a allock in of the can an bot ment of the
impletex in the conter bexit the shat as it shat ctrive and the file the
staske the device the memory the or have of the wither to ime a itime interface

ardiver the semight that ime int is interface the conticall

seed in the she to mememory the shat have regading to dever, which aus have of the pating to the strice perrate in which
                                                                                                                                                                                                                                                                                                                                                   
```

### BPE Transformer — example 1 (prompt: `static int __init `, temp 0.7 top-k 40)

```c
static int __init  :  For values must be used. 

The padata is the caller happens, we can be specified by the memory barrier. 
The CPU 2.0.  The following sequence is the

beled.  The probe is invoked in a task is the
a of the following
the mutex, in the kernel.

	Note that is a kprobe,
	Example:
	O1.  Use this is not the "st_data:
	-genericated_mode=%4.	K_lock/write_t->data_t=_state_control (default)
			[1]
			f= 0x4]	c;
	-<file1)
	<p>
	f0fvoystick-0.g.0.4.0.  For 0,3.  0, "f0" is not be 0, IRQ, if it is used to be 	"  The kernel
	" is a cf course is a gcoverage that it.  This is only:

	struct 0;
	void circ_torture_read_count_max_delay_id_rcu_status

		/*
	-bit(struct kprobe_rcu_t *regs *kretprobe_register_lock);
	/*(struct foo_t *k(void)
		unsigned long_get_t *attr *p *attr(KERN_t *regs *regs), ");

	void struct foo_handler"
	..." (void *addr_mutex);
	size,
	/* Readkprobe_entry of the atomic_t nested for example, and
	the rcu_unregister_t_init *ssign_data,
	of_entry_trustroy_entry_obj_data_hold_mutex
```

### BPE Transformer — example 2 (prompt: `static void `, temp 0.7 top-k 40)

```c
static void  This time.  The default table.  Note that
	Configuration-y. ______________________________________switches has always not work items
	theseconsters.  For example
	c.  The 802
	-specific
	-ROM, and
		-backup-side critical sections, is done a set-side
	-side critical sections, and "-level-your-side critical sectionss"	-side critical sections-2.  If you have the
	.  For example, you may not work a single list of the
	-configuration
	-side critical sections of grace periodiced.  The 802 through that is a loisecond is
	rcu-orders, but it have to the
	-number-side critical sections on the data. AEnable if you need to
				(dev, not use, and mi386, it, "-y *
	struct parport_entity,	* "gdb	*
	kretprobe", ret_p)
		
				-classfile		@vger." == is brokenwise.

	The "E ALSO" for
	"o		The "" is a "-tree.  Note that "o	"	"	"	"		"
	"-"	"-"	.
		"	"		"	"	"		"		"	"	"	"		"	"	"	"	"	"	"	.k	"	"	"	"	"	"	"	"	"	"	" <	" ""	""	"	"	"	"	""	"	"
	      
	"	"	"	""	"	
```

