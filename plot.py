import re, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

series = {}
for path, name in [('/home/riv/kernels/train.log', 'LSTM (full 3.0 stream)'),
                   ('/home/riv/kernels/train_gpt.log', 'Transformer (full 3.0 stream)'),
                   ('/home/riv/kernels/train_gpt_matched.log', 'Transformer (matched 3M slice)')]:
    try:
        steps, losses, ppls = [], [], []
        for line in open(path):
            m = re.search(r'step (\d+) loss ([\d.]+) ppl ([\d.]+)', line)
            if m:
                steps.append(int(m.group(1))); losses.append(float(m.group(2))); ppls.append(float(m.group(3)))
        if steps:
            series[name] = (steps, ppls)
    except FileNotFoundError:
        pass

plt.figure(figsize=(9, 5))
for name, (s, p) in series.items():
    plt.plot(s, p, marker='o', label=name)
plt.xlabel('step'); plt.ylabel('train perplexity')
plt.title('Training PPL vs step — Linux 3.0 char-level models')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig('/home/riv/kernels/training_ppl.png', dpi=120)
print('saved', list(series))
