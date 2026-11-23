import torch, torch.nn as nn, time, math, random
torch.manual_seed(0); random.seed(0); torch.set_num_threads(8)
K = '/home/riv/kernels'
BUDGET_S = 50 * 60

with open(f'{K}/corpus_3.0.txt') as f: train_text = f.read()[:3_000_000]
itos = sorted(set(train_text)) + ['<unk>']
stoi = {c: i for i, c in enumerate(itos)}; V = len(itos)
print(f'vocab {V}, train chars {len(train_text)}', flush=True)
data = torch.tensor([stoi.get(c, V-1) for c in train_text], dtype=torch.long)

EMB, NHEAD, NHID, NLAYERS, SEQ, BS = 384, 8, 384, 6, 256, 32
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, EMB)
        self.pos = nn.Embedding(SEQ, EMB)
        layer = nn.TransformerEncoderLayer(EMB, NHEAD, NHID, batch_first=True, dropout=0.1, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, NLAYERS)
        self.fc = nn.Linear(EMB, V)
    def forward(self, x):
        p = torch.arange(x.size(1))
        h = self.emb(x) + self.pos(p)
        mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf')), 1)
        return self.fc(self.tr(h, mask))

model = GPT()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
lossf = nn.CrossEntropyLoss()
n = len(data); t0 = time.time(); step = 0; tl = 0.0; nb = 0
model.train()
while time.time() - t0 < BUDGET_S:
    xs, ys = [], []
    for b in range(BS):
        o = random.randrange(0, n - SEQ - 1)
        xs.append(data[o:o+SEQ]); ys.append(data[o+1:o+SEQ+1])
    x = torch.stack(xs); y = torch.stack(ys)
    loss = lossf(model(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    tl += loss.item(); nb += 1; step += 1
    if step % 100 == 0:
        el = time.time() - t0
        print(f'step {step} loss {tl/nb:.4f} ppl {math.exp(tl/nb):.2f} elapsed {el:.0f}s chars/s {step*BS*SEQ/el:.0f}', flush=True)
        tl = 0.0; nb = 0
torch.save({'sd': model.state_dict(), 'itos': itos, 'cfg': (EMB, NHEAD, NHID, NLAYERS, SEQ)}, f'{K}/model_gpt_matched.pt')
print(f'DONE steps {step} elapsed {time.time()-t0:.0f}s', flush=True)
