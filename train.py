import torch, torch.nn as nn, time, math, random, sys, os

torch.manual_seed(0); random.seed(0)
torch.set_num_threads(8)
DEV = 'cpu'
K = '/home/riv/kernels'
BUDGET_S = 50 * 60  # training time budget

def load(v):
    with open(f'{K}/corpus_{v}.txt') as f: return f.read()

train_text = load('3.0')
chars = sorted(set(train_text))
itos = chars + ['<unk>']
stoi = {c: i for i, c in enumerate(itos)}
V = len(itos)
print(f'vocab {V}, train chars {len(train_text)}', flush=True)

data = torch.tensor([stoi.get(c, V-1) for c in train_text], dtype=torch.long)

EMB, HID, LAYERS, SEQ, BS = 256, 512, 2, 100, 32
model = nn.Sequential(nn.Embedding(V, EMB),
                      nn.LSTM(EMB, HID, LAYERS, batch_first=True, dropout=0.1),
                      nn.Linear(HID, V))
# unpack manually instead
emb = nn.Embedding(V, EMB); lstm = nn.LSTM(EMB, HID, LAYERS, batch_first=True, dropout=0.1); fc = nn.Linear(HID, V)
model = nn.ModuleList([emb, lstm, fc]).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10**9)
lossf = nn.CrossEntropyLoss()

n = len(data)
order = torch.randperm(n // (SEQ+1)) * (SEQ+1)
t0 = time.time(); step = 0; total_loss = 0.0; nb = 0
model.train()
while time.time() - t0 < BUDGET_S:
    idx = order[step % len(order)]
    if idx + SEQ + 1 > n: break
    x = data[idx:idx+SEQ].view(1, SEQ)
    y = data[idx+1:idx+SEQ+1].view(1, SEQ)
    # build batch
    xs, ys = [], []
    for b in range(BS):
        o = (idx + b * 7919) % (n - SEQ - 1)
        xs.append(data[o:o+SEQ]); ys.append(data[o+1:o+SEQ+1])
    x = torch.stack(xs); y = torch.stack(ys)
    h = None
    e = emb(x); out, h = lstm(e, h); logits = fc(out)
    loss = lossf(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    total_loss += loss.item(); nb += 1; step += 1
    if step % 200 == 0:
        el = time.time() - t0
        print(f'step {step} loss {total_loss/nb:.4f} ppl {math.exp(total_loss/nb):.2f} elapsed {el:.0f}s chars/s {step*BS*SEQ/el:.0f}', flush=True)
        total_loss = 0.0; nb = 0

torch.save({'emb': emb.state_dict(), 'lstm': lstm.state_dict(), 'fc': fc.state_dict(),
            'itos': itos, 'cfg': (EMB, HID, LAYERS)}, f'{K}/model.pt')
print(f'DONE steps {step} elapsed {time.time()-t0:.0f}s', flush=True)
