import torch, torch.nn as nn, math, random
torch.set_num_threads(8); torch.manual_seed(1); random.seed(1)
K = '/home/riv/kernels'
ck = torch.load(f'{K}/model.pt', weights_only=False)
itos = ck['itos']; stoi = {c: i for i, c in enumerate(itos)}; V = len(itos)
EMB, HID, LAYERS = ck['cfg']
emb = nn.Embedding(V, EMB); lstm = nn.LSTM(EMB, HID, LAYERS, batch_first=True); fc = nn.Linear(HID, V)
emb.load_state_dict(ck['emb']); lstm.load_state_dict(ck['lstm']); fc.load_state_dict(ck['fc'])
emb.eval(); lstm.eval(); fc.eval()

def ppl(v, n=300_000):
    with open(f'{K}/corpus_{v}.txt') as f: t = f.read()
    t = t[:n]
    d = torch.tensor([stoi.get(c, V-1) for c in t], dtype=torch.long)
    SEQ = 100; lossf = nn.CrossEntropyLoss(); tot = 0.0; nb = 0
    with torch.no_grad():
        for i in range(0, len(d)-SEQ-1, SEQ):
            x = d[i:i+SEQ].view(1, SEQ); y = d[i+1:i+SEQ+1].view(1, SEQ)
            out, _ = lstm(emb(x)); logits = fc(out)
            tot += lossf(logits.reshape(-1, V), y.reshape(-1)).item(); nb += 1
    p = math.exp(tot/nb)
    print(f'linux-{v}: PPL {p:.2f}  ({len(t)} chars eval)', flush=True)
    return p

def sample(prompt, length, temp):
    d = [stoi.get(c, V-1) for c in prompt]
    h = None; out = []
    with torch.no_grad():
        for c in d:
            o, h = lstm(emb(torch.tensor([[c]])), h)
        for _ in range(length):
            logits = fc(o[0, -1]) / temp
            probs = torch.softmax(logits, -1)
            c = torch.multinomial(probs, 1).item()
            out.append(itos[c])
            o, h = lstm(emb(torch.tensor([[c]])), h)
    return prompt + ''.join(out)

for v in ['2.0.40', '4.0', '5.0', '6.0']:
    ppl(v)

print('\n--- samples ---', flush=True)
for temp in [0.5, 0.8, 1.0]:
    s = sample('static int __init ', 400, temp)
    print(f'\n=== temp {temp} ===\n{s}')
