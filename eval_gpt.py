import torch, torch.nn as nn, math, random
torch.set_num_threads(8); torch.manual_seed(1); random.seed(1)
K = '/home/riv/kernels'
ck = torch.load(f'{K}/model_gpt.pt', weights_only=False)
itos = ck['itos']; stoi = {c: i for i, c in enumerate(itos)}; V = len(itos)
EMB, NHEAD, NHID, NLAYERS, SEQ = ck['cfg']
emb = nn.Embedding(V, EMB); pos = nn.Embedding(SEQ, EMB)
layer = nn.TransformerEncoderLayer(EMB, NHEAD, NHID, batch_first=True, norm_first=True)
tr = nn.TransformerEncoder(layer, NLAYERS); fc = nn.Linear(EMB, V)
sd = ck['sd']
emb.load_state_dict({'weight': sd['emb.weight']})
pos.load_state_dict({'weight': sd['pos.weight']})
tr.load_state_dict({k[len('tr.'):]: v for k, v in sd.items() if k.startswith('tr.')})
fc.load_state_dict({'weight': sd['fc.weight'], 'bias': sd['fc.bias']})
emb.eval(); pos.eval(); tr.eval(); fc.eval()

def fwd(x):
    p = torch.arange(x.size(1))
    h = emb(x) + pos(p)
    mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf')), 1)
    return fc(tr(h, mask))

def ppl(v, n=300_000):
    with open(f'{K}/corpus_{v}.txt') as f: t = f.read()[:n]
    d = torch.tensor([stoi.get(c, V-1) for c in t], dtype=torch.long)
    lossf = nn.CrossEntropyLoss(); tot = 0.0; nb = 0
    with torch.no_grad():
        for i in range(0, len(d)-SEQ-1, SEQ):
            x = d[i:i+SEQ].view(1, SEQ); y = d[i+1:i+SEQ+1].view(1, SEQ)
            tot += lossf(fwd(x).reshape(-1, V), y.reshape(-1)).item(); nb += 1
    print(f'linux-{v}: PPL {math.exp(tot/nb):.2f} ({len(t)} chars eval)', flush=True)

def sample(prompt, length, temp):
    d = [stoi.get(c, V-1) for c in prompt][-SEQ:]
    out = []
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([d])
            logits = fwd(x)[0, -1] / temp
            c = torch.multinomial(torch.softmax(logits, -1), 1).item()
            d = (d + [c])[-SEQ:]; out.append(itos[c])
    return prompt + ''.join(out)

for v in ['2.0.40', '4.0', '5.0', '6.0']: ppl(v)
print('\n--- samples ---', flush=True)
for temp in [0.5, 0.8, 1.0]:
    print(f'\n=== temp {temp} ===\n' + sample('static int __init ', 400, temp))
