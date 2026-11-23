import torch, torch.nn as nn, random
torch.set_num_threads(8); random.seed(7)
K = '/home/riv/kernels'

def topk_sample(logits, temp, k=40):
    logits = logits / temp
    v, idx = torch.topk(logits, k)
    p = torch.softmax(v, -1)
    c = torch.multinomial(p, 1).item()
    return idx[c].item()

def gen_lstm(prompt, length, temp):
    ck = torch.load(f'{K}/model.pt', weights_only=False)
    itos = ck['itos']; stoi = {c: i for i, c in enumerate(itos)}; V = len(itos)
    EMB, HID, LAYERS = ck['cfg']
    emb = nn.Embedding(V, EMB); lstm = nn.LSTM(EMB, HID, LAYERS, batch_first=True); fc = nn.Linear(HID, V)
    emb.load_state_dict(ck['emb']); lstm.load_state_dict(ck['lstm']); fc.load_state_dict(ck['fc'])
    emb.eval(); lstm.eval(); fc.eval()
    d = [stoi.get(c, V-1) for c in prompt]; h = None; out = []
    with torch.no_grad():
        for c in d: o, h = lstm(emb(torch.tensor([[c]])), h)
        for _ in range(length):
            c = topk_sample(fc(o[0, -1]), temp)
            out.append(itos[c]); o, h = lstm(emb(torch.tensor([[c]])), h)
    return prompt + ''.join(out)

def gen_gpt(prompt, length, temp):
    ck = torch.load(f'{K}/model_gpt_matched.pt', weights_only=False)
    itos = ck['itos']; stoi = {c: i for i, c in enumerate(itos)}; V = len(itos)
    EMB, NHEAD, NHID, NLAYERS, SEQ = ck['cfg']
    emb = nn.Embedding(V, EMB); pos = nn.Embedding(SEQ, EMB)
    layer = nn.TransformerEncoderLayer(EMB, NHEAD, NHID, batch_first=True, norm_first=True)
    tr = nn.TransformerEncoder(layer, NLAYERS); fc = nn.Linear(EMB, V)
    sd = ck['sd']
    emb.load_state_dict({'weight': sd['emb.weight']}); pos.load_state_dict({'weight': sd['pos.weight']})
    tr.load_state_dict({k[len('tr.'):]: v for k, v in sd.items() if k.startswith('tr.')})
    fc.load_state_dict({'weight': sd['fc.weight'], 'bias': sd['fc.bias']})
    emb.eval(); pos.eval(); tr.eval(); fc.eval()
    d = [stoi.get(c, V-1) for c in prompt][-SEQ:]; out = []
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([d]); p = torch.arange(len(d))
            mask = torch.triu(torch.full((len(d), len(d)), float('-inf')), 1)
            logits = fc(tr(emb(x) + pos(p), mask))[0, -1]
            c = topk_sample(logits, temp)
            d = (d + [c])[-SEQ:]; out.append(itos[c])
    return prompt + ''.join(out)

with open(f'{K}/samples_t03.md', 'w') as f:
    for name, fn in [('LSTM', gen_lstm), ('Transformer (matched)', gen_gpt)]:
        for i, prompt in enumerate(['static int __init ', 'static void ']):
            s = fn(prompt, 500, 0.3)
            f.write(f'### {name} — example {i+1} (prompt: `{prompt}`, temp 0.5 + top-k 40, 1200 chars)\n\n```c\n{s}\n```\n\n')
print('ok')
