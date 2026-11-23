import torch, torch.nn as nn
from tokenizers import Tokenizer
torch.set_num_threads(8)
K = '/home/riv/kernels'
ck = torch.load(f'{K}/model_bpe.pt', weights_only=False)
EMB, NHEAD, NHID, NLAYERS, SEQ, V = ck['cfg']
tok = Tokenizer.from_file(f'{K}/bpe_tokenizer.json')
emb = nn.Embedding(V, EMB); pos = nn.Embedding(SEQ, EMB)
layer = nn.TransformerEncoderLayer(EMB, NHEAD, NHID, batch_first=True, norm_first=True)
tr = nn.TransformerEncoder(layer, NLAYERS); fc = nn.Linear(EMB, V)
sd = ck['sd']
emb.load_state_dict({'weight': sd['emb.weight']}); pos.load_state_dict({'weight': sd['pos.weight']})
tr.load_state_dict({k[len('tr.'):]: v for k, v in sd.items() if k.startswith('tr.')})
fc.load_state_dict({'weight': sd['fc.weight'], 'bias': sd['fc.bias']})
emb.eval(); pos.eval(); tr.eval(); fc.eval()

def gen(prompt, n, temp=0.7, k=40):
    d = tok.encode(prompt).ids[-SEQ:]; out = []
    with torch.no_grad():
        for _ in range(n):
            x = torch.tensor([d]); p = torch.arange(len(d))
            mask = torch.triu(torch.full((len(d), len(d)), float('-inf')), 1)
            logits = fc(tr(emb(x) + pos(p), mask))[0, -1] / temp
            v, idx = torch.topk(logits, k)
            c = idx[torch.multinomial(torch.softmax(v, -1), 1).item()].item()
            d = (d + [c])[-SEQ:]; out.append(c)
    return tok.decode(out)

with open(f'{K}/samples_bpe.md', 'w') as f:
    for i, prompt in enumerate(['static int __init ', 'static void ']):
        s = gen(prompt, 400)
        f.write(f'### BPE Transformer — example {i+1} (prompt: `{prompt}`, temp 0.7 top-k 40)\n\n```c\n{prompt}{s}\n```\n\n')
print('ok')
