import torch, torch.nn as nn, time
torch.set_num_threads(8)
K = '/home/riv/kernels'
N = 100_000
SEQ = 256

ck = torch.load(f'{K}/model_gpt.pt', weights_only=False)
itos = ck['itos']; stoi = {c: i for i, c in enumerate(itos)}; V = len(itos)
EMB, NHEAD, NHID, NLAYERS, _ = ck['cfg']
emb = nn.Embedding(V, EMB); pos = nn.Embedding(SEQ, EMB)
layer = nn.TransformerEncoderLayer(EMB, NHEAD, NHID, batch_first=True, norm_first=True)
tr = nn.TransformerEncoder(layer, NLAYERS); fc = nn.Linear(EMB, V)
sd = ck['sd']
emb.load_state_dict({'weight': sd['emb.weight']})
pos.load_state_dict({'weight': sd['pos.weight']})
tr.load_state_dict({k[len('tr.'):]: v for k, v in sd.items() if k.startswith('tr.')})
fc.load_state_dict({'weight': sd['fc.weight'], 'bias': sd['fc.bias']})
emb.eval(); pos.eval(); tr.eval(); fc.eval()

class Enc:
    def __init__(self):
        self.low = 0; self.high = (1 << 64) - 1; self.pending = 0; self.out = bytearray()
    def _emit(self, bit):
        self.out.append(bit)
        while self.pending > 0:
            self.out.append(1 - bit); self.pending -= 1
    def _shift(self):
        self.low = (self.low << 1) & ((1 << 64) - 1)
        self.high = ((self.high << 1) | 1) & ((1 << 64) - 1)
    def update(self, lo, hi, tot):
        S = 1 << 32
        tot_i = max(int(round(tot * S)), 1)
        lo_i = int(round(lo * S)); hi_i = int(round(hi * S))
        rng = self.high - self.low + 1
        self.high = self.low + (rng * hi_i) // tot_i - 1
        self.low = self.low + (rng * lo_i) // tot_i
        while True:
            if self.high < (1 << 63):
                self._emit(0); self._shift()
            elif self.low >= (1 << 63):
                self._emit(1); self.low -= (1 << 63); self.high -= (1 << 63); self._shift()
            elif self.low >= (1 << 62) and self.high < 3 * (1 << 62):
                self.pending += 1; self.low -= (1 << 62); self.high -= (1 << 62); self._shift()
            else:
                break
    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < (1 << 62) else 1)
        return len(self.out)  # bits

def encode(text):
    d = [stoi.get(c, V-1) for c in text]
    enc = Enc()
    with torch.no_grad():
        for i in range(0, len(d) - 1, SEQ):
            chunk = d[i:i+SEQ]
            x = torch.tensor([chunk])
            p = torch.arange(len(chunk))
            h = emb(x) + pos(p)
            mask = torch.triu(torch.full((len(chunk), len(chunk)), float('-inf')), 1)
            logp = torch.log_softmax(fc(tr(h, mask))[0], -1)
            for t in range(len(chunk) - 1):
                probs = torch.exp(logp[t].double())
                c = torch.cumsum(probs, 0)
                tot = c[-1].item()
                sym = chunk[t+1]
                lo = c[sym-1].item() if sym > 0 else 0.0
                enc.update(lo, c[sym].item(), tot)
    return enc.finish(), len(d) - 1

for v in ['2.0.40', '3.0', '4.0', '5.0', '6.0']:
    with open(f'{K}/corpus_{v}.txt') as f: t = f.read()[:N]
    t0 = time.time()
    nbits, nch = encode(t)
    bpc = nbits / nch
    print(f'linux-{v}: GPT-arithmetic {nbits/8/1e6:.3f} MB / {nch} chars = {bpc:.3f} bits/char (ratio {8/bpc:.2f}x) [{time.time()-t0:.0f}s]', flush=True)
