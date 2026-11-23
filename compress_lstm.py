import torch, torch.nn as nn, math, time
torch.set_num_threads(8)
K = '/home/riv/kernels'
N = 100_000
SEQ = 100

ck = torch.load(f'{K}/model.pt', weights_only=False)
itos = ck['itos']; stoi = {c: i for i, c in enumerate(itos)}; V = len(itos)
EMB, HID, LAYERS = ck['cfg']
emb = nn.Embedding(V, EMB); lstm = nn.LSTM(EMB, HID, LAYERS, batch_first=True); fc = nn.Linear(HID, V)
emb.load_state_dict(ck['emb']); lstm.load_state_dict(ck['lstm']); fc.load_state_dict(ck['fc'])
emb.eval(); lstm.eval(); fc.eval()

# Standard arithmetic encoder with 64-bit precision and pending-bit carry handling
class Enc:
    def __init__(self):
        self.low = 0
        self.high = (1 << 64) - 1
        self.pending = 0
        self.out = bytearray()
    def _emit(self, bit):
        self.out.append(bit)
        while self.pending > 0:
            self.out.append(1 - bit)
            self.pending -= 1
    def update(self, lo, hi, tot):
        S = 1 << 32
        tot_i = max(int(round(tot * S)), 1)
        lo_i = int(round(lo * S)); hi_i = int(round(hi * S))
        rng = self.high - self.low + 1
        self.high = self.low + (rng * hi_i) // tot_i - 1
        self.low = self.low + (rng * lo_i) // tot_i
        while True:
            if self.high < (1 << 63):
                self._emit(0); self.pending += 0
                self._shift()
            elif self.low >= (1 << 63):
                self._emit(1)
                self.low -= (1 << 63); self.high -= (1 << 63)
                self._shift()
            elif self.low >= (1 << 62) and self.high < 3 * (1 << 62):
                self.pending += 1
                self.low -= (1 << 62); self.high -= (1 << 62)
                self._shift()
            else:
                break
    def _shift(self):
        self.low = (self.low << 1) & ((1 << 64) - 1)
        self.high = ((self.high << 1) | 1) & ((1 << 64) - 1)
    def finish(self):
        self.pending += 1
        if self.low < (1 << 62):
            self._emit(0)
        else:
            self._emit(1)
        return len(self.out)  # each element is one bit

def encode(text):
    d = [stoi.get(c, V-1) for c in text]
    enc = Enc()
    with torch.no_grad():
        for i in range(0, len(d) - 1, SEQ):
            chunk = d[i:i+SEQ]
            x = torch.tensor([chunk])
            out, _ = lstm(emb(x))
            logp = torch.log_softmax(fc(out)[0], -1)
            for t in range(len(chunk) - 1):
                p = torch.exp(logp[t].double())
                c = torch.cumsum(p, 0)
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
    print(f'linux-{v}: LSTM-arithmetic {nbits/8/1e6:.3f} MB / {nch} chars = {bpc:.3f} bits/char (ratio {8/bpc:.2f}x) [{time.time()-t0:.0f}s]', flush=True)
