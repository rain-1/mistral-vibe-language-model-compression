"""Benchmark both BPE checkpoints on the original kernel corpus prefixes."""

import argparse
from bisect import bisect_right
import math
from pathlib import Path
import struct

import numpy as np
import torch
from tokenizers import Tokenizer

from train_bpe_full import SEQUENCE, Transformer


VERSIONS = ("2.0.40", "3.0", "4.0", "5.0", "6.0")
HALF = 1 << 63
QUARTER = HALF >> 1
MASK = (HALF << 1) - 1
TOTAL = 1 << 24
HEADER = struct.Struct(">IIIII")


class ArithmeticStream:
    """64-bit arithmetic intervals shared by the encoder and decoder."""

    def __init__(self):
        self.low = 0
        self.high = MASK

    def narrow(self, lower, upper, total):
        width = self.high - self.low + 1
        self.high = self.low + width * int(upper) // total - 1
        self.low += width * int(lower) // total

    def renormalize(self):
        while True:
            if self.high < HALF:
                base = 0
            elif self.low >= HALF:
                base = HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                base = QUARTER
            else:
                return
            self.low = (self.low - base) << 1
            self.high = ((self.high - base) << 1) | 1
            yield base


def encode(symbols, rows):
    stream = ArithmeticStream()
    bits = bytearray()
    pending = 0
    for symbol, cdf in zip(symbols, rows, strict=True):
        stream.narrow(cdf[symbol], cdf[symbol + 1], int(cdf[-1]))
        for base in stream.renormalize():
            if base == QUARTER:
                pending += 1
            else:
                bit = int(base == HALF)
                bits.append(bit)
                bits.extend(bytes([1 - bit]) * pending)
                pending = 0
    bit = int(stream.low >= QUARTER)
    bits.append(bit)
    bits.extend(bytes([1 - bit]) * (pending + 1))
    return np.packbits(np.frombuffer(bits, dtype=np.uint8)).tobytes()


def decode(payload, rows):
    stream = ArithmeticStream()
    bits = iter(np.unpackbits(np.frombuffer(payload, dtype=np.uint8)))
    code = 0
    for _ in range(64):
        code = (code << 1) | int(next(bits, 0))
    for cdf in rows:
        total = int(cdf[-1])
        target = ((code - stream.low + 1) * total - 1) // (stream.high - stream.low + 1)
        symbol = bisect_right(cdf, target) - 1
        stream.narrow(cdf[symbol], cdf[symbol + 1], total)
        for base in stream.renormalize():
            code = ((code - base) << 1) | int(next(bits, 0))
        yield symbol


def byte_patch(decoded, original):
    """Store a contiguous replacement when the BPE vocabulary loses input bytes."""
    prefix = 0
    limit = min(len(decoded), len(original))
    while prefix < limit and decoded[prefix] == original[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and decoded[-1 - suffix] == original[-1 - suffix]:
        suffix += 1
    return prefix, len(decoded) - prefix - suffix, original[prefix:len(original) - suffix]


@torch.inference_mode()
def perplexity(model, ids, device):
    # Match the original Transformer evaluation stride and partial-window rule.
    starts = list(range(0, len(ids) - SEQUENCE - 1, SEQUENCE))
    loss = torch.zeros((), dtype=torch.float64, device=device)
    for offset in range(0, len(starts), 16):
        batch = torch.tensor(
            [ids[start:start + SEQUENCE + 1] for start in starts[offset:offset + 16]],
            device=device,
        )
        logp = model(batch[:, :-1]).log_softmax(-1)
        loss -= logp.gather(-1, batch[:, 1:, None]).sum(dtype=torch.float64)
    return math.exp(loss.item() / (len(starts) * SEQUENCE))


@torch.inference_mode()
def compression(model, tokenizer, text, device):
    ids = tokenizer.encode(text).ids
    vocab = tokenizer.get_vocab_size()
    uniform = np.arange(vocab + 1, dtype=np.int32)
    decoded = []
    chunks = []
    for offset in range(0, len(ids), 16 * SEQUENCE):
        block = ids[offset:offset + 16 * SEQUENCE]
        # Padding follows all real tokens in each row and is causally invisible.
        padded = block + [0] * (-len(block) % SEQUENCE)
        inputs = torch.tensor(padded, device=device).view(-1, SEQUENCE)
        cdf = model(inputs[:, :-1]).softmax(-1).cumsum(-1)
        cdf = (cdf * (TOTAL - vocab)).round().to(torch.int32)
        cdf += torch.arange(1, vocab + 1, dtype=torch.int32, device=device)
        cdf[..., -1] = TOTAL
        cdf = torch.nn.functional.pad(cdf, (1, 0)).cpu().numpy()
        rows = []
        for index, boundaries in enumerate(cdf):
            remaining = min(SEQUENCE, len(block) - index * SEQUENCE)
            rows.append(uniform)
            rows.extend(boundaries[:remaining - 1])
        payload = encode(block, rows)
        chunks.append(struct.pack(">I", len(payload)) + payload)
        decoded.extend(decode(payload, rows))

    if decoded != ids:
        raise RuntimeError("Arithmetic token round trip failed")
    original = text.encode("utf-8")
    reconstructed = tokenizer.decode(decoded).encode("utf-8")
    position, remove, patch = byte_patch(reconstructed, original)
    # Count the header, payload length per batch, byte padding, and repair bytes.
    archive = HEADER.pack(len(ids), len(chunks), position, remove, len(patch))
    archive += patch + b"".join(chunks)
    _, _, position, remove, patch_size = HEADER.unpack_from(archive)
    restored = (
        reconstructed[:position] + archive[HEADER.size:HEADER.size + patch_size]
        + reconstructed[position + remove:]
    )
    if restored != original:
        raise RuntimeError("UTF-8 byte round trip failed")
    return len(original), len(archive), patch_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("bpe_tokenizer.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    texts = {}
    for version in VERSIONS:
        with (args.corpus_dir / f"corpus_{version}.txt").open(encoding="utf-8") as file:
            texts[version] = file.read(300_000)

    print("checkpoint,step,version,ppl,raw_bytes,coded_bytes,patch_bytes,bits_per_byte", flush=True)
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = Transformer(tokenizer.get_vocab_size()).to(args.device).eval()
        model.load_state_dict(checkpoint["sd"])
        for version, text in texts.items():
            ppl = perplexity(model, tokenizer.encode(text).ids, args.device)
            raw, coded, patch = compression(model, tokenizer, text[:100_000], args.device)
            print(
                f"{path.name},{checkpoint.get('step', '')},{version},{ppl:.6f},"
                f"{raw},{coded},{patch},{8 * coded / raw:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
