"""Train the existing BPE Transformer architecture on a Linux source archive."""

import argparse
import hashlib
import math
from pathlib import Path, PurePosixPath
import tarfile
import time

import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer


SEQUENCE = 256
SEED = 3
SOURCE_EXTENSIONS = {
    ".c", ".h", ".s", ".txt", ".rst", ".dts", ".dtsi", ".yaml", ".json",
    ".pl", ".py", ".sh",
}
SOURCE_NAMES = {
    "Kbuild", "COPYING", "README", "CREDITS", "MAINTAINERS", "TODO",
    "REPORTING-BUGS",
}


def prepare(archive, tokenizer):
    """Use the original source filter and split by path before tokenizing."""
    counts = {"train": [0, 0, 0], "val": [0, 0, 0]}
    pending = []
    with (
        tarfile.open(archive, "r|*") as source,
        open("train.bin.tmp", "wb") as train_file,
        open("val.bin.tmp", "wb") as val_file,
    ):
        outputs = {"train": train_file, "val": val_file}

        def flush():
            for (split, size, _), encoding in zip(
                pending, tokenizer.encode_batch([entry[2] for entry in pending])
            ):
                ids = np.asarray(encoding.ids, dtype="<u2")
                ids.tofile(outputs[split])
                counts[split][0] += 1
                counts[split][1] += size
                counts[split][2] += len(ids)
            pending.clear()

        for member in source:
            path = PurePosixPath(member.name)
            if not member.isfile():
                continue
            if (
                path.suffix.lower() not in SOURCE_EXTENSIONS
                and path.name not in SOURCE_NAMES
                and not path.name.startswith(("Kconfig", "Makefile"))
            ):
                continue
            relative_path = "/".join(path.parts[1:])
            digest = hashlib.sha256(relative_path.encode()).digest()
            split = "val" if int.from_bytes(digest[:4], "big") % 100 == 0 else "train"
            with source.extractfile(member) as file:
                content = file.read().decode("utf-8", errors="ignore").replace(
                    "\r\n", "\n"
                ).replace("\r", "\n")
            pending.append((split, len(content.encode("utf-8")), content + "\n"))
            if len(pending) == 32:
                flush()
        flush()

    for split, (files, size, tokens) in counts.items():
        if tokens <= SEQUENCE:
            raise ValueError(f"Insufficient {split} tokens: {tokens}")
        print(f"{split}: {files:,} files, {size:,} bytes, {tokens:,} tokens", flush=True)
    for split in counts:
        Path(f"{split}.bin.tmp").replace(f"{split}.bin")


class Transformer(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 384)
        self.pos = nn.Embedding(SEQUENCE, 384)
        layer = nn.TransformerEncoderLayer(
            384, 8, 384, batch_first=True, dropout=0.1, norm_first=True
        )
        self.tr = nn.TransformerEncoder(layer, 6, enable_nested_tensor=False)
        self.fc = nn.Linear(384, vocab)
        self.register_buffer(
            "mask", torch.ones(SEQUENCE, SEQUENCE, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, inputs):
        length = inputs.shape[1]
        positions = torch.arange(length, device=inputs.device)
        hidden = self.emb(inputs) + self.pos(positions)
        return self.fc(self.tr(hidden, self.mask[:length, :length], is_causal=True))


def batch(data, starts):
    tokens = data[starts[:, None] + np.arange(SEQUENCE + 1)].astype(np.int64)
    tokens = torch.from_numpy(tokens).to("cuda")
    return tokens[:, :-1], tokens[:, 1:]


def loss_for(model, data, starts):
    x, y = batch(data, starts)
    with torch.autocast("cuda", dtype=torch.float16):
        return nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())


@torch.no_grad()
def validate(model, data, batch_size):
    model.eval()
    starts = np.random.default_rng(SEED).integers(
        len(data) - SEQUENCE, size=256
    )
    total_loss = 0.0
    for offset in range(0, len(starts), batch_size):
        selected = starts[offset:offset + batch_size]
        total_loss += loss_for(model, data, selected).item() * len(selected)
    model.train()
    return total_loss / len(starts)


def train(args, tokenizer):
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    data = np.memmap("train.bin", dtype="<u2", mode="r")
    valid = np.memmap("val.bin", dtype="<u2", mode="r")
    model = Transformer(tokenizer.get_vocab_size()).cuda()
    if args.baseline:
        model.load_state_dict(torch.load(args.baseline, map_location="cpu")["sd"])
        val_loss = validate(model, valid, args.batch_size)
        print(f"baseline val loss {val_loss:.4f} ppl {math.exp(val_loss):.2f}", flush=True)
        torch.manual_seed(SEED)
        model = Transformer(tokenizer.get_vocab_size()).cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")
    checkpoint = Path(args.checkpoint)
    step = 0
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu")
        if state["batch_size"] != args.batch_size or state["train_tokens"] != len(data):
            raise ValueError("Resume requires the original batch size and token corpus")
        model.load_state_dict(state["sd"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        step = state["step"]

    windows = (len(data) - 1) // SEQUENCE
    batches = math.ceil(windows / args.batch_size)
    stop = batches * args.epochs
    if args.steps is not None:
        stop = min(stop, step + args.steps)
    print(
        f"CUDA {torch.cuda.get_device_name()}, batch {args.batch_size}, "
        f"{windows:,} windows per epoch, steps {step:,} to {stop:,}",
        flush=True,
    )
    start_time = time.monotonic()
    total_loss = torch.zeros((), device="cuda")
    logged_batches = 0
    seen = 0
    while step < stop:
        epoch, offset = divmod(step, batches)
        order = np.random.default_rng(SEED + epoch).permutation(windows)
        for index in range(offset, batches):
            starts = order[index * args.batch_size:(index + 1) * args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_for(model, data, starts * SEQUENCE)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.detach()
            logged_batches += 1
            step += 1
            seen += len(starts) * SEQUENCE
            if step == 1 or step % 50 == 0 or step == stop:
                train_loss = total_loss.item() / logged_batches
                elapsed = time.monotonic() - start_time
                print(
                    f"step {step:,}/{stop:,} train loss {train_loss:.4f} "
                    f"tokens/s {seen / elapsed:,.0f} "
                    f"peak GPU MiB {torch.cuda.max_memory_allocated() / 2**20:,.0f}",
                    flush=True,
                )
                total_loss.zero_()
                logged_batches = 0
            if step % args.save_every == 0 or step == stop:
                val_loss = validate(model, valid, args.batch_size)
                print(f"step {step:,} val loss {val_loss:.4f} ppl {math.exp(val_loss):.2f}", flush=True)
                temporary = checkpoint.with_suffix(".tmp")
                torch.save({
                    "sd": model.state_dict(),
                    "cfg": (384, 8, 384, 6, SEQUENCE, tokenizer.get_vocab_size()),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "batch_size": args.batch_size,
                    "train_tokens": len(data),
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                }, temporary)
                temporary.replace(checkpoint)
            if step == stop:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, metavar="ARCHIVE")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, help="cap additional optimizer steps")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint", default="model_bpe_full.pt")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("batch_size", "epochs", "steps", "save_every"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.baseline and args.resume:
        parser.error("--baseline and --resume cannot be combined")
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file("bpe_tokenizer.json")
    if args.prepare:
        prepare(args.prepare, tokenizer)
    else:
        train(args, tokenizer)


if __name__ == "__main__":
    main()
