"""Minimal decoder-only transformer for C source code.

Components: causal MHA with RoPE, pre-norm RMSNorm, SwiGLU FFN,
tied input/output embeddings, next-token prediction via PyTorch SDPA.

Deliberately small and readable; architecture pieces are modular so
experiments can swap them (see model/recurrent.py later).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    intermediate_size: int = 704
    context_length: int = 1024
    normalization: str = "rmsnorm"
    activation: str = "swiglu"
    position_encoding: str = "rope"
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    dropout: float = 0.0
    bias: bool = False
    n_loop: int = 1  # apply the block stack n_loop times (looped/shared layers)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ModelConfig":
        fields = {f for f in ModelConfig.__dataclass_fields__}
        return ModelConfig(**{k: v for k, v in d.items() if k in fields})

    def n_params(self, model: "Transformer") -> int:
        return sum(p.numel() for p in model.parameters())


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute in fp32 for stability, cast back
        x32 = x.float()
        x_normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_normed * self.weight.float()).to(x.dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D) -> rotate pairs (even/odd)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # cos/sin: (T, D/2)
    cos = cos.view(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.view(1, 1, sin.shape[0], sin.shape[1])
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out


def precompute_rope_cache(context_length: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(context_length, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, D/2)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos.to(dtype), sin.to(dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_embd = cfg.n_embd
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.register_buffer("_rope_dummy", torch.empty(0))  # keeps device moves simple

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        norm = RMSNorm if cfg.normalization == "rmsnorm" else nn.LayerNorm
        self.ln1 = norm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = norm(cfg.n_embd)
        self.mlp = SwiGLU(cfg) if cfg.activation == "swiglu" else nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=cfg.bias),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd) if cfg.normalization == "rmsnorm" else nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        # RoPE cache is buffer-like; registered lazily per device
        self.register_buffer("rope_cos", torch.zeros(0), persistent=False)
        self.register_buffer("rope_sin", torch.zeros(0), persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 1.0 / math.sqrt(m.in_features)
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.weight.shape[1]))

    def ensure_rope_cache(self, device, dtype):
        head_dim = self.cfg.n_embd // self.cfg.n_head
        if self.rope_cos.numel() == 0 or self.rope_cos.device != device:
            cos, sin = precompute_rope_cache(
                self.cfg.context_length, head_dim, self.cfg.rope_theta, device, dtype
            )
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. Returns logits (B,T,V) and loss if targets given."""
        B, T = idx.shape
        assert T <= self.cfg.context_length, f"T={T} > context {self.cfg.context_length}"
        self.ensure_rope_cache(idx.device, torch.float32)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        x = self.tok_emb(idx)
        for _ in range(max(1, self.cfg.n_loop)):
            for block in self.blocks:
                x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None):
        """Greedy/sampled decoding. idx: (B, T) prompt."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


def count_parameters(cfg: ModelConfig) -> int:
    """Analytic parameter count (independent of torch) for config tables."""
    V, d, L, H, I = cfg.vocab_size, cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.intermediate_size
    per_block = 3 * d * d + d * d + 3 * d * I  # qkv + proj + swiglu (no bias)
    emb = V * d
    ln = 2 * L * d + d  # per-block ln1/ln2 + final (rmsnorm: 1 weight each)
    total = emb + L * per_block + ln
    # tied head adds nothing extra
    return total


if __name__ == "__main__":
    for name, cfg in [
        ("micro", ModelConfig(n_layer=4, n_embd=128, intermediate_size=352)),
        ("tiny", ModelConfig(n_layer=4, n_embd=256, intermediate_size=704)),
        ("small", ModelConfig(n_layer=6, n_embd=384, intermediate_size=1024)),
        ("medium", ModelConfig(n_layer=8, n_embd=512, intermediate_size=1408)),
        ("large", ModelConfig(n_layer=12, n_embd=512, intermediate_size=1408)),
    ]:
        m = Transformer(cfg)
        print(f"{name:8s} analytic={count_parameters(cfg)/1e6:6.2f}M  actual={sum(p.numel() for p in m.parameters())/1e6:6.2f}M")
