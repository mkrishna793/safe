"""Dense decoder-only Transformer -- the reference architecture.

Everything else in this package is a deviation from this file.  Read it first.

    x_{l+1} = x_l + Attn(LN(x_l))
    x_{l+1} = x_{l+1} + MLP(LN(x_{l+1}))
    logits  = W_vocab · LN(x_L)

with GQA, RoPE, RMSNorm, SwiGLU, pre-norm.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm, rope


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    """SwiGLU(x) = (x W_gate ⊙ SiLU(x W_up)) W_down, with 8/3 d expansion."""

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class CausalSelfAttention(nn.Module):
    """Multi-head attention with grouped KV heads and rotary positions."""

    def __init__(self, d: int, n_heads: int, n_kv_heads: int, d_head: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.theta = theta
        self.group = n_heads // n_kv_heads
        self.w_q = nn.Linear(d, n_heads * d_head, bias=False)
        self.w_k = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.w_v = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.w_o = nn.Linear(n_heads * d_head, d, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, _ = x.shape
        q = self.w_q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        pos = torch.arange(S, device=x.device)
        q = rope(q, pos, self.theta)
        k = rope(k, pos, self.theta)

        # GQA: repeat KV heads to match query heads
        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, -1e9)
        probs = scores.softmax(-1)
        y = probs @ v
        y = y.transpose(1, 2).reshape(B, S, self.n_heads * self.d_head)
        return self.w_o(y), {"attn/probs": probs, "attn/scores": scores}


class DenseBlock(ArchModule):
    AUX_KEYS = ("attn/probs", "mlp/out")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, cfg["n_heads"], cfg.get("n_kv_heads", cfg["n_heads"]), cfg["d_head"])
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, cfg["d_ff"])

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, aux = self.attn(self.norm1(x))
        x = x + a
        m = self.mlp(self.norm2(x))
        x = x + m
        aux["mlp/out"] = m
        return x, aux


@register("dense")
class DenseTransformer(ArchModule):
    AUX_KEYS = ("attn/probs", "lm/logits")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(DenseBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.lm_head.weight = self.embed.weight  # tied

    def forward(self, tokens: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.embed(tokens)
        aux: dict[str, torch.Tensor] = {"embed/out": x}
        for i, blk in enumerate(self.blocks):
            x, a = blk(x)
            for k, v in a.items():
                aux[f"layer{i}/{k}"] = v
        x = self.norm_f(x)
        logits = self.lm_head(x)
        aux["lm/logits"] = logits
        aux["hidden/final"] = x
        return logits, aux


def tiny_config(**over) -> ArchConfig:
    p = dict(d_model=64, n_heads=4, n_kv_heads=2, d_head=16, n_layers=2, d_ff=128, vocab=256, seq=32)
    p.update(over)
    return ArchConfig("dense-tiny", "dense", p)


@register_spec("dense")
def dense_spec() -> MathSpec:
    return MathSpec(
        name="dense",
        family="dense",
        summary="Pre-norm decoder-only Transformer with GQA, RoPE, RMSNorm and SwiGLU.",
        equations=[
            ("rmsnorm", r"\mathrm{RMSNorm}(x) = g \odot x / \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}"),
            ("qkv", r"Q = X W_Q,\quad K = X W_K,\quad V = X W_V,\quad W_K,W_V \in \mathbb{R}^{d \times h_{kv} d_h}"),
            ("rope", r"\mathrm{RoPE}(x, t)_{2i} = x_{2i}\cos(t\theta^{-2i/d_h}) - x_{2i+1}\sin(\cdot),\quad \theta=10^4"),
            ("gqa", r"K^{(h)} = K^{(\lfloor h / (h/h_{kv}) \rfloor)},\quad \text{repeat KV heads}"),
            ("attention", r"\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V,\quad M_{ij} = -\infty \text{ for } j>i"),
            ("swiglu", r"\mathrm{SwiGLU}(x) = \left(\mathrm{SiLU}(xW_g) \odot xW_u\right)W_d"),
            ("residual", r"x_{l+1} = x_l + \mathrm{Attn}(\mathrm{LN}(x_l)),\quad x_{l+1} \leftarrow x_{l+1} + \mathrm{MLP}(\mathrm{LN}(x_{l+1}))"),
            ("objective", r"\mathcal{L} = -\sum_{t=1}^{S-1} \log \mathrm{softmax}(W_v \mathrm{LN}(x_t))_{x_{t+1}}"),
        ],
        assumptions=[
            "Causal mask makes the map autoregressive: y_t depends only on x_{<=t}.",
            "RoPE is relative: <RoPE(q,t), RoPE(k,s)> depends on t-s.",
            "Weight tying: W_vocab = W_embed.",
        ],
        complexity={
            "params": "vocab*d + n_layers*(d*d + 2*d*n_kv_heads*d_head + d*d + 3*d*d_ff) + d",
            "time_per_token": "n_layers*(2*d*d + 4*d*n_kv_heads*d_head + 6*d*d_ff) + 2*d*vocab",
            "kv_bytes_per_token": "2*n_layers*n_kv_heads*d_head*2",
            "attention_flops_per_token": "2*n_layers*n_heads*d_head*S",
        },
        invariants=[
            "attention rows sum to 1",
            "output shape equals input shape for each block",
            "logits depend only on tokens at or before their position",
            "residual stream norm is finite for finite inputs",
        ],
        references=["Vaswani et al. 2017 (arXiv:1706.03762)", "Shazeer 2019 (GQA)", "Su et al. 2021 (RoPE)", "Zhang & Sennrich 2019 (RMSNorm)"],
    )
