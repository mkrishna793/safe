"""Deep Dense Network (MLP / Residual Multi-Layer Perceptron).

A pure feed-forward architecture stack without spatial/sequence token mixing:

    x_{l+1} = x_l + SwiGLU(RMSNorm(x_l))
    logits  = W_vocab · RMSNorm(x_L)

Where SwiGLU is:
    SwiGLU(x) = (SiLU(x W_gate) ⊙ x W_up) W_down

In hidden-space representation theory, the pure MLP operates as a sequence of
coordinate manifold transforms where tokens are processed independently:
    ∂x_{l+1, t} / ∂x_{l, s} = 0 for t ≠ s

This makes the MLP the baseline orthogonal control against token-mixing
architectures (Transformers, SSMs, RNNs). Its hidden space reveals feature
expansion, intrinsic dimensionality contraction, and rank collapse dynamics.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward layer: (SiLU(x W_gate) ⊙ (x W_up)) W_down."""

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        g = self.w_gate(x)
        u = self.w_up(x)
        act = F.silu(g) * u
        out = self.w_down(act)
        aux = {
            "mlp/gate": g,
            "mlp/up": u,
            "mlp/act": act,
            "mlp/out": out,
        }
        return out, aux


class DenseMLPBlock(ArchModule):
    AUX_KEYS = ("mlp/out", "mlp/act", "mlp/gate")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        d_ff = cfg["d_ff"]
        self.norm = RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        norm_x = self.norm(x)
        ffn_out, aux = self.ffn(norm_x)
        out = x + ffn_out
        aux["mlp/block_out"] = out
        return out, aux


@register("mlp")
class MLPModel(ArchModule):
    AUX_KEYS = ("mlp/out", "lm/logits")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(DenseMLPBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.lm_head.weight = self.embed.weight

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
    p = dict(d_model=64, n_layers=3, d_ff=128, vocab=256, seq=32)
    p.update(over)
    return ArchConfig("mlp-tiny", "mlp", p)


@register_spec("mlp")
def mlp_spec() -> MathSpec:
    return MathSpec(
        name="mlp",
        family="mlp",
        summary="Deep Residual Multi-Layer Perceptron network with SwiGLU projections and RMSNorm.",
        equations=[
            ("swiglu", r"\mathrm{SwiGLU}(x) = \left(\mathrm{SiLU}(xW_g) \odot xW_u\right)W_d"),
            ("residual", r"x_{l+1} = x_l + \mathrm{SwiGLU}(\mathrm{RMSNorm}(x_l))"),
            ("ntk_kernel", r"\Theta(x, x') = \sum_{w \in \theta} \frac{\partial f(x)}{\partial w} \frac{\partial f(x')}{\partial w}^\top"),
            ("token_independence", r"\frac{\partial x_{l, t}}{\partial x_{l-1, s}} = 0 \quad \forall t \neq s"),
            ("objective", r"\mathcal{L} = -\sum_{t=1}^{S-1} \log \mathrm{softmax}(W_v \mathrm{RMSNorm}(x_t))_{x_{t+1}}"),
        ],
        assumptions=[
            "Strict point-wise token transformation without cross-sequence mixing.",
            "Pre-norm residual stream maintains bounded gradient norm.",
            "Weight tying between embedding and vocabulary head.",
        ],
        complexity={
            "params": "vocab*d_model + n_layers * (3 * d_model * d_ff + d_model) + d_model",
            "time_per_token": "n_layers * 6 * d_model * d_ff + 2 * d_model * vocab",
            "memory": "n_layers * d_model * seq",
        },
        invariants=[
            "outputs are strictly independent along sequence axis",
            "residual norm is finite for all finite inputs",
            "gradient flows directly through residual skip connection",
        ],
        references=[
            "Rumelhart, Hinton, & Williams 1986 (Backpropagation)",
            "Jacot et al. 2018 (Neural Tangent Kernel)",
            "Shazeer 2020 (GLU Variants)",
        ],
    )
