"""Titans-style neural test-time memory.

Short-term memory = attention.  Long-term memory = a matrix M updated at test
time by a surprise-driven learning rule.  A gate mixes the two reads.

    M_t = (1 - alpha_t) M_{t-1} + alpha_t * (v_t - M_{t-1} k_t) k_t^T     (delta rule)
    read_t = M_{t-1} q_t
    y_t = g_t * attn_t + (1 - g_t) * read_t,    g_t = sigma(W_g [attn_t; read_t])
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm, rope

# reuse dense attention pieces
from .dense import CausalSelfAttention, RMSNorm, SwiGLU


class NeuralMemory(nn.Module):
    """A matrix-valued fast weight updated by a delta rule with forgetting."""

    def __init__(self, d: int, d_mem: int) -> None:
        super().__init__()
        self.d = d
        self.d_mem = d_mem
        self.w_k = nn.Linear(d, d_mem, bias=False)
        self.w_v = nn.Linear(d, d_mem, bias=False)
        self.w_q = nn.Linear(d, d_mem, bias=False)
        self.w_o = nn.Linear(d_mem, d, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.decay = nn.Parameter(torch.tensor(0.99))
        self.momentum = nn.Parameter(torch.tensor(0.9))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, _ = x.shape
        k = self.w_k(x)  # [B,S,m]
        v = self.w_v(x)
        q = self.w_q(x)

        M = torch.zeros(B, self.d_mem, self.d_mem, device=x.device, dtype=x.dtype)
        mom = torch.zeros_like(M)
        reads = []
        surprises = []
        deltas = []
        alpha = torch.sigmoid(self.alpha)
        decay = torch.sigmoid(self.decay)
        momentum = torch.sigmoid(self.momentum)

        for t in range(S):
            kt = k[:, t]  # [B,m]
            vt = v[:, t]
            qt = q[:, t]
            # surprise = ||M k - v||
            Mk = torch.bmm(M, kt.unsqueeze(-1)).squeeze(-1)  # [B,m]
            surprise = (Mk - vt).pow(2).mean(-1)  # [B]
            # delta rule gradient on (M k - v)
            resid = Mk - vt  # [B,m]
            grad = torch.bmm(resid.unsqueeze(-1), kt.unsqueeze(1))  # [B,m,m]
            mom = momentum * mom + (1 - momentum) * grad
            # forgetting (weight decay) + update
            M = decay * M - alpha * mom
            reads.append(torch.bmm(M, qt.unsqueeze(-1)).squeeze(-1))  # read uses updated M
            surprises.append(surprise)
            deltas.append(grad.flatten(1).norm(dim=-1).mean())

        read = torch.stack(reads, dim=1)  # [B,S,m]
        read = self.w_o(read)
        surprise_t = torch.stack(surprises, dim=1)  # [B,S]
        aux = {
            "state/memory": M,
            "state/surprise": surprise_t,
            "state/memory_delta": torch.tensor([float(dd.detach()) if torch.is_tensor(dd) else float(dd) for dd in deltas], device=x.device, dtype=x.dtype),
            "gate/memory": alpha.detach().expand(B, S),
        }
        return read, aux


class TitansBlock(ArchModule):
    AUX_KEYS = ("attn/probs", "state/memory", "gate/memory")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, cfg["n_heads"], cfg.get("n_kv_heads", cfg["n_heads"]), cfg["d_head"])
        self.mem = NeuralMemory(d, cfg.get("memory_dim", d // 2))
        self.gate = nn.Linear(2 * d, d)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, cfg["d_ff"])

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, aux = self.attn(self.norm1(x))
        mread, maux = self.mem(self.norm1(x))
        g = torch.sigmoid(self.gate(torch.cat([a, mread], dim=-1)))  # [B,S,d]
        mixed = g * a + (1 - g) * mread
        x = x + mixed
        x = x + self.mlp(self.norm2(x))
        aux.update(maux)
        aux["gate/memory"] = g.mean(dim=-1)
        return x, aux


@register("titans")
class TitansTransformer(ArchModule):
    AUX_KEYS = ("attn/probs", "state/memory", "gate/memory", "lm/logits")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(TitansBlock(cfg) for _ in range(cfg["n_layers"]))
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
    p = dict(d_model=64, n_heads=4, n_kv_heads=2, d_head=16, n_layers=2, d_ff=128,
             vocab=256, seq=16, memory_dim=32)
    p.update(over)
    return ArchConfig("titans-tiny", "titans", p)


@register_spec("titans")
def titans_spec() -> MathSpec:
    return MathSpec(
        name="titans",
        family="titans",
        summary="Titans: attention as short-term memory, a neural matrix memory updated at test time by a surprise-driven delta rule, mixed by a gate.",
        equations=[
            ("memory_update", r"M_t=(1-\alpha_t)M_{t-1}+\alpha_t\,(v_t-M_{t-1}k_t)k_t^\top"),
            ("surprise", r"\mathrm{surprise}_t=\|M_{t-1}k_t-v_t\|^2"),
            ("read", r"r_t=M_{t-1}q_t"),
            ("gate", r"g_t=\sigma(w_g^\top[a_t;r_t]),\quad y_t=g_t a_t+(1-g_t)r_t"),
            ("forgetting", r"M_t \leftarrow \gamma M_t,\quad \gamma\in(0,1)"),
        ],
        assumptions=[
            "The memory is a fast weight: its parameters are the state, updated by a learning rule, not by backprop through the whole sequence.",
            "Surprise measures how poorly the current memory predicts the new value; high surprise -> large update.",
            "Attention handles precise short-range retrieval; memory handles compressed long-range associations.",
        ],
        complexity={
            "params": "d_model*d_model*n_layers*4 + d_model*d_model//2",
            "time_per_token": "d_model*d_model + d_model*d_model//2",
            "memory_state": "d_model*d_model//2",
        },
        invariants=[
            "memory norm finite",
            "gate values in [0,1]",
            "surprise non-negative",
        ],
        references=[
            "Behrouz et al. 2025 (arXiv:2501.00663) -- Titans",
        ],
    )

def associative_recall_demo(steps: int = 100, seq: int = 16, vocab: int = 16, seed: int = 0) -> dict[str, float]:
    """Tiny next-token probe: trains the Titans model and reports the loss curve."""
    torch.manual_seed(seed)
    cfg = tiny_config(d_model=32, n_heads=2, d_head=16, d_ff=64, n_layers=1, vocab=vocab, seq=seq, memory_dim=16)
    model = TitansTransformer(cfg)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(steps):
        toks = torch.randint(0, vocab, (4, seq))
        opt.zero_grad()
        out, aux = model(toks)
        logits = out.float()
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, vocab), toks[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return {"final_loss": losses[-1] if losses else float("nan"),
            "initial_loss": losses[0] if losses else float("nan"),
            "steps": float(steps)}
