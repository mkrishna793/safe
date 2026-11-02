"""Sparse Mixture-of-Experts decoder, DeepSeekMoE style.

The dense block replaces one SwiGLU FFN with

    s_t = W_r x_t,                    p_t = softmax(s_t)
    I_t = TopK_k(s_t + b)             (token-choice routing)
    g_{i,t} = p_{i,t} / sum_{j in I_t} p_{j,t}
    h_t = x_t + sum_{i in I_t} g_{i,t} FFN_i(x_t) + sum_j FFN^s_j(x_t)

so a token pays for ``top_k`` experts plus one always-on shared expert, while
the model stores ``n_experts`` of them.  Two load-balancing mechanisms are
exposed side by side:

  * the classical auxiliary loss  alpha * N * sum_i f_i P_i, and
  * the aux-loss-free bias  b_i <- b_i + u * sign(c_bar - c_i), which only
    shifts the top-k *selection* and therefore contributes no gradient.

The router logits/probs, selected ids, renormalised gates, per-expert load,
entropy, both balancing signals and the capacity-drop rate are returned in
``aux`` under ``route/*``.

Input is ``tokens`` of shape [B, S] (long); output is [B, S, vocab] logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, register, rms_norm, rope


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    """SwiGLU(x) = (x W_gate (.) SiLU(x W_up)) W_down -- one expert."""

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class CausalSelfAttention(nn.Module):
    """GQA + RoPE causal attention, identical in spirit to impl/dense.py."""

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

        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
        # Use a finite sentinel so every exposed tensor stays finite; -1e9
        # underflows to exactly zero after the softmax in float32.
        masked = scores.masked_fill(~mask, -1e9)
        probs = masked.softmax(-1)
        y = probs @ v
        y = y.transpose(1, 2).reshape(B, S, self.n_heads * self.d_head)
        return self.w_o(y), {"attn/probs": probs, "attn/scores": masked, "attn/mask": mask}


class MoEFFN(ArchModule):
    """Token-choice top-k MoE with one shared expert and both balancing rules.

    Aux shapes (B, S, N = n_experts, k = top_k, T = B*S):

        route/logits      [B, S, N]   unbiased router logits s_t
        route/probs       [B, S, N]   p_t = softmax(s_t)
        route/topk_ids    [B, S, k]   selected experts (descending logit)
        route/gates       [B, S, k]   renormalised gates over selected experts
        route/load        [N]         token-normalised load, sum = B*S
        route/load_frac   [N]         fraction of tokens routed, sum = top_k
        route/counts      [N]         raw assignment counts, sum = B*S*top_k
        route/entropy     [B, S]      per-token router entropy, nats
        route/entropy_mean scalar
        route/aux_loss    scalar      alpha * N * sum_i f_i P_i
        route/bias        [N]         aux-loss-free bias (not a parameter)
        route/bias_error  scalar      mean |c_i - c_bar| / c_bar (control error)
        route/max_vio     scalar      (max_i c_i - c_bar) / c_bar (paper's MaxVio)
        route/dropped     scalar      fraction of assignments dropped by capacity
        route/drop_mask   [B, S, k]   True where the assignment was kept
        route/capacity    scalar      C, or T*top_k when capacity is disabled
        mlp/shared_out    [B, S, d]   always-on shared-expert output
    """

    AUX_KEYS = (
        "route/logits",
        "route/probs",
        "route/topk_ids",
        "route/gates",
        "route/load",
        "route/entropy",
        "route/aux_loss",
        "route/bias",
        "route/dropped",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.n_experts = int(cfg["n_experts"])
        self.top_k = min(int(cfg["top_k"]), self.n_experts)
        self.n_shared = int(cfg.get("n_shared", 1))
        self.d_ff = int(cfg["d_ff"])

        self.router = nn.Linear(d, self.n_experts, bias=False)
        # Buffers, not parameters: the bias must never receive a gradient.
        self.register_buffer("expert_bias", torch.zeros(self.n_experts))
        self.experts = nn.ModuleList(SwiGLU(d, self.d_ff) for _ in range(self.n_experts))
        self.shared = nn.ModuleList(SwiGLU(d, self.d_ff) for _ in range(self.n_shared))

        self.capacity_factor = float(cfg.get("capacity_factor", 1.25))
        self.aux_alpha = float(cfg.get("aux_alpha", 0.01))
        self.bias_update_rate = float(cfg.get("bias_update_rate", 1e-3))
        self.bias_update = bool(cfg.get("bias_update", True))

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, d = x.shape
        T = B * S

        flat = x.reshape(T, d)
        logits = self.router(flat)                              # [T, N], differentiable
        probs = logits.softmax(-1)                              # [T, N]

        # ---- top-k selection ------------------------------------------------
        # The bias shifts *selection only*; p and g stay functions of s alone.
        bias = self.expert_bias.to(dtype=logits.dtype, device=logits.device)
        select_logits = logits + bias                           # [T, N]
        topk_ids = select_logits.topk(self.top_k, dim=-1).indices   # [T, k]
        gate = probs.gather(1, topk_ids)
        gate = gate / gate.sum(-1, keepdim=True).clamp_min(1e-9)     # renormalise

        # ---- load accounting (pre-drop; the aux loss sees raw routing) ------
        counts = torch.bincount(topk_ids.reshape(-1), minlength=self.n_experts)
        counts_f = counts.to(dtype=logits.dtype)
        f = counts_f / T                                        # sum_i f_i = top_k
        P = probs.mean(0)                                       # sum_i P_i = 1
        aux_loss = self.aux_alpha * self.n_experts * (f * P).sum()

        c_bar = float(T * self.top_k) / float(self.n_experts)
        max_vio = (counts_f.max() - c_bar) / max(c_bar, 1e-9)
        bias_error = (c_bar - counts_f).abs().mean() / max(c_bar, 1e-9)

        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1)     # [T]

        # ---- aux-loss-free bias update: dual ascent, no autograd ------------
        if self.training and self.bias_update and self.bias_update_rate > 0:
            with torch.no_grad():
                err = c_bar - counts_f
                self.expert_bias.add_(self.bias_update_rate * torch.sign(err))

        # ---- capacity / token dropping --------------------------------------
        capacity = float(T * self.top_k)
        if self.capacity_factor > 0:
            capacity = float(math.ceil(self.capacity_factor * T * self.top_k / self.n_experts))

        flat_ids = topk_ids.reshape(-1)                          # [T*k]
        token_of = torch.arange(T, device=x.device).unsqueeze(1).expand(T, self.top_k).reshape(-1)
        keep = torch.ones_like(flat_ids, dtype=torch.bool)
        for e in range(self.n_experts):
            pos = (flat_ids == e).nonzero(as_tuple=False).squeeze(1)
            if pos.numel() > capacity:
                keep[pos[int(capacity):]] = False
        # Order is token-major, so this is first-come-first-served in token order.

        dropped = 1.0 - keep.to(dtype=logits.dtype).mean()

        # ---- expert computation: only assigned tokens enter each expert -----
        out = torch.zeros_like(flat)
        flat_gate = gate.reshape(-1)
        for e, expert in enumerate(self.experts):
            pos = (flat_ids == e) & keep
            if not bool(pos.any()):
                continue
            tok = token_of[pos]
            y = expert(flat[tok]) * flat_gate[pos].unsqueeze(-1)
            out = out.index_add(0, tok, y)

        shared_out = flat.new_zeros(T, d)
        for expert in self.shared:
            shared_out = shared_out + expert(flat)

        y = x + (out + shared_out).reshape(B, S, d)

        aux = {
            "route/logits": logits.reshape(B, S, self.n_experts),
            "route/probs": probs.reshape(B, S, self.n_experts),
            "route/topk_ids": topk_ids.reshape(B, S, self.top_k),
            "route/gates": gate.reshape(B, S, self.top_k),
            "route/load": (counts_f / self.top_k).detach(),        # sum = B*S
            "route/load_frac": f.detach(),                         # sum = top_k
            "route/counts": counts.detach(),
            "route/entropy": entropy.reshape(B, S),
            "route/entropy_mean": entropy.mean().detach(),
            "route/aux_loss": aux_loss,
            "route/bias": self.expert_bias.detach().clone(),
            "route/bias_error": bias_error.detach(),
            "route/max_vio": max_vio.detach(),
            "route/dropped": dropped.detach(),
            "route/drop_mask": keep.reshape(B, S, self.top_k),
            "route/capacity": logits.new_tensor(capacity),
            "mlp/shared_out": shared_out.reshape(B, S, d),
        }
        return y, aux


class MoEBlock(ArchModule):
    """Pre-norm attention + pre-norm sparse MoE FFN, both with residuals."""

    AUX_KEYS = ("attn/probs", "route/logits", "route/load")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, cfg["n_heads"], cfg.get("n_kv_heads", cfg["n_heads"]), cfg["d_head"])
        self.norm2 = RMSNorm(d)
        self.moe = MoEFFN(cfg)

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, aux = self.attn(self.norm1(x))
        x = x + a
        m, route_aux = self.moe(self.norm2(x))
        x = x + m
        aux.update(route_aux)
        return x, aux


@register("moe")
class SparseMoETransformer(ArchModule):
    """Decoder-only Transformer whose FFNs are sparse top-k MoE layers."""

    AUX_KEYS = (
        "attn/probs",
        "route/logits",
        "route/probs",
        "route/topk_ids",
        "route/gates",
        "route/load",
        "route/entropy",
        "route/aux_loss",
        "route/bias",
        "route/dropped",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(MoEBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.lm_head.weight = self.embed.weight  # tied

    def forward(self, tokens: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.embed(tokens)
        aux: dict[str, torch.Tensor] = {"embed/out": x}
        total_aux = x.new_zeros(())
        loads: list[torch.Tensor] = []
        drops: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x, a = blk(x)
            for k, v in a.items():
                aux[f"layer{i}/{k}"] = v
            total_aux = total_aux + a["route/aux_loss"]
            loads.append(a["route/load"])
            drops.append(a["route/dropped"])
        x = self.norm_f(x)
        logits = self.lm_head(x)

        # Model-level summary: the objective term and cross-layer routing view.
        aux["route/aux_loss"] = total_aux
        aux["route/load"] = torch.stack(loads)          # [n_layers, n_experts]
        aux["route/dropped"] = torch.stack(drops).mean()
        aux["lm/logits"] = logits
        aux["hidden/final"] = x
        return logits, aux


def tiny_config(**over) -> ArchConfig:
    p = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        n_layers=2,
        d_ff=128,
        vocab=256,
        seq=32,
        n_experts=8,
        top_k=2,
        n_shared=1,
        capacity_factor=1.25,
        aux_alpha=0.01,
        bias_update_rate=1e-3,
    )
    p.update(over)
    return ArchConfig("moe-tiny", "sparse-moe", p)


# Importing the symbolic module registers the spec with core, so
# `get_spec("moe")` works even when only `archmath.impl.moe` was imported.
from ..symbolic import moe as _moe_spec  # noqa: E402,F401
