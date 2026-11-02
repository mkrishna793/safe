"""Spiking / event-driven transformer block with leaky integrate-and-fire neurons.

The mathematical object is a discrete LIF neuron unrolled over ``T`` steps:

    V_t = beta * V_{t-1} + W x_t - V_th * s_{t-1}
    s_t = Theta(V_t - V_th)                  (forward)
    r   = (1/T) sum_t s_t                    (rate decode)

``Theta`` is the Heaviside step.  Its true derivative is zero almost everywhere
(and a Dirac mass at the threshold), so backpropagation would receive no
gradient.  We use the standard surrogate-gradient trick: the forward pass is
the hard step, the backward pass is a smooth sigmoid/arctan derivative with
slope ``alpha``.  This is a *biased* gradient, not the gradient of the spike
function; the bias is the subject of section 2.3 of ``docs/neuromorphic.md``.

The block is a drop-in replacement for a transformer MLP: the spiking layer
produces a firing-rate vector ``[B, S, d_ff]`` which a linear readout maps back
to ``d_model``.  Attention stays dense in this implementation.  That is a
deliberate scope choice, not an accident: making Q/K/V sparse as well is a
different architecture (Spikformer, QKFormer) with its own scaling problems.

``energy_proxy`` is the scalar requested by the assignment:

    E_proxy = n_spikes * E_MAC

It counts only spike-driven synaptic events.  The dense input projection, the
readout, membrane updates and memory traffic are *not* included; a fair
accounting adds them, and ``energy/total_proxy`` does.  See the docs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm, rope


# --------------------------------------------------------------------------- #
# Surrogate spiking function
# --------------------------------------------------------------------------- #

_SURROGATES = {"fast_sigmoid": 0, "arctan": 1}


class _SpikeFunction(torch.autograd.Function):
    """Heaviside forward, smooth surrogate backward.

    Forward:  s = 1[V >= V_th]                      (exactly binary)
    Backward: dL/dV = dL/ds * g'(V - V_th)

    with one of

        fast sigmoid:  g'(x) = alpha / (1 + alpha*|x|)^2
        arctan:        g'(x) = alpha / (1 + (alpha*x)^2)

    Both have peak slope ``alpha`` at the threshold.  The surrogate is applied
    to the *pre-spike* membrane potential, so the reset path remains part of
    the computational graph through ``-V_th * s_{t-1}``.
    """

    @staticmethod
    def forward(ctx, v: torch.Tensor, threshold: float, alpha: float,
                surrogate: int) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.threshold = float(threshold)
        ctx.alpha = float(alpha)
        ctx.surrogate = int(surrogate)
        return (v >= ctx.threshold).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v,) = ctx.saved_tensors
        x = v - ctx.threshold
        if ctx.surrogate == 0:
            sg = ctx.alpha / (1.0 + ctx.alpha * x.abs()).pow(2)
        else:
            sg = ctx.alpha / (1.0 + (ctx.alpha * x).pow(2))
        return grad_output * sg, None, None, None


def spike_fn(v: torch.Tensor, threshold: float = 1.0, alpha: float = 10.0,
             surrogate: str = "fast_sigmoid") -> torch.Tensor:
    """Binary spike with a surrogate gradient.  See :class:`_SpikeFunction`."""
    if surrogate not in _SURROGATES:
        raise ValueError(f"unknown surrogate {surrogate!r}; choose from {sorted(_SURROGATES)}")
    return _SpikeFunction.apply(v, threshold, alpha, _SURROGATES[surrogate])


# --------------------------------------------------------------------------- #
# LIF layer
# --------------------------------------------------------------------------- #

class LIFLayer(ArchModule):
    """Leaky integrate-and-fire layer over ``steps`` internal timesteps.

    Input ``x`` has shape ``[B, S, d_in]``.  The input current is
    ``I = LayerNorm(W x) * gain``, held constant across the ``T`` steps of the
    unroll (a rate-coded static stimulus).  ``beta`` is per neuron and
    parameterised as ``sigmoid(beta_logit)`` so it stays in ``(0, 1)``.

    The membrane is reset by subtraction (soft reset), exactly as in
    ``V_t = beta V_{t-1} + I - V_th s_{t-1}``.
    """

    AUX_KEYS = (
        "spike/rate", "spike/trace", "spike/sparsity",
        "state/membrane", "gate/beta", "energy_proxy",
    )

    def __init__(self, d_in: int, d_out: int, steps: int = 8, beta: float = 0.9,
                 threshold: float = 1.0, alpha: float = 10.0,
                 surrogate: str = "fast_sigmoid", gain: float = 0.3,
                 e_mac_pj: float = 0.9) -> None:
        super().__init__()
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.d_in = d_in
        self.d_out = d_out
        self.steps = steps
        self.alpha = alpha
        self.surrogate = surrogate
        self.gain = gain
        self.e_mac_pj = e_mac_pj

        self.proj = nn.Linear(d_in, d_out)
        # beta = sigmoid(beta_logit), initialised at the configured value.
        self.beta_logit = nn.Parameter(torch.full((d_out,), math.log(beta / (1.0 - beta))))
        self.register_buffer("threshold", torch.tensor(float(threshold)))

        # Diagnostics filled on the last forward pass.
        self.energy_proxy = 0.0
        self.last_sparsity = 0.0

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.shape[-1] != self.d_in:
            raise ValueError(f"expected last dim {self.d_in}, got {x.shape[-1]}")
        B, S, _ = x.shape

        # Static input current, normalised per token so its scale does not
        # depend on the layer's fan-in.  Computed once and reused every step.
        current = F.layer_norm(self.proj(x), (self.d_out,)) * self.gain
        beta = torch.sigmoid(self.beta_logit)

        v = current.new_zeros(B, S, self.d_out)
        s_prev = current.new_zeros(B, S, self.d_out)
        spikes: list[torch.Tensor] = []
        for _ in range(self.steps):
            v = beta * v + current - self.threshold * s_prev
            s = spike_fn(v, float(self.threshold), self.alpha, self.surrogate)
            spikes.append(s)
            s_prev = s

        trace = torch.stack(spikes, dim=0)                # [T, B, S, d_out]
        rate = trace.mean(dim=0)                          # [B, S, d_out]
        n_spikes = trace.sum()
        n_possible = float(trace.numel())
        sparsity = 1.0 - n_spikes / max(n_possible, 1.0)

        # E_proxy = n_spikes * E_MAC.  Detached: it is a measurement, not a
        # differentiable loss term.
        energy_proxy = (n_spikes.detach() * self.e_mac_pj)
        dense_macs = torch.tensor(
            float(B * S * self.d_in * self.d_out), device=x.device, dtype=x.dtype
        )

        self.energy_proxy = float(energy_proxy)
        self.last_sparsity = float(sparsity.detach())

        aux = {
            "spike/rate": rate,
            "spike/trace": trace,
            "spike/sparsity": sparsity.detach(),
            "spike/count": n_spikes.detach(),
            "state/membrane": v,
            "gate/beta": beta,
            "gate/beta_mean": beta.mean().detach(),
            "energy_proxy": energy_proxy,
            "energy/spikes": n_spikes.detach(),
            "energy/dense_macs": dense_macs,
        }
        return rate, aux

    def extra_repr(self) -> str:
        return (f"d_in={self.d_in}, d_out={self.d_out}, steps={self.steps}, "
                f"alpha={self.alpha}, surrogate={self.surrogate}")


class SpikingMLP(ArchModule):
    """LIF layer + linear readout, shaped like the transformer MLP."""

    AUX_KEYS = ("spike/rate", "state/membrane", "spike/sparsity", "gate/beta",
                "energy_proxy", "mlp/out")

    def __init__(self, d: int, d_ff: int, cfg: ArchConfig) -> None:
        super().__init__()
        self.lif = LIFLayer(
            d, d_ff,
            steps=cfg.get("steps", 8),
            beta=cfg.get("beta", 0.9),
            threshold=cfg.get("threshold", 1.0),
            alpha=cfg.get("alpha", 10.0),
            surrogate=cfg.get("surrogate", "fast_sigmoid"),
            gain=cfg.get("lif_gain", 0.3),
            e_mac_pj=cfg.get("e_mac_pj", 0.9),
        )
        self.out = nn.Linear(d_ff, d, bias=False)
        self.energy_proxy = 0.0

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rate, aux = self.lif(x)
        y = self.out(rate)
        B, S, _ = x.shape
        readout_macs = torch.tensor(
            float(B * S * self.lif.d_out * self.lif.d_in), device=x.device, dtype=x.dtype
        )
        aux["energy/dense_macs"] = aux["energy/dense_macs"] + readout_macs
        aux["mlp/out"] = y
        self.energy_proxy = self.lif.energy_proxy
        return y, aux


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #

class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv_heads: int, d_head: int,
                 theta: float = 10_000.0) -> None:
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
        scores = scores.masked_fill(~mask, -1e9)
        probs = scores.softmax(-1)
        y = probs @ v
        y = y.transpose(1, 2).reshape(B, S, self.n_heads * self.d_head)
        return self.w_o(y), {"attn/probs": probs, "attn/scores": scores}


class SpikingBlock(ArchModule):
    AUX_KEYS = ("attn/probs", "mlp/out", "spike/rate", "state/membrane",
                "spike/sparsity", "gate/beta", "energy_proxy")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = nn.Parameter(torch.ones(d))
        self.norm2 = nn.Parameter(torch.ones(d))
        self.attn = CausalSelfAttention(
            d, cfg["n_heads"], cfg.get("n_kv_heads", cfg["n_heads"]), cfg["d_head"]
        )
        self.mlp = SpikingMLP(d, cfg["d_ff"], cfg)
        self.energy_proxy = 0.0
        self.sparsity = 0.0

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, aux = self.attn(rms_norm(x, self.norm1))
        x = x + a
        m, maux = self.mlp(rms_norm(x, self.norm2))
        x = x + m
        aux.update(maux)
        aux["mlp/out"] = m
        self.energy_proxy = self.mlp.energy_proxy
        self.sparsity = float(maux["spike/sparsity"])
        return x, aux


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

@register("neuromorphic")
class NeuromorphicTransformer(ArchModule):
    """Decoder-only transformer whose MLP is a LIF spiking layer.

    Input: token ids ``[B, S]``.  Output: logits ``[B, S, vocab]``.
    """

    AUX_KEYS = (
        "attn/probs", "mlp/out", "spike/rate", "state/membrane",
        "spike/sparsity", "gate/beta", "energy_proxy", "lm/logits",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(SpikingBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = nn.Parameter(torch.ones(d))
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.lm_head.weight = self.embed.weight
        self.e_mac_pj = float(cfg.get("e_mac_pj", 0.9))

        self.energy_proxy = 0.0
        self.energy_total_proxy = 0.0
        self.spike_sparsity = 0.0

    def forward(self, tokens: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.embed(tokens)
        aux: dict[str, torch.Tensor] = {"embed/out": x}
        last: dict[str, torch.Tensor] = {}
        total_spikes = 0.0
        total_dense = 0.0
        total_energy = 0.0
        total_possible = 0.0
        for i, blk in enumerate(self.blocks):
            x, a = blk(x)
            last = a
            for k, v in a.items():
                aux[f"layer{i}/{k}"] = v
            total_spikes += float(a["energy/spikes"])
            total_dense += float(a["energy/dense_macs"])
            total_energy += float(a["energy_proxy"])
            total_possible += float(a["spike/trace"].numel())

        x = rms_norm(x, self.norm_f)
        logits = self.lm_head(x)
        aux["lm/logits"] = logits
        aux["hidden/final"] = x

        total_sparsity = 1.0 - total_spikes / max(total_possible, 1.0)
        total_proxy = total_energy + total_dense * self.e_mac_pj
        self.energy_proxy = total_energy
        self.energy_total_proxy = total_proxy
        self.spike_sparsity = total_sparsity

        # Aggregate scalars and re-expose the last block's canonical tensors.
        aux["energy_proxy"] = torch.tensor(total_energy, device=x.device)
        aux["energy/spikes"] = torch.tensor(total_spikes, device=x.device)
        aux["energy/dense_macs"] = torch.tensor(total_dense, device=x.device)
        aux["energy/total_proxy"] = torch.tensor(total_proxy, device=x.device)
        aux["spike/sparsity"] = torch.tensor(total_sparsity, device=x.device)
        for k in ("spike/rate", "state/membrane", "gate/beta"):
            if k in last:
                aux[k] = last[k]
        return logits, aux


def tiny_config(**over) -> ArchConfig:
    p = dict(
        d_model=64, n_heads=4, n_kv_heads=2, d_head=16,
        n_layers=2, d_ff=128, vocab=256, seq=16,
        steps=8, beta=0.9, threshold=1.0, alpha=10.0,
        surrogate="fast_sigmoid", lif_gain=0.3, e_mac_pj=0.9,
    )
    p.update(over)
    return ArchConfig("neuromorphic-tiny", "neuromorphic", p)


# The symbolic specification (contract file 2) lives in
# archmath/symbolic/neuromorphic.py.  Importing the implementation registers it
# too, so get_spec("neuromorphic") works after either import -- the same
# convenience the dense reference provides by defining its spec inline.
from ..symbolic import neuromorphic as _symbolic_neuromorphic  # noqa: E402,F401
