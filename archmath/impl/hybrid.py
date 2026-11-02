"""Production hybrid stacks: softmax, linear attention, and Mamba-2 layers.

A hybrid model is a *sequence of layer types*, not a single new primitive.
Write ``t_l in {0,1,2}`` for layer ``l``:

    0 = softmax attention   (exact recall, KV cache, O(S^2) prefill)
    1 = linear attention    (fixed recurrent state, O(S) prefill)
    2 = Mamba-2 / SSM       (fixed recurrent state, O(S) prefill)

The softmax fraction is

    f = (1/L) * sum_l 1[t_l = 0].

The three recipes below are the layer ratios used by real 2026 stacks:

    solar     1 softmax : 3 linear       (S-L-L-L, f = 1/4)
    nemotron  1 softmax : 11 Mamba-2     (~8.5% attention, f ~ 1/12)
    minimax   1 softmax : 7 lightning    (softmax every 8, f = 1/8)

The linear and SSM layers are implemented inline (no imports from other
architecture modules) so this file is self-contained.  ``analyze_ratio`` gives
an analytic account of the KV-cache and decode-FLOP savings relative to an
all-softmax stack of the same width and depth.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm, rope


# --------------------------------------------------------------------------- #
# Local primitives
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# --------------------------------------------------------------------------- #
# Layer 0: softmax attention (with optional sparse window)
# --------------------------------------------------------------------------- #

class SoftmaxAttention(nn.Module):
    """GQA softmax attention.  ``sparse`` uses a sliding window + token 0 sink."""

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        use_rope: bool = True,
        sparse: bool = False,
        window: int = 8,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.group = n_heads // n_kv_heads
        self.use_rope = use_rope
        self.sparse = sparse
        self.window = window
        self.w_q = nn.Linear(d, n_heads * d_head, bias=False)
        self.w_k = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.w_v = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.w_o = nn.Linear(n_heads * d_head, d, bias=False)
        self.gate = nn.Parameter(torch.zeros(n_heads * d_head))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, _ = x.shape
        q = self.w_q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.use_rope:
            pos = torch.arange(S, device=x.device)
            q = rope(q, pos)
            k = rope(k, pos)

        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
        if self.sparse:
            idx = torch.arange(S, device=x.device)
            window = (idx[:, None] - idx[None, :]) < self.window
            sink = idx[None, :] == 0
            allowed = causal & (window | sink)
        else:
            allowed = causal
        scores = scores.masked_fill(~allowed, -1e9)
        probs = scores.softmax(-1)
        y = (probs @ v).transpose(1, 2).reshape(B, S, self.n_heads * self.d_head)
        # Query-dependent sigmoid output gate (Solar Open 2 uses this on its
        # softmax layers); it also keeps the attention output bounded.
        y = y * torch.sigmoid(self.gate)
        return self.w_o(y), {"attn/probs": probs, "attn/mask": allowed}


# --------------------------------------------------------------------------- #
# Layer 1: tiny linear attention (gated delta rule / lightning)
# --------------------------------------------------------------------------- #

class TinyLinearAttention(nn.Module):
    """Fixed-state linear attention.

    ``variant='delta'`` is a gated delta rule:

        S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
        o_t = S_t^T q_t

    ``variant='lightning'`` is decay-only lightning attention:

        S_t = alpha_t * S_{t-1} + v_t k_t^T
        o_t = S_t^T q_t

    ``S_t`` is ``[d_head, d_head]`` per head, so decode cost is O(d_head^2)
    and state memory is independent of context length.  ``beta = 2*sigmoid(.)``
    follows Solar Open 2's negative-eigenvalue extension, allowing the state to
    erase and flip sign rather than only decay.
    """

    def __init__(self, d: int, n_heads: int, d_head: int, variant: str = "delta") -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.variant = variant
        self.w_q = nn.Linear(d, n_heads * d_head, bias=False)
        self.w_k = nn.Linear(d, n_heads * d_head, bias=False)
        self.w_v = nn.Linear(d, n_heads * d_head, bias=False)
        self.w_o = nn.Linear(n_heads * d_head, d, bias=False)
        self.log_decay = nn.Parameter(torch.full((n_heads,), 2.0))
        self.beta = nn.Linear(d, n_heads, bias=True) if variant == "delta" else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        q = F.normalize(self.w_q(x).view(B, S, H, Dh), dim=-1).transpose(1, 2)
        k = F.normalize(self.w_k(x).view(B, S, H, Dh), dim=-1).transpose(1, 2)
        v = self.w_v(x).view(B, S, H, Dh).transpose(1, 2)

        alpha = torch.sigmoid(self.log_decay)  # [H] in (0,1)
        beta = None
        if self.beta is not None:
            beta = (2.0 * torch.sigmoid(self.beta(x))).transpose(1, 2)  # [B,H,S] in (0,2)

        state = x.new_zeros(B, H, Dh, Dh)
        outs = []
        for t in range(S):
            kt = k[:, :, t]  # [B,H,Dh]
            vt = v[:, :, t]
            a = alpha.view(1, H, 1, 1)
            if beta is not None:
                b = beta[:, :, t].view(B, H, 1, 1)
                pred = torch.einsum("bhdk,bhk->bhd", state, kt)
                err = vt - pred
                state = a * state + b * err.unsqueeze(-1) * kt.unsqueeze(-2)
            else:
                state = a * state + vt.unsqueeze(-1) * kt.unsqueeze(-2)
            outs.append(torch.einsum("bhdk,bhk->bhd", state, q[:, :, t]))

        y = torch.stack(outs, dim=1).reshape(B, S, H * Dh)
        decay = alpha.view(1, H, 1).expand(B, H, S).transpose(1, 2)
        return self.w_o(y), {"state/final": state, "gate/decay": decay}


# --------------------------------------------------------------------------- #
# Layer 2: tiny Mamba-2-like selective SSM
# --------------------------------------------------------------------------- #

class TinyMamba(nn.Module):
    """Selective diagonal SSM with input-dependent dt, B, C and a SiLU gate.

        h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t
        y_t = C_t^T h_t + D x_t
        out = (y * SiLU(z)) W_o

    ``A`` is diagonal and negative (``-exp(A_log)``).  This is the Mamba-2
    state-space core without the short convolution; keeping it inline makes the
    hybrid stack independent of any external SSM implementation.
    """

    def __init__(self, d: int, n_heads: int, d_head: int, d_state: int = 16) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        inner = n_heads * d_head
        self.in_proj = nn.Linear(d, 2 * inner, bias=False)
        self.dt_proj = nn.Linear(d, n_heads, bias=True)
        self.b_proj = nn.Linear(d, d_state, bias=False)
        self.c_proj = nn.Linear(d, d_state, bias=False)
        self.out_proj = nn.Linear(inner, d, bias=False)
        self.A_log = nn.Parameter(torch.zeros(n_heads, d_head, d_state))
        self.D = nn.Parameter(torch.ones(n_heads, d_head))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, _ = x.shape
        H, Dh, N = self.n_heads, self.d_head, self.d_state
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = x_in.view(B, S, H, Dh)
        z = z.view(B, S, H, Dh)

        dt = F.softplus(self.dt_proj(x)).clamp(max=1.0)  # [B,S,H]
        Bt = self.b_proj(x)  # [B,S,N]
        Ct = self.c_proj(x)
        A = -torch.exp(self.A_log)  # [H,Dh,N]

        state = x.new_zeros(B, H, Dh, N)
        ys = []
        for t in range(S):
            d = dt[:, t].view(B, H, 1, 1)
            decay = torch.exp(d * A.view(1, H, Dh, N))
            state = decay * state + d * x_in[:, :, t].unsqueeze(-1) * Bt[:, t].view(B, 1, 1, N)
            y = (state * Ct[:, t].view(B, 1, 1, N)).sum(-1) + self.D.view(1, H, Dh) * x_in[:, :, t]
            ys.append(y)

        y = torch.stack(ys, dim=1).reshape(B, S, H * Dh)
        y = y * F.silu(z)
        return self.out_proj(y), {"state/final": state, "gate/decay": dt.transpose(1, 2)}


# --------------------------------------------------------------------------- #
# Recipe algebra
# --------------------------------------------------------------------------- #

#: Layer-type codes.  These are the values exposed in ``hybrid/layer_types``.
SOFTMAX, LINEAR, SSM = 0, 1, 2

RECIPES: dict[str, dict] = {
    "solar": {
        "softmax_fraction": 0.25,
        "mixer": LINEAR,
        "linear_variant": "delta",
        "attention": "full",
        "rope": False,
        "softmax_first": True,
        "ratio": "1 softmax : 3 linear (S-L-L-L)",
        "source": "Solar Open 2 (arXiv:2607.20062)",
        "reported": "1 softmax + 3 linear per block; ~1/4 of the memory and compute of an all-softmax stack at 1M context",
    },
    "nemotron": {
        "softmax_fraction": 1.0 / 12.0,
        "mixer": SSM,
        "linear_variant": "mamba",
        "attention": "sparse",
        "rope": False,
        "softmax_first": False,
        "ratio": "1 softmax : 11 Mamba-2 (~8.5% attention)",
        "source": "Nemotron-H (arXiv:2504.03624)",
        "reported": "4 attention layers in 52, 10 in 118 (~8%); up to 3x faster inference",
    },
    "minimax": {
        "softmax_fraction": 0.125,
        "mixer": LINEAR,
        "linear_variant": "lightning",
        "attention": "full",
        "rope": True,
        "softmax_first": False,
        "ratio": "1 softmax : 7 lightning (softmax every 8)",
        "source": "MiniMax-01 (arXiv:2501.08313)",
        "reported": "1M-token training / 4M-token inference; 20-32x longer context at comparable cost",
    },
}

#: Aliases accepted by the builder.
RECIPE_ALIASES = {
    "solar_open_2": "solar",
    "nemotron_h": "nemotron",
    "minimax_01": "minimax",
    "lightning": "minimax",
}


def _parse_ratio(ratio) -> tuple[int, int]:
    if isinstance(ratio, str):
        a, b = ratio.replace(":", " ").split()
        return int(a), int(b)
    if isinstance(ratio, int):
        return 1, ratio
    if len(ratio) != 2:
        raise ValueError("ratio must be (n_softmax, n_other), an int, or 'a:b'")
    return int(ratio[0]), int(ratio[1])


def build_layer_types(
    recipe: str = "solar",
    n_layers: int = 4,
    ratio: tuple[int, int] | str | int | None = None,
) -> list[int]:
    """Construct the layer-type sequence for a recipe.

    ``ratio=(1, 3)`` means one softmax layer per three non-softmax layers.  The
    position of the softmax layer follows the real recipe: Solar Open 2 puts it
    first in each four-layer block (S-L-L-L); MiniMax puts it last (L-L-L-L-L-L-L-S);
    Nemotron disperses its rare attention layers, with the final layer attending.

    The number of softmax layers is ``round(f * n_layers)``, clamped to at least
    one for non-empty stacks.  That last clamp is why a 4-layer diagnostic stack
    has ``f = 1/4`` even for the Nemotron recipe: a 4-layer stack cannot express
    an 8% ratio without dropping exact attention entirely.  ``ratio_actual``
    reports the realised value, never the nominal one.
    """
    if recipe in RECIPE_ALIASES:
        recipe = RECIPE_ALIASES[recipe]
    if recipe not in RECIPES:
        raise KeyError(f"unknown hybrid recipe {recipe!r}; choose from {sorted(RECIPES)}")
    if n_layers <= 0:
        return []

    spec = RECIPES[recipe]
    if ratio is None:
        f = spec["softmax_fraction"]
    else:
        s, o = _parse_ratio(ratio)
        if s <= 0 or o < 0:
            raise ValueError("ratio needs a positive softmax count and non-negative other count")
        f = s / (s + o)

    k = max(1, min(n_layers, int(round(f * n_layers))))
    if spec["softmax_first"]:
        positions = [min(n_layers - 1, int(i * n_layers / k)) for i in range(k)]
    else:
        positions = [min(n_layers - 1, int((i + 1) * n_layers / k) - 1) for i in range(k)]

    types = [spec["mixer"]] * n_layers
    for p in sorted(set(positions)):
        types[p] = SOFTMAX
    return types


def build_stack(
    recipe: str = "solar",
    n_layers: int = 4,
    ratio: tuple[int, int] | str | int | None = None,
) -> list[int]:
    """Alias for :func:`build_layer_types` (the stack builder)."""
    return build_layer_types(recipe, n_layers, ratio)


# --------------------------------------------------------------------------- #
# Blocks and model
# --------------------------------------------------------------------------- #

class HybridBlock(ArchModule):
    AUX_KEYS = ("hybrid/layer_type", "attn/probs", "state/final")

    def __init__(self, cfg: ArchConfig, layer_type: int, recipe: str) -> None:
        super().__init__()
        d = cfg["d_model"]
        spec = RECIPES[recipe]
        self.layer_type = layer_type
        self.norm1 = RMSNorm(d)
        if layer_type == SOFTMAX:
            self.mixer = SoftmaxAttention(
                d,
                cfg["n_heads"],
                cfg.get("n_kv_heads", cfg["n_heads"]),
                cfg["d_head"],
                use_rope=bool(spec["rope"]),
                sparse=spec["attention"] == "sparse",
                window=cfg.get("window", 8),
            )
        elif layer_type == LINEAR:
            self.mixer = TinyLinearAttention(
                d,
                cfg["n_heads"],
                cfg["d_head"],
                variant=spec["linear_variant"],
            )
        elif layer_type == SSM:
            self.mixer = TinyMamba(
                d,
                cfg["n_heads"],
                cfg["d_head"],
                d_state=cfg.get("d_state", 16),
            )
        else:
            raise ValueError(f"bad layer type {layer_type}")
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, cfg["d_ff"])

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y, aux = self.mixer(self.norm1(x))
        x = x + y
        m = self.mlp(self.norm2(x))
        x = x + m
        aux["hybrid/layer_type"] = torch.tensor(float(self.layer_type), device=x.device)
        aux["mlp/out"] = m
        return x, aux


@register("hybrid")
class HybridStack(ArchModule):
    AUX_KEYS = ("hybrid/layer_types", "hybrid/ratio_actual", "attn/probs")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.recipe = cfg.get("recipe", "solar")
        if self.recipe in RECIPE_ALIASES:
            self.recipe = RECIPE_ALIASES[self.recipe]
        self.layer_types = build_layer_types(
            self.recipe, cfg["n_layers"], cfg.get("ratio")
        )
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(
            HybridBlock(cfg, t, self.recipe) for t in self.layer_types
        )
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.embed(tokens)
        types = torch.tensor(self.layer_types, device=tokens.device, dtype=torch.long)
        f = float((types == SOFTMAX).float().mean()) if types.numel() else 0.0
        aux: dict[str, torch.Tensor] = {
            "embed/out": x,
            "hybrid/layer_types": types,
            "hybrid/ratio_actual": torch.tensor(f, device=tokens.device),
        }
        for i, blk in enumerate(self.blocks):
            x, a = blk(x)
            for key, value in a.items():
                aux[f"layer{i}/{key}"] = value

        h = self.norm_f(x)
        logits = self.lm_head(h)
        aux["lm/logits"] = logits
        aux["hidden/final"] = h

        analysis = analyze_ratio(self.cfg, recipe=self.recipe)
        aux["hybrid/kv_bytes_per_token"] = torch.tensor(
            analysis["kv_bytes_per_token_hybrid"], device=tokens.device
        )
        aux["hybrid/decode_flops_per_token"] = torch.tensor(
            analysis["decode_flops_hybrid"], device=tokens.device
        )
        return logits, aux


# --------------------------------------------------------------------------- #
# Analytic ratio report
# --------------------------------------------------------------------------- #

def analyze_ratio(
    cfg: ArchConfig,
    recipe: str | None = None,
    ratio: tuple[int, int] | str | int | None = None,
) -> dict[str, float]:
    """Analytic KV and decode-FLOP savings vs. an all-softmax stack.

    Let ``L`` be the number of layers, ``k`` the number of softmax layers,
    ``f = k/L``, and ``S`` the context length.  Per softmax layer a token's
    KV cache is ``kappa = 2 n_kv d_h b`` bytes (K and V, ``b`` bytes each).
    Every non-softmax layer stores a context-independent recurrent state of
    ``rho`` bytes instead.

        M_all(S)  = L kappa S
        M_hyb(S)  = k kappa S + (L-k) rho

    So the asymptotic cache ratio is

        M_hyb / M_all -> f        (KV savings -> 1 - f)

    and hybrid wins once ``S > rho / kappa``: the fixed state has to be cheaper
    than the KV that the replaced layer would have accumulated.

    For decode FLOPs per token, let ``a = 4 n_h d_h`` be the QK/AV work per key
    and let ``b`` be the constant per-layer projection + MLP cost.  A recurrent
    layer instead pays a constant state-update cost ``c``.  With averages over
    the actual layer multiset,

        F_all(S)  = L (a S + b)
        F_hyb(S)  = k (a S + b) + (L-k) c

    hence ``F_hyb = f a S + const``.  Hybrid wins when

        S > (c - b) / a.

    The returned numbers use the actual layer multiset, not just ``f``.
    """
    recipe = recipe or cfg.get("recipe", "solar")
    if recipe in RECIPE_ALIASES:
        recipe = RECIPE_ALIASES[recipe]
    if recipe not in RECIPES:
        raise KeyError(f"unknown hybrid recipe {recipe!r}")
    types = build_layer_types(recipe, int(cfg["n_layers"]), ratio)
    L = len(types)
    k = sum(1 for t in types if t == SOFTMAX)
    f = k / L if L else 0.0

    d = int(cfg["d_model"])
    H = int(cfg["n_heads"])
    Dh = int(cfg["d_head"])
    n_kv = int(cfg.get("n_kv_heads", H))
    d_ff = int(cfg["d_ff"])
    d_state = int(cfg.get("d_state", 16))
    bpe = float(cfg.get("kv_bytes", 2))
    sbpe = float(cfg.get("state_bytes", bpe))
    S = float(cfg.get("seq", 32))

    kappa = 2.0 * n_kv * Dh * bpe  # bytes per token per softmax layer
    state_per_layer: list[float] = []
    rec_cost: list[float] = []
    attn_proj = 8.0 * d * d
    mlp_cost = 6.0 * d * d_ff
    b = attn_proj + mlp_cost
    a = 4.0 * H * Dh

    for t in types:
        if t == SOFTMAX:
            continue
        if t == LINEAR:
            # Gated delta: read, erase/write, and query the [Dh,Dh] state.
            variant = RECIPES[recipe].get("linear_variant", "delta")
            update = 6.0 * H * Dh * Dh if variant == "delta" else 4.0 * H * Dh * Dh
            state_per_layer.append(H * Dh * Dh * sbpe)
            rec_cost.append(b + update)
        else:  # SSM
            update = 4.0 * H * Dh * d_state
            state_per_layer.append(H * Dh * d_state * sbpe)
            rec_cost.append(b + update)

    state_total = float(sum(state_per_layer))
    kv_all = L * kappa * S
    kv_hybrid = k * kappa * S + state_total
    kv_ratio = kv_hybrid / kv_all if kv_all > 0 else 0.0
    kv_crossover = state_total / ((L - k) * kappa) if (L - k) * kappa > 0 else 0.0

    flops_all = L * (a * S + b)
    flops_hybrid = k * (a * S + b) + float(sum(rec_cost))
    flops_ratio = flops_hybrid / flops_all if flops_all > 0 else 0.0
    c_mean = float(sum(rec_cost)) / max(len(rec_cost), 1)
    flops_crossover = max(0.0, (c_mean - b) / a) if a > 0 else 0.0

    return {
        "recipe": recipe,
        "n_layers": float(L),
        "n_softmax": float(k),
        "softmax_fraction": float(f),
        "kv_bytes_per_token_all_softmax": float(L * kappa),
        "kv_bytes_per_token_hybrid": float(kv_hybrid / S) if S > 0 else 0.0,
        "kv_ratio_hybrid_over_all": float(kv_ratio),
        "kv_savings": float(1.0 - kv_ratio),
        "kv_savings_asymptotic": float(1.0 - f),
        "kv_crossover_context": float(kv_crossover),
        "decode_flops_all_softmax": float(flops_all),
        "decode_flops_hybrid": float(flops_hybrid),
        "decode_flops_ratio": float(flops_ratio),
        "decode_flops_savings": float(1.0 - flops_ratio),
        "decode_flops_savings_asymptotic": float(1.0 - f),
        "decode_flops_crossover_context": float(flops_crossover),
        "context": float(S),
        "a_flops_per_key": float(a),
        "b_constant_softmax": float(b),
        "c_constant_recurrent_mean": float(c_mean),
        "state_bytes_total": float(state_total),
    }


# --------------------------------------------------------------------------- #
# Config + symbolic spec
# --------------------------------------------------------------------------- #

def tiny_config(**over) -> ArchConfig:
    p = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=4,
        d_head=16,
        n_layers=4,
        d_ff=128,
        vocab=256,
        seq=32,
        recipe="solar",
        d_state=16,
        window=8,
        kv_bytes=2,
        state_bytes=2,
    )
    p.update(over)
    p["n_softmax"] = sum(1 for t in build_layer_types(p["recipe"], p["n_layers"]) if t == SOFTMAX)
    return ArchConfig("hybrid-tiny", "hybrid", p)


@register_spec("hybrid")
def hybrid_spec() -> MathSpec:
    return MathSpec(
        name="hybrid",
        family="hybrid",
        summary="Builder for softmax/linear/SSM layer ratios with analytic KV-cache and decode-FLOP accounting.",
        equations=[
            (
                "layer_sequence",
                r"\mathbf{t}=(t_1,\dots,t_L),\quad t_l\in\{0,1,2\},\quad "
                r"0=\text{softmax},\;1=\text{linear},\;2=\text{SSM}",
            ),
            (
                "ratio",
                r"f=\frac{1}{L}\sum_{l=1}^{L}\mathbf{1}[t_l=0],\qquad "
                r"\text{solar}:f=\tfrac14,\quad\text{nemotron}:f\approx\tfrac{1}{12},\quad\text{minimax}:f=\tfrac18",
            ),
            (
                "softmax",
                r"\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M_{\text{causal}}\right)V",
            ),
            (
                "gated_delta",
                r"S_t=\alpha_t S_{t-1}+\beta_t\,(v_t-S_{t-1}k_t)k_t^\top,\quad "
                r"o_t=S_t^\top q_t,\quad \alpha_t,\beta_t\in(0,1)\ \text{or}\ \beta_t\in(0,2)",
            ),
            (
                "lightning",
                r"S_t=\alpha_t S_{t-1}+v_t k_t^\top,\quad o_t=S_t^\top q_t",
            ),
            (
                "mamba",
                r"h_t=e^{\Delta_t A}h_{t-1}+\Delta_t B_t x_t,\quad y_t=C_t^\top h_t+D x_t,\quad A=\operatorname{diag}(-\exp(a))",
            ),
            (
                "kv_cache",
                r"M_{\text{all}}(S)=L\kappa S,\quad M_{\text{hyb}}(S)=k\kappa S+(L-k)\rho,"
                r"\quad \frac{M_{\text{hyb}}}{M_{\text{all}}}\xrightarrow{S\to\infty} f",
            ),
            (
                "kv_crossover",
                r"M_{\text{hyb}}<M_{\text{all}}\iff S>\frac{\rho}{\kappa},\qquad \rho=\text{bytes of one recurrent state}",
            ),
            (
                "decode_flops",
                r"F_{\text{all}}(S)=L(aS+b),\quad F_{\text{hyb}}(S)=k(aS+b)+(L-k)c"
                r"=f\,aS+\text{const},\quad a=4n_h d_h",
            ),
            (
                "flops_crossover",
                r"F_{\text{hyb}}<F_{\text{all}}\iff S>\frac{c-b}{a}",
            ),
            (
                "expressivity",
                r"\text{hybrid}(L-1\text{ full}+2^{3L^2}\text{ linear})\not\supseteq\text{full-attention sequential composition}",
            ),
        ],
        assumptions=[
            "Each non-softmax layer keeps a context-independent state; the KV cache of softmax layers grows linearly with S.",
            "Decode FLOPs are counted per generated token; prefill is quadratic only in the softmax layers.",
            "The softmax layers use GQA; the linear/SSM layers use full-head state.",
            "The builder clamps to at least one softmax layer for non-empty tiny stacks; ratio_actual reports the realised fraction.",
            "Hybrid stacks trade exact long-range recall for constant-memory recurrence; the trade is provably non-trivial.",
        ],
        complexity={
            "params": (
                "vocab*d_model + n_layers*(8*d_model*d_model + 3*d_model*d_ff"
                " + 2*d_model*d_state) + d_model"
            ),
            "time_per_token": (
                "n_softmax*(4*n_heads*d_head*seq + 8*d_model*d_model + 6*d_model*d_ff)"
                " + (n_layers-n_softmax)*(8*d_model*d_model + 6*d_model*d_ff"
                " + 6*n_heads*d_head*d_state)"
            ),
            "kv_bytes_per_token": (
                "2*n_softmax*n_kv_heads*d_head*kv_bytes"
                " + (n_layers-n_softmax)*n_heads*d_head*d_state*state_bytes/seq"
            ),
            "softmax_fraction": "n_softmax/n_layers",
            "crossover_context": "3*d_state/2",
        },
        invariants=[
            "hybrid/layer_types has length n_layers",
            "hybrid/ratio_actual equals the fraction of zero entries in layer_types",
            "every softmax layer exposes attn/probs whose rows sum to 1",
            "KV cache is linear in the number of softmax layers at long context",
            "state/final is finite for finite inputs",
        ],
        references=[
            "Park et al. 2026, Solar Open 2 Technical Report (arXiv:2607.20062)",
            "NVIDIA 2025, Nemotron-H (arXiv:2504.03624)",
            "MiniMax 2025, MiniMax-01: Scaling Foundation Models with Lightning Attention (arXiv:2501.08313)",
            "Zhang et al. 2026, Empowering Hybrid Attention Models on NPUs (arXiv:2609.32114)",
            "Ye et al. 2026, A Provable Expressiveness Hierarchy in Hybrid Linear-Full Attention (arXiv:2602.01763)",
            "Yang et al. 2024, Gated Delta Networks (arXiv:2412.06464)",
            "Dao & Gu 2024, Mamba-2 (arXiv:2405.21060)",
        ],
    )
