"""Multi-head Latent Attention (MLA) with decoupled RoPE and weight absorption.

MLA replaces per-head K/V projections with one low-rank latent per token:

    c_t^Q  = W_DQ x_t                         query latent       [d_q]
    c_t^KV = W_DKV x_t                        KV latent          [d_c]
    k_t^C  = W_UK RMSNorm(c_t^KV)             up-projected key   [n_h d_h]
    v_t^C  = W_UV RMSNorm(c_t^KV)             up-projected value [n_h d_h]
    q_t^C  = W_UQ c_t^Q                       up-projected query [n_h d_h]
    q_t^R  = RoPE(W_QR c_t^Q, t)              decoupled query    [n_h d_r]
    k_t^R  = RoPE(W_KR x_t, t)                decoupled key      [d_r]

The KV cache stores ``c_t^KV`` (plus the small shared ``k_t^R``), never the
per-head K/V.  At inference the two up-projections are *absorbed*:

    s_tj^C = (W_UQ c_t^Q)^T (W_UK c_j^KV)
           = c_t^Q^T (W_UQ^T W_UK) c_j^KV
           = (c_t^Q^T A) c_j^KV,             A = W_UQ^T W_UK

and, on the output side,

    sum_j p_tj W_UV c_j^KV = W_UV (sum_j p_tj c_j^KV).

Thus ``W_UK`` moves into the query map and ``W_UV`` moves into ``W_O``.  The
decoupled RoPE score is *not* absorbed: a rotation does not commute with
``W_UK``, so the ``q_t^R k_j^R`` term is evaluated explicitly.

This module computes both paths.  The naive path is evaluated under
``torch.no_grad`` and compared with the absorbed path; a mismatch raises.
Set ``cfg["check_absorption"] = False`` for inference-only use.

Input is ``tokens`` of shape [B, S] (long); output is [B, S, vocab] logits.
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
    """SwiGLU(x) = (x W_gate (.) SiLU(x W_up)) W_down."""

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MultiHeadLatentAttention(nn.Module):
    """MLA attention with a latent KV cache and optional absorption check.

    Aux shapes (B, S; H = n_heads, d_h = d_head, d_c, d_r = d_rope):

        latent/c_q                [B, S, d_q]    compressed query
        latent/c_kv               [B, S, d_c]    cached KV latent
        latent/kv_norm            [B, S, d_c]    RMSNorm(c_kv), the absorbed key
        latent/compression_ratio  scalar         (2 H d_h) / d_c
        latent/absorption_error   scalar         max |y_absorbed - y_naive|
        attn/q_rope               [B, H, S, d_r] decoupled rotary query
        attn/k_rope               [B, 1, S, d_r] shared decoupled rotary key
        attn/scores               [B, H, S, S]   absorbed logits (masked)
        attn/probs                [B, H, S, S]   attention weights
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        d_head: int,
        d_c: int,
        d_rope: int,
        d_q: int | None = None,
        theta: float = 10_000.0,
        check_absorption: bool = True,
    ) -> None:
        super().__init__()
        if d_rope % 2 != 0:
            raise ValueError("d_rope must be even for the rotary half-split")
        self.d = d
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_c = d_c
        self.d_rope = d_rope
        self.d_q = d_q if d_q is not None else d_c
        self.theta = theta
        self.check_absorption = check_absorption
        self.scale = math.sqrt(d_head + d_rope)

        # Query path: low-rank query latent, then two heads of the same latent:
        # a non-rotary part (absorbable) and a decoupled rotary part.
        self.w_dq = nn.Linear(d, self.d_q, bias=False)
        self.w_uq = nn.Linear(self.d_q, n_heads * d_head, bias=False)
        self.w_qr = nn.Linear(self.d_q, n_heads * d_rope, bias=False)

        # KV path: one latent per token, one shared rotary key.
        self.w_dkv = nn.Linear(d, d_c, bias=False)
        self.kv_norm = RMSNorm(d_c)
        self.w_uk = nn.Linear(d_c, n_heads * d_head, bias=False)
        self.w_uv = nn.Linear(d_c, n_heads * d_head, bias=False)
        self.w_kr = nn.Linear(d, d_rope, bias=False)

        self.w_o = nn.Linear(n_heads * d_head, d, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, _ = x.shape
        H, dh, dc, dr = self.n_heads, self.d_head, self.d_c, self.d_rope
        pos = torch.arange(S, device=x.device)

        c_q = self.w_dq(x)
        c_kv = self.w_dkv(x)
        kv = self.kv_norm(c_kv)

        # Decoupled RoPE path: per-head query, one shared key broadcast over heads.
        q_rope = self.w_qr(c_q).view(B, S, H, dr).transpose(1, 2)
        q_rope = rope(q_rope, pos, self.theta)
        k_rope = self.w_kr(x).view(B, S, 1, dr).transpose(1, 2)
        k_rope = rope(k_rope, pos, self.theta)

        causal = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
        neg = -1e4  # finite mask value; exp(neg - max) underflows to zero

        # ---- absorbed path (the path used for the output) ------------------
        # A_h = W_UQ,h W_UK,h^T  maps c_q (d_q) to an absorbed query in d_c.
        w_uq = self.w_uq.weight.view(H, dh, self.d_q)
        w_uk = self.w_uk.weight.view(H, dh, dc)
        a = torch.einsum("heq,hec->hqc", w_uq, w_uk)              # [H, d_q, d_c]
        q_abs = torch.einsum("bsq,hqc->bhsc", c_q, a)             # [B, H, S, d_c]

        kv_h = kv.unsqueeze(1)                                    # [B, 1, S, d_c]
        scores_abs = (q_abs @ kv_h.transpose(-2, -1)
                      + q_rope @ k_rope.transpose(-2, -1)) / self.scale
        scores_abs = scores_abs.masked_fill(~causal, neg)
        probs = scores_abs.softmax(-1)
        z = probs @ kv_h                                          # [B, H, S, d_c]

        # B_h = W_O,h W_UV,h  maps the attended latent (d_c) straight to d_model.
        w_o = self.w_o.weight.view(self.d, H, dh).permute(1, 0, 2)
        w_uv = self.w_uv.weight.view(H, dh, dc)
        b = torch.einsum("hde,hec->hdc", w_o, w_uv)               # [H, d, d_c]
        y = torch.einsum("bhsc,hdc->bsd", z, b)                   # [B, S, d]

        # ---- naive path (invariant check only) ----------------------------
        err = torch.zeros((), device=x.device, dtype=x.dtype)
        if self.check_absorption:
            with torch.no_grad():
                q_nope = self.w_uq(c_q).view(B, S, H, dh).transpose(1, 2)
                k_nope = self.w_uk(kv).view(B, S, H, dh).transpose(1, 2)
                v = self.w_uv(kv).view(B, S, H, dh).transpose(1, 2)
                scores_naive = (q_nope @ k_nope.transpose(-2, -1)
                                + q_rope @ k_rope.transpose(-2, -1)) / self.scale
                scores_naive = scores_naive.masked_fill(~causal, neg)
                probs_naive = scores_naive.softmax(-1)
                y_naive = (probs_naive @ v).transpose(1, 2).reshape(B, S, H * dh)
                y_naive = self.w_o(y_naive)
                err = (y.detach() - y_naive).abs().max()
            tol = 1e-4 if x.dtype == torch.float32 else 1e-2
            if not torch.isfinite(err) or err > tol:
                raise AssertionError(
                    f"MLA absorption mismatch: max|y_abs-y_naive|={err.item():.3e} > {tol:.1e}"
                )

        aux: dict[str, torch.Tensor] = {
            "latent/c_q": c_q,
            "latent/c_kv": c_kv,
            "latent/kv_norm": kv,
            "latent/compression_ratio": torch.as_tensor(2.0 * H * dh / dc, device=x.device, dtype=x.dtype),
            "latent/absorption_error": err.detach(),
            "attn/q_rope": q_rope,
            "attn/k_rope": k_rope,
            "attn/scores": scores_abs,
            "attn/probs": probs,
        }
        return y, aux


class MLABlock(ArchModule):
    AUX_KEYS = ("latent/c_kv", "latent/kv_norm", "attn/probs", "latent/compression_ratio")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        d_c = cfg.get("d_c", max(2, cfg["n_heads"] * cfg["d_head"] // 2))
        d_rope = cfg.get("d_rope", max(2, cfg["d_head"] // 2))
        d_q = cfg.get("d_q", d_c)
        self.norm1 = RMSNorm(d)
        self.attn = MultiHeadLatentAttention(
            d,
            cfg["n_heads"],
            cfg["d_head"],
            d_c=d_c,
            d_rope=d_rope,
            d_q=d_q,
            check_absorption=cfg.get("check_absorption", True),
        )
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, cfg["d_ff"])

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, aux = self.attn(self.norm1(x))
        x = x + a
        m = self.mlp(self.norm2(x))
        x = x + m
        aux["mlp/out"] = m
        return x, aux


@register("mla")
class MLATransformer(ArchModule):
    """Pre-norm decoder-only Transformer whose attention blocks are MLA."""

    AUX_KEYS = (
        "latent/c_kv",
        "latent/kv_norm",
        "latent/compression_ratio",
        "attn/probs",
        "lm/logits",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(MLABlock(cfg) for _ in range(cfg["n_layers"]))
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
    p = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,   # accepted for interface compatibility; MLA has one latent
        d_head=16,
        d_c=32,
        d_q=32,
        d_rope=8,
        n_layers=2,
        d_ff=128,
        vocab=256,
        seq=32,
    )
    p.update(over)
    return ArchConfig("mla-tiny", "mla", p)


@register_spec("mla")
def mla_spec() -> MathSpec:
    return MathSpec(
        name="mla",
        family="latent-attention",
        summary=(
            "Multi-head Latent Attention: joint low-rank KV compression with "
            "decoupled RoPE and the absorption trick."
        ),
        equations=[
            ("latent_query", r"c_t^Q = W^{DQ} x_t,\quad c_t^Q \in \mathbb{R}^{d_q}"),
            ("latent_kv", r"c_t^{KV} = W^{DKV} x_t,\quad c_t^{KV} \in \mathbb{R}^{d_c},\quad d_c \ll n_h d_h"),
            ("latent_norm", r"\bar c_t^{KV} = \mathrm{RMSNorm}(c_t^{KV})"),
            ("up_project", r"k_t^C = W^{UK}\bar c_t^{KV},\qquad v_t^C = W^{UV}\bar c_t^{KV}"),
            ("decoupled_rope", r"q_t^R = \mathrm{RoPE}(W^{QR} c_t^Q,t),\qquad k_t^R = \mathrm{RoPE}(W^{KR} x_t,t)"),
            ("naive_scores", r"s_{tj} = \frac{(W^{UQ}c_t^Q)^\top(W^{UK}\bar c_j^{KV}) + (q_t^R)^\top k_j^R}{\sqrt{d_h+d_r}}"),
            ("absorption_identity", r"c_t^{Q\top} A_h \bar c_j^{KV} = (W_h^{UQ}c_t^Q)^\top(W_h^{UK}\bar c_j^{KV}),\quad A_h = W_h^{UQ\top}W_h^{UK}"),
            ("absorbed_scores", r"\tilde s_{tj} = \frac{(c_t^{Q\top}A_h)\bar c_j^{KV} + (q_t^R)^\top k_j^R}{\sqrt{d_h+d_r}}"),
            ("output_absorption", r"\sum_j p_{tj}W_h^{UV}\bar c_j^{KV} = W_h^{UV}\sum_j p_{tj}\bar c_j^{KV}"),
            ("absorbed_output", r"y_t = \sum_h \left(\sum_{j\le t}p_{tj}\bar c_j^{KV}\right) W_h^{UV}W_h^{O}"),
            ("eckart_young", r"\min_{\operatorname{rank}(M)\le d_c}\|KV-M\|_F^2 = \sum_{i>d_c}\sigma_i^2(KV)"),
            ("cache", r"\mathrm{cache}_{MLA} = n_l\,d_c\,S\quad\text{vs.}\quad \mathrm{cache}_{dense} = n_l\,2n_hd_h\,S"),
            ("compression_ratio", r"\rho = \frac{2n_hd_h}{d_c} = \frac{\mathrm{cache}_{dense}}{\mathrm{cache}_{MLA}}"),
        ],
        assumptions=[
            "The cache stores one d_c-vector c_KV per token per layer (plus a small shared d_r rotary key), not per-head K/V.",
            "RMSNorm acts on the latent before W_UK/W_UV; both paths use the same normalized latent, so the normalization is absorbable.",
            "If the per-head stacked [K;V] matrix has rank at most d_c, the latent bottleneck is exact. Otherwise Eckart-Young says the optimal rank-d_c approximation has squared Frobenius error equal to the discarded singular values.",
            "RoPE is not absorbed: R_t W_UK != W_UK R_t in general. The decoupled rotary score is computed explicitly.",
            "The absorption identity is exact in real arithmetic; the implementation asserts a floating-point tolerance.",
            "n_kv_heads is accepted for config compatibility, but MLA has a single latent per token, so it does not multiply the cache.",
        ],
        complexity={
            "params": (
                "vocab*d_model + n_layers*(d_model*d_q + d_q*n_heads*d_head + d_q*n_heads*d_rope "
                "+ d_model*d_c + d_c*n_heads*d_head + d_c*n_heads*d_head + d_model*d_rope "
                "+ n_heads*d_head*d_model + 3*d_model*d_ff + 2*d_model + d_c)"
            ),
            "time_per_token": (
                "n_layers*(2*d_model*d_q + 2*d_q*n_heads*(d_c + d_rope) "
                "+ 2*d_model*(d_c + d_rope) + 2*n_heads*(2*d_c + d_rope)*seq "
                "+ 2*n_heads*d_model*d_c + 6*d_model*d_ff) + 2*d_model*vocab"
            ),
            "absorbed_attention_flops_per_token": "2*n_layers*n_heads*(2*d_c + d_rope)*seq",
            "dense_attention_flops_per_token": "4*n_layers*n_heads*d_head*seq",
            "kv_cache_elements": "n_layers*d_c*seq",
            "kv_cache_total_elements": "n_layers*(d_c + d_rope)*seq",
            "dense_kv_cache_elements": "n_layers*2*n_heads*d_head*seq",
            "kv_cache_latent_bytes_per_token": "n_layers*d_c*2",
            "compression_ratio": "2*n_heads*d_head/d_c",
        },
        invariants=[
            "naive and absorbed attention outputs agree to floating-point tolerance",
            "attention rows sum to 1",
            "the KV cache stores one d_c-vector per token per layer, independent of n_heads",
            "latent rank is at most d_c",
            "logits at position t depend only on tokens at or before t",
            "all returned aux tensors are finite for finite inputs",
        ],
        references=[
            "DeepSeek-AI 2024 (arXiv:2405.04434)",
            "DeepSeek-AI 2025 (arXiv:2512.02556)",
            "DeepSeek-AI 2026 (arXiv:2609.19969)",
            "Yuan et al. 2025 (arXiv:2502.11089)",
        ],
    )
