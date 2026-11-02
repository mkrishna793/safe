"""Symbolic spec for Multi-Head Latent Attention (DeepSeek low-rank KV compression, matrix absorption)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("mla")
def mla_spec() -> MathSpec:
    return MathSpec(
        name="mla",
        family="latent-attention",
        summary=(
            "Multi-head Latent Attention: joint low-rank KV compression with "
            "decoupled RoPE, matrix absorption for score/value projections, and Eckart-Young bounds."
        ),
        equations=[
            ("latent_query", r"c_t^Q = W^{DQ} x_t,\quad c_t^Q \in \mathbb{R}^{d_q}"),
            ("latent_kv", r"c_t^{KV} = W^{DKV} x_t,\quad c_t^{KV} \in \mathbb{R}^{d_c},\quad d_c \ll n_h d_h"),
            ("latent_norm", r"\bar c_t^{KV} = \mathrm{RMSNorm}(c_t^{KV})"),
            ("up_project", r"k_t^C = W^{UK}\bar c_t^{KV},\qquad v_t^C = W^{UV}\bar c_t^{KV}"),
            ("decoupled_rope", r"q_t^R = \mathrm{RoPE}(W^{QR} c_t^Q,t),\qquad k_t^R = \mathrm{RoPE}(W^{KR} x_t,t)"),
            ("absorption_identity", r"c_t^{Q\top} A_h \bar c_j^{KV} = (W_h^{UQ}c_t^Q)^\top(W_h^{UK}\bar c_j^{KV}),\quad A_h = W_h^{UQ\top}W_h^{UK}"),
            ("absorbed_scores", r"\tilde s_{tj} = \frac{(c_t^{Q\top}A_h)\bar c_j^{KV} + (q_t^R)^\top k_j^R}{\sqrt{d_h+d_r}}"),
            ("absorbed_output", r"y_t = \sum_h \left(\sum_{j\le t}p_{tj}\bar c_j^{KV}\right) W_h^{UV}W_h^{O}"),
            ("eckart_young", r"\min_{\operatorname{rank}(M)\le d_c}\|KV-M\|_F^2 = \sum_{i>d_c}\sigma_i^2(KV)"),
            ("cache", r"\mathrm{cache}_{MLA} = n_l\,d_c\,S\quad\text{vs.}\quad \mathrm{cache}_{dense} = n_l\,2n_hd_h\,S"),
            ("compression_ratio", r"\rho = \frac{2n_hd_h}{d_c} = \frac{\mathrm{cache}_{dense}}{\mathrm{cache}_{MLA}}"),
        ],
        assumptions=[
            "The KV cache stores a single compressed d_c latent vector per token per layer plus decoupled rotary keys, rather than per-head keys and values.",
            "RMSNorm is applied to the latent space before up-projections, enabling exact weight matrix absorption at inference.",
            "Decoupled RoPE retains positional inductive bias without destroying low-rank associativity.",
            "Eckart-Young-Mirsky theorem establishes the optimality of the low-rank SVD truncation.",
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
            "DeepSeek-AI 2024 (arXiv:2405.04434) -- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
            "DeepSeek-AI 2025 (arXiv:2512.02556) -- DeepSeek-V3 Technical Report",
            "Yuan et al. 2025 (arXiv:2502.11089) -- Hardware-Aligned Multi-Head Latent Attention",
        ],
    )
