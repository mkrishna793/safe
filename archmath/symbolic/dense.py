"""Symbolic spec for Dense Decoder-only Transformer (RoPE, RMSNorm, SwiGLU, GQA)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("dense")
def dense_spec() -> MathSpec:
    return MathSpec(
        name="dense",
        family="dense",
        summary="Pre-norm decoder-only Transformer with GQA, RoPE, RMSNorm and SwiGLU.",
        equations=[
            ("rmsnorm", r"\mathrm{RMSNorm}(x) = g \odot x / \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}"),
            ("qkv", r"Q = X W_Q,\quad K = X W_K,\quad V = X W_V,\quad W_K,W_V \in \mathbb{R}^{d \times h_{kv} d_h}"),
            ("rope", r"\mathrm{RoPE}(x, t)_{2i} = x_{2i}\cos(t\theta^{-2i/d_h}) - x_{2i+1}\sin(t\theta^{-2i/d_h}),\quad \theta=10^4"),
            ("gqa", r"K^{(h)} = K^{(\lfloor h / (h/h_{kv}) \rfloor)},\quad V^{(h)} = V^{(\lfloor h / (h/h_{kv}) \rfloor)}"),
            ("attention", r"\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V,\quad M_{ij} = -\infty \text{ for } j>i"),
            ("swiglu", r"\mathrm{SwiGLU}(x) = \left(\mathrm{SiLU}(xW_g) \odot xW_u\right)W_d"),
            ("residual", r"x_{l+1} = x_l + \mathrm{Attn}(\mathrm{LN}(x_l)),\quad x_{l+1} \leftarrow x_{l+1} + \mathrm{MLP}(\mathrm{LN}(x_{l+1}))"),
            ("objective", r"\mathcal{L} = -\sum_{t=1}^{S-1} \log \mathrm{softmax}(W_v \mathrm{LN}(x_t))_{x_{t+1}}"),
        ],
        assumptions=[
            "Causal mask makes the map autoregressive: y_t depends only on x_{<=t}.",
            "RoPE is relative: <RoPE(q,t), RoPE(k,s)> depends on t-s.",
            "Weight tying: W_vocab = W_embed.",
            "Grouped Query Attention (GQA) broadcasts h_kv key/value heads to h query heads.",
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
        references=[
            "Vaswani et al. 2017 (arXiv:1706.03762) -- Attention Is All You Need",
            "Shazeer 2019 (arXiv:1911.02150) -- Fast Transformer Decoding: One Write-Head is All You Need (GQA)",
            "Su et al. 2021 (arXiv:2104.09864) -- RoFormer: Enhanced Transformer with Rotary Position Embedding",
            "Zhang & Sennrich 2019 (arXiv:1910.07467) -- Root Mean Square Layer Normalization",
            "Shazeer 2020 (arXiv:2002.05202) -- GLU Variants Improve Transformer",
        ],
    )
