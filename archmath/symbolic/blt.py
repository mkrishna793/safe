"""Symbolic spec for Byte Latent Transformer (Byte n-gram patcher, entropy pooling, cross-attention)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("blt")
def blt_spec() -> MathSpec:
    return MathSpec(
        name="blt",
        family="blt",
        summary=(
            "Byte Latent Transformer: dynamic byte n-gram patching via entropy pooling, "
            "global latent patch cross-attention, and local light byte decoders."
        ),
        equations=[
            ("local_context", r"g_i = \mathrm{CausalConv}\big(E[x_{<i}]\big)_i,\qquad p_\phi(x_i = v \mid x_{<i}) = \mathrm{softmax}_v\big(W_e\, g_i\big)"),
            ("entropy_pooling", r"H_i = -\sum_{v=1}^{256} p_\phi(x_i = v \mid x_{<i})\,\log p_\phi(x_i = v \mid x_{<i})"),
            ("patch_threshold", r"\theta = \mu_H + \kappa\,\sigma_H \quad (\text{adaptive})\qquad \text{or}\qquad \theta = \theta_0"),
            ("segmentation", r"b_i = \mathbf{1}[i=0] \;\lor\; \Big(\mathbf{1}[i>0] \land \big(H_i > \theta \;\lor\; (i - s_i) \equiv 0 \pmod{L_{\max}}\big)\Big)"),
            ("patch_encoder", r"z_p = W_{\mathrm{enc}}\left(\frac{1}{|\mathcal{P}_p|}\sum_{i \in \mathcal{P}_p} h_i\right),\qquad h_i = \mathrm{CausalConv}(E[x_i])_i"),
            ("global_transformer", r"u_{1:P} = \mathrm{Transformer}_{\text{causal}}\big(z_{1:P}\big)"),
            ("cross_attention", r"h^{\mathrm{dec}}_i = \mathrm{CrossAttn}(Q=g_i,\, K=u_{1:P},\, V=u_{1:P})"),
            ("local_decoder", r"p_\theta(x_i \mid x_{<i}) = \mathrm{softmax}\big(W_d\,[\,u_{\pi_i - 1}\,;\, g_i\,]\big)"),
            ("compute_scaling", r"C = \mathcal{O}(P^2 d + P d^2) + \mathcal{O}(N d_{\mathrm{local}}^2 + N d_{\mathrm{local}} d),\quad P \approx N / \bar{L}"),
        ],
        assumptions=[
            "Patch segmentation is dynamically computed using next-byte prediction entropy thresholds.",
            "Local byte models are lightweight causal convolutional/MLP sub-networks with linear O(N) cost.",
            "Global Transformer runs quadratically in patch count P << N rather than raw byte sequence length N.",
            "Byte representations eliminate the tokenizer bottleneck and vocabulary out-of-vocabulary anomalies.",
        ],
        complexity={
            "params": "vocab*d_local + local_kernel*d_local + d_local*n_bytes + d_local*d_model + n_layers*(2*d_model*d_model + 2*d_model*n_kv_heads*d_head + 3*d_model*d_ff + 2*d_model) + (d_model+d_local)*d_ff + d_ff*n_bytes + d_model",
            "patches": "ceil(seq/patch_size_max)",
            "local_flops_per_byte": "2*local_kernel*d_local*d_local + 2*d_local*n_bytes + 2*d_local*d_model + 2*(d_model+d_local)*d_ff + 2*d_ff*n_bytes",
            "global_flops_per_patch": "n_layers*(2*d_model*d_model + 4*d_model*n_kv_heads*d_head + 4*d_model*ceil(seq/patch_size_max) + 6*d_model*d_ff)",
            "total_flops": "seq*(2*local_kernel*d_local*d_local + 2*d_local*n_bytes + 2*d_local*d_model + 2*(d_model+d_local)*d_ff + 2*d_ff*n_bytes) + ceil(seq/patch_size_max)*n_layers*(2*d_model*d_model + 4*d_model*n_kv_heads*d_head + 4*d_model*ceil(seq/patch_size_max) + 6*d_model*d_ff)",
            "attention_flops_per_patch": "2*n_layers*n_heads*d_head*ceil(seq/patch_size_max)",
        },
        invariants=[
            "patch boundaries are monotone: patch ids are non-decreasing in i",
            "every patch has size in [1, patch_size_max]",
            "patch_count * mean_patch_length == seq",
            "the global attention matrix is P x P, not N x N",
            "logits at byte i depend only on bytes < i",
            "byte entropy is non-negative and bounded by log(256)",
        ],
        references=[
            "Pagnoni et al. 2024 (arXiv:2412.09871) -- Byte Latent Transformer: Patches vs. Tokens",
            "Kallini et al. 2026 (arXiv:2605.08044) -- Fast Byte Latent Transformer with Sub-Quadratic Scaling",
        ],
    )
