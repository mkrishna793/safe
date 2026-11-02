"""Symbolic spec for Diffusion LMs / DiT (Discrete masked diffusion, probability flow ODE, score matching)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("diffusion")
def diffusion_spec() -> MathSpec:
    return MathSpec(
        name="diffusion",
        family="diffusion",
        summary="Masked discrete diffusion LM / DiT: absorbing-state forward process, score matching, probability flow ODE, and bidirectional ELBO.",
        equations=[
            ("forward_process", r"q_t(x_t \mid x_0) = \prod_{i=1}^{S} \big[\, t\,\delta(x_t^i, x_0^i) + (1-t)\,\delta(x_t^i, \mathrm{M}) \,\big],\quad t \in (0,1]"),
            ("score_matching", r"\nabla_{x_t}\log q_t(x_t) = -\mathbb{E}[x_0 - x_t \mid x_t] / \sigma_t^2"),
            ("probability_flow_ode", r"dx = \left[f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)\right] dt"),
            ("mask_marginal", r"q_t(x_t^i = \mathrm{M} \mid x_0) = 1-t, \qquad q_t(x_t^i = x_0^i \mid x_0) = t"),
            ("reverse_model", r"p_\theta(x_0 \mid x_t) = \prod_{i=1}^{S} p_\theta(x_0^i \mid x_t),\qquad p_\theta(x_0^i = v \mid x_t) = \mathrm{softmax}_v\!\big(W h_\theta(x_t)^i\big)"),
            ("bidirectional_attention", r"\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_h})}{\sum_{j'=1}^{S}\exp(q_i \cdot k_{j'} / \sqrt{d_h})}\ \ \forall j \quad (\text{no causal mask})"),
            ("elbo", r"\mathcal{L}(\theta) = \mathbb{E}_{t\sim\mathcal{U}[0,1]}\ \mathbb{E}_{x_0}\ \mathbb{E}_{x_t\sim q_t(\cdot\mid x_0)} \left[ \frac{1}{1-t} \sum_{i:\,x_t^i=\mathrm{M}} -\log p_\theta(x_0^i \mid x_t) \right]"),
            ("confidence_sampling", r"c_i = \max_{v}\, p_\theta(x_i = v \mid x_t),\qquad \mathcal{U}_k = \operatorname{TopK}_{i\,:\,x_t^i=\mathrm{M}}\,(c_i)"),
        ],
        assumptions=[
            "The forward process is absorbing: a masked token never reverts, yielding exact analytic marginals q_t(x_t|x_0).",
            "The reverse model factorizes conditionally over positions given x_t, enabling parallel iterative decoding.",
            "The denoiser backbone employs bidirectional non-causal attention across all positions.",
            "Probability flow ODE and score matching connect discrete jump processes to continuous diffusion formulations.",
            "1/(1-t) loss weighting cancels out the marginal probability of masking, providing an unbiased per-token NLL estimate.",
        ],
        complexity={
            "params": "(vocab+1)*d_model + n_layers*(2*d_model*d_model + 2*d_model*n_kv_heads*d_head + 3*d_model*d_ff + 2*d_model) + d_model",
            "time_per_token": "n_layers*(2*d_model*d_model + 4*d_model*n_kv_heads*d_head + 4*d_model*seq + 6*d_model*d_ff) + 2*d_model*vocab",
            "attention_flops_per_token": "2*n_layers*n_heads*d_head*seq",
            "train_flops_per_sequence": "seq*(n_layers*(2*d_model*d_model + 4*d_model*n_kv_heads*d_head + 4*d_model*seq + 6*d_model*d_ff) + 2*d_model*vocab)",
            "sampling_forwards": "diffusion_steps",
            "sampling_flops_per_sequence": "diffusion_steps*seq*(n_layers*(2*d_model*d_model + 4*d_model*n_kv_heads*d_head + 4*d_model*seq + 6*d_model*d_ff) + 2*d_model*vocab)",
        },
        invariants=[
            "attention rows sum to 1 and the mask is all-ones (bidirectional)",
            "0 <= t <= 1 and the ELBO weight is 1/(1-t) > 0",
            "E[1/(1-t) * 1[masked]] = 1 for every position, so the loss is an unbiased per-token NLL estimate",
            "the mask token id (vocab) is outside the predicted vocabulary",
            "at initialization the per-token ELBO is approximately log(vocab)",
            "sampling unmasks every position after exactly `diffusion_steps` steps",
            "final sampled tokens are always in [0, vocab)",
        ],
        references=[
            "Song et al. 2020 (arXiv:2011.13456) -- Score-Based Generative Modeling through Stochastic Differential Equations",
            "Peebles & Xie 2023 (arXiv:2212.09748) -- Scalable Diffusion Models with Transformers (DiT)",
            "Nie et al. 2025 (arXiv:2502.09992) -- Large Language Diffusion Models (LLaDA)",
            "Khanna et al. 2025 (arXiv:2506.17298) -- Mercury: Ultra-Fast Diffusion Language Models",
        ],
    )
