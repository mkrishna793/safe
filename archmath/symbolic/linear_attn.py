"""Symbolic spec for Linear Attention & Gated DeltaNet (Katharopoulos kernel, Delta update least-squares SGD)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("linear_attn")
def linear_attn_spec() -> MathSpec:
    return MathSpec(
        name="linear_attn",
        family="linear_attention",
        summary=(
            "Linear Attention & Gated DeltaNet: Katharopoulos kernel linearization and "
            "per-channel decayed online least-squares SGD delta-rule state update with exact chunkwise scan."
        ),
        equations=[
            ("katharopoulos_kernel", r"o_t = \frac{\phi(q_t)^\top \sum_{j=1}^t \phi(k_j)v_j^\top}{\phi(q_t)^\top \sum_{j=1}^t \phi(k_j)},\quad \phi(x)=\mathrm{elu}(x)+1"),
            ("state", r"S_t \in \mathbb{R}^{d_k \times d_v},\quad o_t = S_t^\top q_t"),
            ("recurrence", r"S_t = \alpha_t \odot S_{t-1} + \beta_t\,k_t\,\big(v_t - S_{t-1}^\top k_t\big)^\top,\qquad o_t = S_t^\top q_t"),
            ("canonical", r"S_t = \alpha_t \odot S_{t-1} + \beta_t\,k_t\,\big(v_t - (\alpha_t \odot S_{t-1})^\top k_t\big)^\top"),
            ("gate_alpha", r"\alpha_t = \exp(-\mathrm{softplus}(a_t)) \in (0,1)^{d_k}"),
            ("gate_beta", r"\beta_t = \sigma(b_t) \in (0,1)"),
            ("least_squares", r"\mathcal{L}_t(S) = \tfrac12\|S^\top k_t - v_t\|_2^2,\qquad \nabla_S \mathcal{L}_t(S) = k_t (S^\top k_t - v_t)^\top"),
            ("delta_step", r"S_t = D_t S_{t-1} - \beta_t \nabla_S \mathcal{L}_t(S_{t-1}) = D_t S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top"),
            ("chunk_transform", r"T_t = \Delta_t^{-1} S_t,\quad \Delta_t = \prod_{j=1}^{t}\alpha_j"),
            ("chunk_solve", r"(I + L)D = V - X S_0,\quad L_{ij} = \beta_j\, \bar k_j^\top x_i\ (j<i),\quad D_i = v_i - T_{i-1}^\top x_i"),
            ("chunk_output", r"O = \bar Q S_0 + \mathrm{tril}(C)\,D,\quad C_{ij} = \beta_j\, \bar k_j^\top \bar q_i,\quad \bar q_i = \Delta_i \odot q_i"),
            ("chunk_carry", r"S_C = \Delta_C \odot S_0 + \sum_{j=1}^{C} \beta_j\,(\Delta_C/\Delta_j)\odot k_j\,D_j^\top"),
            ("output", r"y_t = W_o\,\mathrm{concat}_h(o_t^{(h)}) + x_t,\qquad x_{t+1} = x_t + \mathrm{MLP}(\mathrm{RMSNorm}(y_t))"),
        ],
        assumptions=[
            "Causality: the state at t depends only on (k, v, alpha, beta) at or before t.",
            "Per-channel decay is diagonal in key dimension; write gate beta is scalar per head so transition is rank-one update of diagonal.",
            "The chunked form is an exact algebraic identity, not an approximation: it tracks cumulative decay as sign * exp(sum log|alpha|).",
            "The delta rule is one online least-squares SGD step on per-token regression loss with proximal decay.",
            "Katharopoulos kernel replaces softmax denominator with associative prefix sums over feature maps phi.",
        ],
        complexity={
            "params": "vocab*d_model + n_layers*(2*d_model*n_heads*d_state + 2*d_model*n_heads*d_head + d_model*n_heads*d_state + d_model*n_heads + 3*d_model*d_ff) + d_model",
            "time": "n_layers*(S*n_heads*(4*d_state*d_head + 4*d_state + 2*d_head) + 6*S*d_model*d_ff) + 2*S*d_model*vocab",
            "time_attention_equivalent": "n_layers*n_heads*S*S*d_head",
            "state_per_token": "n_layers*n_heads*d_state*d_head",
            "chunk_workspace": "n_layers*n_heads*chunk*chunk*d_state",
        },
        invariants=[
            "recurrent form equals chunkwise form to < 2e-4 max abs error",
            "state norm is finite and (for positive alpha, beta in (0,1)) non-expansive: ||S_t|| <= ||S_{t-1}|| + ||v_t||",
            "output shape equals input shape for every block",
            "token t's output depends only on tokens at or before t",
            "alpha in (0,1) for the positive parameterisation",
            "the intra-chunk kernel is lower triangular (strictly causal)",
        ],
        references=[
            "Katharopoulos et al. 2020 (arXiv:2006.16236) -- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
            "Yang, Kautz, Hatamizadeh 2024 (arXiv:2412.06464) -- Gated Delta Networks: Improving Mamba2 with Delta Rule",
            "Yang et al. 2024 (arXiv:2406.06484) -- Parallelizing Linear Transformers with the Delta Rule over Sequence",
            "Dao & Gu 2024 (arXiv:2405.21060) -- Transformers are SSMs: Generalized Models and Efficient Algorithms (SSD)",
        ],
    )
