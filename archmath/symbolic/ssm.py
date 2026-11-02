"""Symbolic spec for State Space Models & Mamba-2 SSD (Continuous ODE discretization, HiPPO, chunked SSD)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("ssm")
def ssm_spec() -> MathSpec:
    return MathSpec(
        name="ssm",
        family="state_space",
        summary=(
            "State Space Models & Mamba-2 SSD: Continuous ODE zero-order-hold discretization, "
            "HiPPO memory measure, scalar-identity transition, causal conv1d, and chunked semi-separable scan."
        ),
        equations=[
            ("continuous_ode", r"\dot{h}(t) = A h(t) + B x(t),\qquad y(t) = C h(t) + D x(t)"),
            ("zoh_discretization", r"\bar A = \exp(\Delta A),\qquad \bar B = (\Delta A)^{-1}(\exp(\Delta A)-I)\cdot \Delta B \approx \Delta B"),
            ("hippo_matrix", r"A_{nk} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ n+1 & \text{if } n=k \\ 0 & \text{if } n < k \end{cases}\quad (\text{HiPPO-LegS})"),
            ("conv", r"u_t = \sum_{r=0}^{3} w_r \odot x_{t-r}\quad(\text{depthwise, causal, width }4)"),
            ("delta", r"\Delta_t = \mathrm{softplus}(W_\Delta \mathrm{LN}(x)_t + b_\Delta) \in \mathbb{R}^{H}"),
            ("A_bar", r"\bar A_t = \exp(\Delta_t A),\qquad A = -\exp(A_{\log}) \in \mathbb{R}^{H}_{<0},\qquad \bar A_t \in (0,1)"),
            ("recurrence", r"h_t = \bar A_t \odot h_{t-1} + B_t\,u_t^\top,\qquad y_t = C_t^\top h_t + D\odot u_t"),
            ("scalar_identity", r"\bar A_t = a_t I_{d_s}\ \Longrightarrow\ \prod_{r=j+1}^{t}\bar A_r = \exp\Big(\sum_{r=j+1}^{t}\Delta_r A\Big) I"),
            ("chunk_decay", r"G_i = \sum_{r=1}^{i}\log \bar A_r,\qquad M_{ij} = e^{G_i-G_j}\ \ (j\le i)"),
            ("chunk_output", r"y_i = e^{G_i}\,C_i^\top h_0 + \sum_{j\le i} e^{G_i-G_j}(C_i\cdot B_j)\,u_j"),
            ("chunk_carry", r"h_C = e^{G_C} h_0 + \sum_{j=1}^{C} e^{G_C-G_j} B_j u_j^\top"),
            ("block_matrix", r"Y = \mathrm{tril}\big((C B^\top)\odot M\big)U + \big(e^{G}\mathbf{1}^\top\big)\odot(C h_0)"),
            ("output", r"y_t \leftarrow y_t \odot \mathrm{SiLU}(z_t),\qquad x_{t+1} = x_t + W_o y_t + \mathrm{MLP}(\mathrm{RMSNorm}(\cdot))"),
        ],
        assumptions=[
            "Continuous-time LTI state space model is discretized via Zero-Order Hold (ZOH) with step size Delta.",
            "HiPPO (High-order Polynomial Projection Operators) maintains optimal memory representations under continuous time measures.",
            "The transition matrix in Mamba-2 SSD is constrained to a scalar times identity per head, transforming the matrix scan into a 1D semi-separable matrix transformation.",
            "A < 0 and Delta > 0 guarantee A_bar in (0, 1), ensuring contractive, non-expansive state dynamics.",
            "Selective parameters B_t, C_t, Delta_t are input-dependent token projections.",
        ],
        complexity={
            "params": "vocab*d_model + n_layers*(3*d_model*d_head*n_heads + 2*d_model*n_heads*d_state + d_model*n_heads + d_model*d_head*n_heads + 3*d_model*d_ff + n_heads*4 + n_heads) + d_model",
            "time": "n_layers*(S*n_heads*(4*d_state*d_head + 4*d_state + 2*d_head) + S*4*d_model*d_head*n_heads + 6*S*d_model*d_ff) + 2*S*d_model*vocab",
            "time_attention_equivalent": "n_layers*n_heads*S*S*d_head",
            "state_per_token": "n_layers*n_heads*d_state*d_head",
            "conv_state_per_token": "n_layers*3*n_heads*d_head",
            "chunk_workspace": "n_layers*n_heads*chunk*chunk",
        },
        invariants=[
            "naive recurrence equals chunked scan to < 2e-5 max abs error",
            "A_bar in (0,1) for every token and head",
            "state norm is finite and bounded by ||h_0|| + sum_t ||B_t|| ||u_t||",
            "output shape equals input shape for every block",
            "token t's output depends only on tokens at or before t",
            "convolution is causal (no future leakage)",
        ],
        references=[
            "Gu et al. 2020 (arXiv:2008.07669) -- HiPPO: Recurrent Memory with Optimal Polynomial Projections",
            "Gu & Dao 2023 (arXiv:2312.00752) -- Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
            "Dao & Gu 2024 (arXiv:2405.21060) -- Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)",
        ],
    )
