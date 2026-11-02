"""Symbolic spec for Titans Test-Time Memory (Neural memory module, surprise metric gradient descent)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("titans")
def titans_spec() -> MathSpec:
    return MathSpec(
        name="titans",
        family="titans",
        summary="Titans: long-term neural memory module trained at test time via surprise-metric gradient descent, gated with short-term attention.",
        equations=[
            ("memory_update", r"M_t = (1-\alpha_t)M_{t-1} + \alpha_t\,(v_t - M_{t-1}k_t)k_t^\top"),
            ("test_time_objective", r"\ell_t(M) = \tfrac{1}{2}\|M k_t - v_t\|_2^2"),
            ("surprise_gradient", r"\nabla_M \ell_t(M) = (M k_t - v_t) k_t^\top"),
            ("surprise_metric", r"\mathcal{S}_t = \|M_{t-1}k_t - v_t\|_2^2"),
            ("momentum_memory", r"S_t = \eta_t S_{t-1} - \theta_t \nabla_M \ell_t(M_{t-1}),\quad M_t = (1-\alpha_t)M_{t-1} + S_t"),
            ("read_operation", r"r_t = M_{t-1} q_t"),
            ("gated_branch_mix", r"g_t = \sigma(W_g^\top [a_t; r_t]),\quad y_t = g_t a_t + (1-g_t) r_t"),
            ("forgetting", r"M_t \leftarrow \gamma M_t,\quad \gamma \in (0, 1)"),
        ],
        assumptions=[
            "The memory module is an in-context fast weight updated at inference via an online gradient-descent learning rule.",
            "Surprise quantifies prediction discrepancy: tokens with high surprise incur larger gradient steps.",
            "Attention provides high-precision local context, while the neural memory module maintains compressed long-horizon associations.",
            "Scalar or learned gates adaptively weigh attention and test-time memory representations.",
        ],
        complexity={
            "params": "d_model*d_model*n_layers*4 + floor(d_model*d_model/2)",
            "time_per_token": "d_model*d_model + floor(d_model*d_model/2)",
            "memory_state": "floor(d_model*d_model/2)",
        },
        invariants=[
            "memory matrix Frobenius norm remains finite for all inputs",
            "branch gating factors are strictly within [0, 1]",
            "surprise metric is non-negative",
            "memory state update is strictly causal",
        ],
        references=[
            "Behrouz et al. 2025 (arXiv:2501.00663) -- Titans: Learning to Memorize at Test Time",
            "Ba et al. 2016 (arXiv:1610.06258) -- Using Fast Weights to Attend to the Recent Past",
            "Schlag et al. 2021 (arXiv:2102.11174) -- Linear Transformers Are Secretly Fast Weight Programmers",
        ],
    )
