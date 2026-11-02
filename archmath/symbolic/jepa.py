"""Symbolic spec for World Models / JEPA (Latent space joint embedding predictive architecture)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("jepa")
def jepa_spec() -> MathSpec:
    return MathSpec(
        name="jepa",
        family="jepa",
        summary=(
            "Joint Embedding Predictive Architecture (JEPA): non-generative prediction in latent space, "
            "energy-based loss with stop-gradient EMA target encoder and variance regularisation to prevent collapse."
        ),
        equations=[
            ("online_encoder", r"s_x = f_\theta(x)"),
            ("target_encoder", r"s_y = f_\xi(y),\quad \xi \leftarrow \tau\xi+(1-\tau)\theta\ \text{(EMA, stop-gradient)}"),
            ("latent_predictor", r"\hat s_y = g_\phi(s_x, a)"),
            ("energy_loss", r"\mathcal{E}(x, y, a) = \|\hat s_y - \mathrm{sg}(s_y)\|_2^2 + \lambda_v\,\mathcal{V}(s_x) + \lambda_c\,\mathcal{C}(s_x)"),
            ("variance_regulariser", r"\mathcal{V}(Z) = \frac{1}{d}\sum_{j=1}^{d}\max\big(0,\, 1 - \sqrt{\mathrm{Var}(Z_{:,j}) + \epsilon}\big)"),
            ("covariance_regulariser", r"\mathcal{C}(Z) = \frac{1}{d}\sum_{i \neq j} \mathrm{Cov}(Z)_{ij}^2"),
            ("collapse_condition", r"\nabla_\theta \mathcal{E} \neq 0 \iff \mathrm{Var}(f_\theta(x)) \ge 1 \land \mathrm{stop\_gradient}(\xi)"),
        ],
        assumptions=[
            "Predicting representations in abstract latent space bypasses pixel/token level reconstruction artifacts.",
            "Stop-gradient on the slowly-moving EMA target prevents trivial collapse to a constant state.",
            "Variance and covariance regularization ensures information capacity across orthogonal latent dimensions.",
            "Actions condition the latent world transition function g_phi(s_x, a).",
        ],
        complexity={
            "params": "d_model*d_model*n_layers*4 + d_model*d_model",
            "time_per_token": "d_model*d_model",
        },
        invariants=[
            "latent standard deviation strictly positive across dimensions (no dimensional collapse)",
            "EMA target parameters update smoothly without backpropagation gradients",
            "prediction energy loss is non-negative and finite",
        ],
        references=[
            "LeCun 2022 -- A Path Towards Autonomous Machine Intelligence (JEPA Architecture)",
            "Bardes et al. 2022 (arXiv:2105.04906) -- VICReg: Variance-Invariance-Covariance Regularization",
            "Assran et al. 2023 (arXiv:2301.08243) -- Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)",
            "Bardes et al. 2024 (arXiv:2304.07193) -- V-JEPA: Latent Video Prediction",
        ],
    )
