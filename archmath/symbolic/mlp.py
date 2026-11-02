"""Symbolic spec for Deep Dense Network (SwiGLU, point-wise token independence, NTK kernel)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("mlp")
def mlp_spec() -> MathSpec:
    return MathSpec(
        name="mlp",
        family="mlp",
        summary="Deep Residual Multi-Layer Perceptron network with SwiGLU activations, point-wise token independence, and Neural Tangent Kernel (NTK) analysis.",
        equations=[
            ("swiglu", r"\mathrm{SwiGLU}(x) = \left(\mathrm{SiLU}(xW_g) \odot xW_u\right)W_d"),
            ("residual", r"x_{l+1} = x_l + \mathrm{SwiGLU}(\mathrm{RMSNorm}(x_l))"),
            ("ntk_kernel", r"\Theta(x, x') = \sum_{w \in \theta} \frac{\partial f(x)}{\partial w} \frac{\partial f(x')}{\partial w}^\top"),
            ("ntk_dynamics", r"\frac{d f(x; \theta_t)}{dt} = -\Theta_t(x, X)\nabla_{f}\mathcal{L}(f(X), Y) \xrightarrow{d \to \infty} -\Theta_\infty(x, X)(f_t(X) - Y)"),
            ("token_independence", r"\frac{\partial x_{l, t}}{\partial x_{l-1, s}} = 0 \quad \forall t \neq s"),
            ("objective", r"\mathcal{L} = -\sum_{t=1}^{S-1} \log \mathrm{softmax}(W_v \mathrm{RMSNorm}(x_t))_{x_{t+1}}"),
        ],
        assumptions=[
            "Strict point-wise token transformation without cross-sequence mixing or attention operations.",
            "Pre-norm residual stream maintains bounded gradient norm across arbitrary network depths.",
            "In the infinite-width limit, training dynamics converge to kernel gradient descent with a constant NTK.",
            "Weight tying between embedding table and vocabulary output projection.",
        ],
        complexity={
            "params": "vocab*d_model + n_layers * (3 * d_model * d_ff + d_model) + d_model",
            "time_per_token": "n_layers * 6 * d_model * d_ff + 2 * d_model * vocab",
            "memory": "n_layers * d_model * seq",
        },
        invariants=[
            "outputs are strictly independent along sequence axis",
            "residual norm is finite for all finite inputs",
            "gradient flows directly through residual skip connection",
        ],
        references=[
            "Rumelhart, Hinton, & Williams 1986 -- Learning Representations by Back-Propagating Errors",
            "Jacot et al. 2018 (arXiv:1806.07572) -- Neural Tangent Kernel: Convergence and Generalization in Neural Networks",
            "Shazeer 2020 (arXiv:2002.05202) -- GLU Variants Improve Transformer",
        ],
    )
