"""Symbolic spec for RNN & LSTM (Constant Error Carousel, Jacobian spectral radius)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("rnn")
def rnn_spec() -> MathSpec:
    return MathSpec(
        name="rnn",
        family="rnn",
        summary="Deep stacked LSTM recurrent neural network with Constant Error Carousel and Jacobian spectral analysis.",
        equations=[
            ("forget_gate", r"f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)"),
            ("input_gate", r"i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)"),
            ("candidate_cell", r"\tilde{c}_t = \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c)"),
            ("cell_update", r"c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t"),
            ("output_gate", r"o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)"),
            ("hidden_state", r"h_t = o_t \odot \tanh(c_t)"),
            ("cec_jacobian", r"\frac{\partial c_t}{\partial c_{t-1}} = \mathrm{diag}(f_t)"),
            ("cec_flow", r"\frac{\partial c_T}{\partial c_t} = \prod_{k=t+1}^{T} \mathrm{diag}(f_k) = \exp\left(\sum_{k=t+1}^{T}\log f_k\right)"),
            ("spectral_radius", r"\rho\left(\frac{\partial h_t}{\partial h_{t-1}}\right) = \max_i |\lambda_i(J_t)| \le 1 \iff \text{recurrent stability}"),
            ("residual", r"x_{l+1} = x_l + h^{(l)}(x_l) + \mathrm{MLP}(\mathrm{LN}(x_l + h^{(l)}))"),
        ],
        assumptions=[
            "Strictly causal token recursion: hidden state h_t depends only on inputs x_{<=t}.",
            "Constant Error Carousel (CEC) creates an additive linear shortcut preventing exponential gradient vanishing when f_t ≈ 1.",
            "Jacobian spectral radius rho(J) <= 1 guarantees contractive stability of recurrent dynamics.",
            "Hidden state activations are bounded in (-1, 1) by the output hyperbolic tangent.",
        ],
        complexity={
            "params": "vocab*d_model + n_layers*(4*(d_model*d_model + d_model) + 4*d_model*d_model + 2*d_model*d_ff) + d_model",
            "time_per_token": "n_layers*(8*d_model*d_model + 4*d_model*d_ff)",
            "memory": "n_layers * d_model * seq",
        },
        invariants=[
            "forget, input, and output gate values are strictly in (0, 1)",
            "hidden states are bounded within (-1, 1)",
            "gradient flows across sequence via linear cell update",
            "finite hidden state norms for finite inputs",
        ],
        references=[
            "Hochreiter & Schmidhuber 1997 -- Long Short-Term Memory (Neural Computation)",
            "Gers et al. 2000 -- Learning to Forget: Continual Prediction with LSTM",
            "Pascanu et al. 2013 (arXiv:1211.5063) -- On the Difficulty of Training Recurrent Neural Networks",
        ],
    )
