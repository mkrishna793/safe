"""Symbolic spec for the neuromorphic (spiking) architecture."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("neuromorphic")
def neuromorphic_spec() -> MathSpec:
    return MathSpec(
        name="neuromorphic",
        family="neuromorphic",
        summary="Spiking transformer: LIF neurons with surrogate gradients, event-driven and energy-efficient.",
        equations=[
            ("rc_circuit", r"\tau\,\frac{dV}{dt} = -V(t) + R\,I(t)"),
            ("lif_discrete", r"V_t = \beta\,V_{t-1} + W x_t - V_{\mathrm{th}}\,s_{t-1},\quad s_t = \Theta(V_t - V_{\mathrm{th}})"),
            ("surrogate", r"\frac{\partial s}{\partial V}\big|_{\mathrm{bwd}} = \frac{\alpha}{(1+\alpha|V-V_{\mathrm{th}}|)^2}\quad \text{(fast sigmoid)}"),
            ("membrane_reset", r"s_t\in\{0,1\};\quad \text{reset by subtracting } V_{\mathrm{th}}\text{ (soft reset)}"),
            ("energy", r"E \approx n_{\mathrm{spikes}}\cdot E_{\mathrm{MAC}} + E_{\mathrm{leak}}\,T"),
        ],
        assumptions=[
            "The LIF neuron is a first-order low-pass filter of the input current; it is not a universal approximator without the surrogate.",
            "Surrogate gradients introduce bias: the backward pass is not the true gradient of the Heaviside, which causes dead-neuron and gradient-bias problems.",
            "Energy advantage requires sparse spike trains; dense firing removes the benefit and adds temporal overhead.",
        ],
        complexity={
            "params": "d_model*d_model*n_layers*4",
            "time_per_token": "steps*d_model*d_model",
            "energy_proxy_pj": "e_mac_pj",
        },
        invariants=[
            "membrane potential finite for finite inputs",
            "spike rate in [0, 1]",
            "surrogate gradient finite and non-zero almost everywhere",
        ],
        references=[
            "Maass 1997 -- networks of spiking neurons",
            "Neftci et al. 2019 -- surrogate gradient learning in spiking neural networks",
        ],
    )
