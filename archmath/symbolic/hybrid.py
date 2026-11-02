"""Symbolic spec for Hybrid Attention-SSM (Pareto frontier parameter/FLOP trade-offs)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("hybrid")
def hybrid_spec() -> MathSpec:
    return MathSpec(
        name="hybrid",
        family="hybrid",
        summary="Hybrid Attention-SSM: Pareto frontier parameter/FLOP trade-offs with analytic KV-cache and decode-cost scaling.",
        equations=[
            (
                "layer_sequence",
                r"\mathbf{t}=(t_1,\dots,t_L),\quad t_l\in\{0,1,2\},\quad "
                r"0=\text{softmax},\;1=\text{linear},\;2=\text{SSM}",
            ),
            (
                "ratio",
                r"f=\frac{1}{L}\sum_{l=1}^{L}\mathbf{1}[t_l=0],\qquad "
                r"\text{solar}:f=\tfrac14,\quad\text{nemotron}:f\approx\tfrac{1}{12},\quad\text{minimax}:f=\tfrac18",
            ),
            (
                "softmax",
                r"\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M_{\text{causal}}\right)V",
            ),
            (
                "gated_delta",
                r"S_t=\alpha_t S_{t-1}+\beta_t\,(v_t-S_{t-1}k_t)k_t^\top,\quad o_t=S_t^\top q_t",
            ),
            (
                "mamba",
                r"h_t=e^{\Delta_t A}h_{t-1}+\Delta_t B_t x_t,\quad y_t=C_t^\top h_t+D x_t,\quad A=\operatorname{diag}(-\exp(a))",
            ),
            (
                "kv_cache",
                r"M_{\text{all}}(S)=L\kappa S,\quad M_{\text{hyb}}(S)=k\kappa S+(L-k)\rho,"
                r"\quad \frac{M_{\text{hyb}}}{M_{\text{all}}}\xrightarrow{S\to\infty} f",
            ),
            (
                "kv_crossover",
                r"M_{\text{hyb}}<M_{\text{all}}\iff S>\frac{\rho}{\kappa},\qquad \rho=\text{bytes of recurrent state}",
            ),
            (
                "decode_flops",
                r"F_{\text{hyb}}(S)=k(aS+b)+(L-k)c=f\,aS+\text{const},\quad a=4n_h d_h",
            ),
            (
                "pareto_tradeoff",
                r"\Delta \mathrm{FLOPs} = 4(1-f) n_h d_h S^2 - 6(1-f) n_h d_h d_s S",
            ),
        ],
        assumptions=[
            "Each non-softmax layer retains constant context-independent state; KV cache of softmax layers grows linearly with S.",
            "Decode FLOPs are counted per generated token; prefill is quadratic only in the softmax layers.",
            "The softmax layers use GQA; linear/SSM layers use recurrent matrix states.",
            "The builder clamps to at least one softmax layer for non-empty stacks.",
            "Hybrid stacks trade exact long-range recall for constant-memory recurrence along the empirical Pareto frontier.",
        ],
        complexity={
            "params": (
                "vocab*d_model + n_layers*(8*d_model*d_model + 3*d_model*d_ff"
                " + 2*d_model*d_state) + d_model"
            ),
            "time_per_token": (
                "n_softmax*(4*n_heads*d_head*seq + 8*d_model*d_model + 6*d_model*d_ff)"
                " + (n_layers-n_softmax)*(8*d_model*d_model + 6*d_model*d_ff"
                " + 6*n_heads*d_head*d_state)"
            ),
            "kv_bytes_per_token": (
                "2*n_softmax*n_kv_heads*d_head*kv_bytes"
                " + (n_layers-n_softmax)*n_heads*d_head*d_state*state_bytes/seq"
            ),
            "softmax_fraction": "n_softmax/n_layers",
            "crossover_context": "3*d_state/2",
        },
        invariants=[
            "hybrid/layer_types has length n_layers",
            "hybrid/ratio_actual equals the fraction of zero entries in layer_types",
            "every softmax layer exposes attn/probs whose rows sum to 1",
            "KV cache is linear in the number of softmax layers at long context",
            "state/final is finite for finite inputs",
        ],
        references=[
            "Park et al. 2026 (arXiv:2607.20062) -- Solar Open 2 Technical Report",
            "NVIDIA 2025 (arXiv:2504.03624) -- Nemotron-H",
            "MiniMax 2025 (arXiv:2501.08313) -- MiniMax-01: Scaling Foundation Models with Lightning Attention",
            "Ye et al. 2026 (arXiv:2602.01763) -- A Provable Expressiveness Hierarchy in Hybrid Linear-Full Attention",
            "Dao & Gu 2024 (arXiv:2405.21060) -- Transformers are SSMs: Mamba-2 (SSD)",
        ],
    )
