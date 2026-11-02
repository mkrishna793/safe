"""Symbolic specification for the sparse Mixture-of-Experts family.

The executable implementation lives in :mod:`archmath.impl.moe`; this module
states what the layer *is*, independently of any tensor library.
"""

from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("moe")
def moe_spec() -> MathSpec:
    return MathSpec(
        name="moe",
        family="sparse-moe",
        summary="Token-choice top-k sparse MoE with fine-grained routed experts, one shared expert, auxiliary-loss and aux-loss-free load balancing, and optional capacity dropping.",
        equations=[
            ("router", r"s_t = W_r\,x_t,\qquad p_t = \mathrm{softmax}(s_t)\in\Delta^{N-1}"),
            ("topk", r"\mathcal{I}_t = \mathrm{TopK}_k\!\left(s_t + b\right),\qquad |\mathcal{I}_t| = k"),
            ("gate_renorm", r"g_{i,t} = \frac{p_{i,t}}{\sum_{j\in\mathcal{I}_t}p_{j,t}}\,\mathbf{1}[i\in\mathcal{I}_t],\qquad \sum_{i=1}^{N} g_{i,t} = 1"),
            ("expert", r"\mathrm{FFN}_i(x) = \left(\mathrm{SiLU}(xW_g^i)\odot xW_u^i\right)W_d^i,\qquad W_g^i,W_u^i\in\mathbb{R}^{d\times d_{ff}},\; W_d^i\in\mathbb{R}^{d_{ff}\times d}"),
            ("shared", r"h_t = x_t + \sum_{i\in\mathcal{I}_t} g_{i,t}\,\mathrm{FFN}_i(x_t) + \sum_{j=1}^{n_s}\mathrm{FFN}^{s}_j(x_t)"),
            ("aux_loss", r"\mathcal{L}_{\mathrm{aux}} = \alpha N\sum_{i=1}^{N} f_i P_i,\quad f_i=\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[i\in\mathcal{I}_t],\quad P_i=\frac{1}{T}\sum_{t=1}^{T}p_{i,t}"),
            ("aux_covariance", r"\mathcal{L}_{\mathrm{aux}} = \alpha k + \alpha N^{2}\,\mathrm{Cov}(f,P)\ \ge\ \alpha k,\qquad \mathrm{Cov}(f,P)\ge 0\ \text{under top-k comonotonicity}"),
            ("bias_update", r"b_i \leftarrow b_i + u\,\mathrm{sign}\!\left(\bar{c}-c_i\right),\qquad \bar{c}=\frac{kT}{N},\qquad c_i=\sum_{t=1}^{T}\mathbf{1}[i\in\mathcal{I}_t]"),
            ("max_vio", r"\mathrm{MaxVio} = \frac{\max_i c_i-\bar{c}}{\bar{c}}"),
            ("capacity", r"C=\left\lceil \mathrm{cf}\cdot\frac{kT}{N}\right\rceil,\qquad \text{keep the first }C\text{ assignments per expert, drop the rest}"),
            ("objective", r"\mathcal{L} = \mathcal{L}_{\mathrm{LM}} + \sum_{l=1}^{L}\mathcal{L}^{(l)}_{\mathrm{aux}}"),
            ("all_to_all", r"\mathrm{comm} = 4\,k\,d\ \text{bytes per token per MoE layer (bf16 send + return)}"),
        ],
        assumptions=[
            "Token-choice routing: each token independently selects its own top-k experts; the router is not expert-choice.",
            "Gates are renormalised over the selected set (Mixtral/Kimi convention). DeepSeekMoE's original g_i=p_i is the unnormalised variant and changes the scale of the expert sum.",
            "The load-balance bias b is a non-trainable buffer; it shifts top-k selection but not p or g, so it carries no task gradient.",
            "Top-k indices are detached: argmax is piecewise constant, so gradient flows through p and g but not through the selection set.",
            "Capacity dropping is per expert and first-come-first-served in token order; a dropped assignment contributes zero.",
            "Shared experts are always applied to every token, independently of the router.",
            "The bias is updated once per training forward from pre-drop counts; it is not updated in eval mode.",
        ],
        complexity={
            "params": "vocab*d_model + n_layers*(2*d_model*n_heads*d_head + 2*d_model*n_kv_heads*d_head + d_model*n_experts + (n_experts+n_shared)*3*d_model*d_ff + 2*d_model) + d_model",
            "active_params": "vocab*d_model + n_layers*(2*d_model*n_heads*d_head + 2*d_model*n_kv_heads*d_head + d_model*n_experts + (top_k+n_shared)*3*d_model*d_ff + 2*d_model) + d_model",
            "time_per_token": "n_layers*(4*d_model*n_heads*d_head + 4*d_model*n_kv_heads*d_head + 4*n_heads*d_head*seq + 2*d_model*n_experts + (top_k+n_shared)*6*d_model*d_ff) + 2*d_model*vocab",
            "memory_bytes": "2*(vocab*d_model + n_layers*(2*d_model*n_heads*d_head + 2*d_model*n_kv_heads*d_head + d_model*n_experts + (n_experts+n_shared)*3*d_model*d_ff + 2*d_model) + d_model)",
            "all_to_all_bytes_per_token": "n_layers*4*top_k*d_model",
        },
        invariants=[
            "gates sum to 1 over the selected top_k experts",
            "exactly top_k routed experts fire per token before capacity dropping",
            "expert load sums to B*S (token-normalised load); raw assignment counts sum to B*S*top_k",
            "the shared expert is applied to every token regardless of the router",
            "the routing bias is a buffer, not a parameter, and receives no gradient",
            "capacity overflow can only remove assignments, never add them",
            "all aux tensors are finite for finite inputs",
        ],
        references=[
            "Shazeer et al. 2017, Outrageously Large Neural Networks (arXiv:1701.06538)",
            "Lepikhin et al. 2020, GShard (arXiv:2006.16668)",
            "Fedus et al. 2021, Switch Transformers (arXiv:2101.03961)",
            "Jiang et al. 2024, Mixtral of Experts (arXiv:2401.04088)",
            "Dai et al. 2024, DeepSeekMoE (arXiv:2401.06066)",
            "Wang et al. 2024, Auxiliary-Loss-Free Load Balancing (arXiv:2408.15664)",
            "Kimi Team 2025, Kimi K2 (arXiv:2507.20534)",
            "DeepSeek-AI 2026, DeepSeek-V4.1-Flash (arXiv:2609.19969)",
        ],
    )
