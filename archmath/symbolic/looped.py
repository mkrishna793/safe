"""Symbolic spec for the looped / recurrent-depth Transformer."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("looped")
def looped_spec() -> MathSpec:
    return MathSpec(
        name="looped",
        family="looped",
        summary="Looped Transformer: a core block applied n_loops times, with loop-specific LoRA, deep supervision and adaptive halting.",
        equations=[
            ("loop_iteration", r"h_{k+1}=F(h_k, x)=\mathrm{Block}(h_k)+\mathrm{LoRA}_{k}(h_k),\quad h_0=x"),
            ("unrolled", r"h_n = F^n(x) = W^n x + \sum_{i=0}^{n-1} W^i b_i"),
            ("fixed_point", r"h^* = F(h^*),\ \text{stable iff }\rho(\partial F/\partial h)<1"),
            ("contraction", r"\|h_{k+1}-h_k\| \le \rho\,\|h_k-h_{k-1}\|,\quad \rho=\rho(\partial F/\partial h)"),
            ("params", r"P_{\text{looped}}=P_{\text{core}}+L_{\text{unique}}P_{\text{layer}},\quad \text{effective depth}=n_{\text{loops}}\cdot L_{\text{core}}"),
            ("deep_supervision", r"\mathcal{L}=\sum_{k=1}^{n_{\text{loops}}}\lambda^{n_{\text{loops}}-k}\,\mathcal{L}_k,\quad \lambda\in(0,1]"),
            ("halting", r"p_{\mathrm{halt},k}=\sigma(w^\top h_k),\quad \mathbb{E}[\text{loops}]=\sum_k k\,p_{\mathrm{halt},k}\prod_{j<k}(1-p_{\mathrm{halt},j})"),
        ],
        assumptions=[
            "Shared weights make the loop an iterated map; marginal information gain decays once the state contracts to a fixed point.",
            "Loop-specific LoRA breaks exact weight sharing and can sustain diversity across loops (but costs parameters).",
            "Deep supervision stabilises training by forcing every loop to be a plausible exit point.",
        ],
        complexity={
            "params": "d_model*d_model*n_layers + n_loops*d_model*lora_rank*2",
            "serial_decode_cost": "n_loops*n_layers*d_model*d_model",
            "effective_depth": "n_loops*n_layers",
        },
        invariants=[
            "with lora_rank=0 all loops are identical maps",
            "loop delta ||h_{k+1}-h_k|| is non-increasing under contraction",
            "output shape preserved",
        ],
        references=[
            "Saunshi et al. 2025 (arXiv:2510.14404) -- looped transformers",
            "Geiping et al. 2025 (arXiv:2502.03492) -- recurrent depth",
            "Bae et al. 2025 (arXiv:2504.07819) -- looped transformers with adaptive depth",
        ],
    )
