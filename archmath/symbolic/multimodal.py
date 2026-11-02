"""Symbolic spec for Multimodal Early Fusion (Continuous cross-modal token intertwining, manifold alignment)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("multimodal")
def multimodal_spec() -> MathSpec:
    return MathSpec(
        name="multimodal",
        family="native-multimodal",
        summary=(
            "Multimodal Early Fusion: continuous cross-modal token intertwining, "
            "manifold alignment, modality-aware attention masks, and balanced multi-task loss."
        ),
        equations=[
            (
                "token_intertwining",
                r"\mathcal{S} = [\,x^{I}_{1:P}\;\|\;x^{A}_{1:A}\;\|\;x^{T}_{1:T}\,],\qquad "
                r"h^{(0)}_i = W_{m(i)} x_i + e_{m(i)} + p_{m(i),\,i}",
            ),
            (
                "modality_mask",
                r"M_{ij}=0 \iff \big(m_i=m_j\in\{I,A\}\big)\;\lor\;\big(m_i=m_j=T\land j\le i\big)"
                r"\;\lor\;\big(m_i=T\land m_j\in\{I,A\}\big),\quad M_{ij}=-\infty\ \text{otherwise}",
            ),
            (
                "attention",
                r"\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V",
            ),
            (
                "manifold_alignment",
                r"\mathcal{D}_{\mathrm{align}} = \|\Pi_{\mathcal{M}_1}(z) - \Pi_{\mathcal{M}_2}(z)\|_2^2 \le \epsilon",
            ),
            (
                "balanced_loss",
                r"\mathcal{L}=\frac{1}{M}\sum_{m=1}^{M}\frac{1}{N_m}\sum_{t:\,m(t)=m}\ell_t"
                r"=\sum_{t} \frac{1}{M\,N_{m(t)}}\,\ell_t,\qquad \sum_t w_t=1",
            ),
            (
                "late_projector",
                r"P_m = U_m V_m^\top,\quad U_m\in\mathbb{R}^{d\times r},\;V_m\in\mathbb{R}^{E\times r}"
                r"\;\Rightarrow\;\operatorname{rank}(P_m)\le r,\quad \dim\operatorname{Im}(P_m)\le r",
            ),
            (
                "rank_capacity",
                r"I(x_m;P_m E_m(x_m))\le \frac{r}{2}\log\!\left(1+\frac{\sigma_x^2}{\sigma_\epsilon^2}\right)",
            ),
        ],
        assumptions=[
            "All modalities share a single unified transformer width d_model; modality encoders project inputs into this common space.",
            "Vision and audio tokens act as prefix modalities with bidirectional attention within each modality, but no cross-prefix attention.",
            "Text tokens attend to the entire multimodal prefix bidirectionally and to past text tokens autoregressively.",
            "Balanced loss weights each modality inversely proportional to token count, preventing dominant modal gradients.",
            "Manifold alignment projects disparate modality embeddings into a topologically continuous shared manifold.",
        ],
        complexity={
            "params": (
                "vocab*d_model + (patch_dim+audio_dim)*d_model + 6*seq*d_model + 3*d_model"
                " + n_layers*(4*d_model*d_model + d_model*n_experts + n_experts*3*d_model*d_ff)"
                " + (patch_dim+audio_dim)*d_model"
            ),
            "time_per_token": (
                "n_layers*(8*d_model*d_model + 2*n_experts*d_model*d_ff"
                " + 4*n_heads*d_head*(vision_patches+audio_frames+seq))"
                " + 2*d_model*vocab"
            ),
            "kv_bytes_per_token": "2*2*n_layers*n_kv_heads*d_head",
            "projector_params": "2*(encoder_dim*projector_rank + projector_rank*d_model + 2*projector_rank)",
            "late_capacity_ratio": "(2*projector_rank)/((vision_patches+audio_frames)*d_model)",
        },
        invariants=[
            "attention rows sum to 1 for every modality block",
            "no image or audio token attends to text; image and audio never attend to each other",
            "sequence length equals vision_patches + audio_frames + seq",
            "balanced-loss weights sum to 1",
            "each per-modality loss is finite",
            "late-fusion projector rank is at most projector_rank",
        ],
        references=[
            "An et al. 2026 (arXiv:2605.25343) -- Toward Native Multimodal Modeling: A Roadmap",
            "Shukor et al. 2025 (arXiv:2504.07951) -- Scaling Laws for Native Multimodal Models",
            "Chameleon Team 2024 (arXiv:2405.09818) -- Chameleon: Mixed-Modal Early-Fusion Foundation Models",
        ],
    )
