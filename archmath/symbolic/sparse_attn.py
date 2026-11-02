"""Symbolic spec for Native Sparse Attention (Block sparse, sliding window, BigBird/Longformer)."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("sparse_attn")
def sparse_attn_spec() -> MathSpec:
    return MathSpec(
        name="sparse_attn",
        family="sparse-attention",
        summary=(
            "Native and structured sparse attention: dynamic top-k lightning indexing, "
            "sliding window, block sparse, and BigBird/Longformer global token patterns."
        ),
        equations=[
            ("indexer", r"I_{tj}=\sum_{a=1}^{H_I}w_{t,a}\,\mathrm{ReLU}(q_{t,a}^\top k_j^I),\quad q_{t,a}=W_a^Qx_t,\ k_j^I=W^Kx_j"),
            ("indexer_weights", r"w_{t,a}=\sigma(W_wx_t)_a\ge 0"),
            ("selection", r"\mathcal{S}_t=\mathrm{TopK}_{j\le t}(I_{tj}),\quad |\mathcal{S}_t|=\min(k,t+1)"),
            ("mask", r"M_{tj}=\mathbf{1}[j\in\mathcal{S}_t]"),
            ("sparse_attention", r"p_{tj}=\frac{M_{tj}\exp(s_{tj})}{\sum_{\ell\le t}M_{t\ell}\exp(s_{t\ell})},\qquad y_t=\sum_{j\in\mathcal{S}_t}p_{tj}v_j"),
            ("sliding_window", r"M^{\mathrm{window}}_{ij} = \mathbf{1}[0 \le i - j \le w]"),
            ("block_sparse", r"M^{\mathrm{block}}_{ij} = \mathbf{1}[\lfloor i/B \rfloor = \lfloor j/B \rfloor]"),
            ("bigbird_union", r"M^{\mathrm{BigBird}}_{ij} = M^{\mathrm{window}}_{ij} \lor M^{\mathrm{block}}_{ij} \lor M^{\mathrm{random}}_{ij} \lor M^{\mathrm{global}}_{ij}"),
            ("sparsity", r"\mathrm{sparsity}=\frac{\sum_{tj}M_{tj}}{\sum_{tj}\mathbf{1}[j\le t]}\approx\frac{k}{S}"),
            ("recall", r"\mathrm{recall}=\sum_{j\in\mathcal{S}_t}p^{(\mathrm{dense})}_{tj}=1-\varepsilon_k"),
            ("power_law_bound", r"p_{(i)}\le Ci^{-\alpha},\ \alpha>1\ \Rightarrow\ \varepsilon_k\le\frac{C}{\alpha-1}k^{1-\alpha}"),
            ("tv_bound", r"\min_{|S|=k}\|p-p_S\|_1=2\varepsilon_k"),
            ("straight_through", r"\tilde M_{tj}=M_{tj}+q_{tj}-\mathrm{sg}(q_{tj}),\quad q_{tj}=\mathrm{softmax}_{j\le t}(I_{tj})"),
        ],
        assumptions=[
            "Causal: S_t is a subset of {1,...,t}; the query's own position is always eligible.",
            "The indexer is shared across main attention heads: one scalar I_tj per query/key pair.",
            "The forward mask is hard; the straight-through surrogate only changes gradients during backprop.",
            "Native sparse attention reduces exact computation from O(S^2 d) to O(S k d).",
            "BigBird / Longformer patterns combine local sliding window, dilated block strides, and selected global tokens.",
            "If top_k >= S, the mask is full causal and the block reduces exactly to dense attention.",
        ],
        complexity={
            "params": (
                "vocab*d_model + n_layers*(2*d_model*n_heads*d_head + 2*d_model*n_kv_heads*d_head "
                "+ 2*d_model*n_index_heads*d_index + d_model*n_index_heads "
                "+ 3*d_model*d_ff + 2*d_model)"
            ),
            "time_per_token": (
                "n_layers*(2*d_model*(2*n_heads*d_head + 2*n_kv_heads*d_head) "
                "+ 2*d_model*n_index_heads*d_index + d_model*n_index_heads "
                "+ 2*n_index_heads*d_index*seq + 4*n_heads*d_head*top_k "
                "+ seq*log(top_k) + 6*d_model*d_ff) + 2*d_model*vocab"
            ),
            "attention_flops_per_token": "4*n_layers*n_heads*d_head*top_k",
            "indexer_flops_per_token": "2*n_layers*n_index_heads*d_index*seq",
            "dense_attention_flops_per_token": "4*n_layers*n_heads*d_head*seq",
            "attention_flops_per_sequence": "4*n_layers*n_heads*d_head*seq*top_k",
            "dense_attention_flops_per_sequence": "4*n_layers*n_heads*d_head*seq*seq",
            "sparsity": "top_k/seq",
            "speedup_vs_dense": "seq/top_k",
            "kv_cache_elements": "n_layers*2*n_kv_heads*d_head*seq",
            "kv_cache_bytes_per_token": "n_layers*2*n_kv_heads*d_head*2",
        },
        invariants=[
            "attention rows sum to 1",
            "each row selects at most top_k positions and only positions <= t",
            "probabilities are zero outside the selected mask",
            "when top_k >= seq, sparse attention equals dense causal attention",
            "recall_proxy is in [0, 1] and sparsity is approximately top_k/seq",
            "forward attention is unchanged by the straight-through surrogate",
            "all returned aux tensors are finite for finite inputs",
        ],
        references=[
            "Beltagy et al. 2020 (arXiv:2004.05150) -- Longformer: The Long-Document Transformer",
            "Zaheer et al. 2020 (arXiv:2007.14062) -- Big Bird: Transformers for Longer Sequences",
            "DeepSeek-AI 2025 (arXiv:2512.02556) -- Native Sparse Attention Architecture",
            "Yuan et al. 2025 (arXiv:2502.11089) -- Native Sparse Attention: Hardware-Aligned and Differentiable",
            "Baek et al. 2026 (arXiv:2608.30181) -- Scalable Sparse Key-Value Routing",
        ],
    )
