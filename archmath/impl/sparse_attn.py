"""Native Sparse Attention / DeepSeek Sparse Attention.

A learned lightning indexer ranks past positions; the main attention then runs
exactly over the top-k ranked positions only:

    I_tj = sum_a w_{t,a} ReLU(q_{t,a}^T k_j^I)          indexer score
    S_t  = TopK_{j <= t}(I_tj)                          hard selection
    p_tj = M_tj exp(s_tj) / sum_l M_tl exp(s_tl)        exact sparse attention
    y_t  = sum_{j in S_t} p_tj v_j

The indexer is shared across the main attention heads (one scalar I_tj per
query/key pair), and uses few heads and a small dimension: its cost is
O(S H_I d_I) per query versus O(S H d_h) for dense attention.

Top-k is non-differentiable.  The forward pass uses the hard mask; the backward
pass uses a straight-through surrogate

    p = p_hard + q - sg(q),      q = softmax_{j <= t}(I_tj)

where sg is stop-gradient.  Forward is exactly p_hard, while dL/dI is the dense
softmax gradient, so the indexer receives a learning signal for every past
position (including positions that were not selected).

Input is ``tokens`` of shape [B, S] (long); output is [B, S, vocab] logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm, rope


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    """SwiGLU(x) = (x W_gate (.) SiLU(x W_up)) W_down."""

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LightningIndexer(nn.Module):
    """I_tj = sum_a w_{t,a} ReLU(q_{t,a} . k_j^I), with w = sigmoid(...)."""

    def __init__(self, d: int, n_index_heads: int, d_index: int) -> None:
        super().__init__()
        self.n_heads = n_index_heads
        self.d_index = d_index
        self.w_q = nn.Linear(d, n_index_heads * d_index, bias=False)
        self.w_k = nn.Linear(d, n_index_heads * d_index, bias=False)
        self.w_w = nn.Linear(d, n_index_heads, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape
        H, di = self.n_heads, self.d_index
        q = self.w_q(x).view(B, S, H, di).transpose(1, 2)       # [B,H,S,d_i]
        k = self.w_k(x).view(B, S, H, di).transpose(1, 2)       # [B,H,S,d_i]
        dots = q @ k.transpose(-2, -1)                          # [B,H,S,S]
        weights = torch.sigmoid(self.w_w(x)).transpose(1, 2).unsqueeze(-1)  # [B,H,S,1]
        scores = (weights * F.relu(dots)).sum(1)                # [B,S,S]
        return scores, weights.squeeze(-1).transpose(1, 2)      # weights [B,S,H]


class SparseCausalSelfAttention(nn.Module):
    """GQA + RoPE attention restricted to a learned top-k support.

    Aux shapes (B, S; H = n_heads, H_kv = n_kv_heads, d_h = d_head, k = top_k):

        attn/indexer_scores  [B, S, S]    I_tj (causal-masked, finite)
        attn/indexer_weights [B, S, H_I]  w_{t,a}
        attn/selected_mask   [B, 1, S, S] hard top-k mask (float 0/1)
        attn/scores          [B, H, S, S] exact logits, masked
        attn/probs           [B, H, S, S] p = p_hard + q - sg(q)
        attn/sparsity        scalar       selected / valid causal positions
        attn/recall_proxy    scalar       dense mass captured by the top-k set
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        top_k: int,
        n_index_heads: int,
        d_index: int,
        theta: float = 10_000.0,
        measure_recall: bool = True,
    ) -> None:
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.top_k = top_k
        self.theta = theta
        self.group = n_heads // n_kv_heads
        self.measure_recall = measure_recall
        self.w_q = nn.Linear(d, n_heads * d_head, bias=False)
        self.w_k = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.w_v = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.w_o = nn.Linear(n_heads * d_head, d, bias=False)
        self.indexer = LightningIndexer(d, n_index_heads, d_index)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, _ = x.shape
        H, dh, k = self.n_heads, self.d_head, min(self.top_k, S)
        pos = torch.arange(S, device=x.device)

        q = self.w_q(x).view(B, S, H, dh).transpose(1, 2)
        kk = self.w_k(x).view(B, S, self.n_kv_heads, dh).transpose(1, 2)
        vv = self.w_v(x).view(B, S, self.n_kv_heads, dh).transpose(1, 2)
        q = rope(q, pos, self.theta)
        kk = rope(kk, pos, self.theta)
        kk = kk.repeat_interleave(self.group, dim=1)
        vv = vv.repeat_interleave(self.group, dim=1)

        causal = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
        neg = -1e4  # finite mask value; exp(neg - max) underflows to zero

        indexer_scores, indexer_weights = self.indexer(x)       # [B,S,S], [B,S,H_I]
        indexer_masked = indexer_scores.masked_fill(~causal.unsqueeze(0), neg)

        # Hard top-k over past positions.  Rows with t < k keep all causal
        # positions: the -1e4 future entries are filtered out below.
        top_idx = indexer_masked.topk(k, dim=-1).indices           # [B,S,k]
        selected = torch.zeros(B, S, S, device=x.device, dtype=torch.bool)
        selected.scatter_(-1, top_idx, True)
        selected = selected & causal.unsqueeze(0)                 # [B,S,S]

        scores = (q @ kk.transpose(-2, -1)) / math.sqrt(dh)       # [B,H,S,S]
        attn_mask = selected.unsqueeze(1)                         # [B,1,S,S]
        masked_scores = scores.masked_fill(~attn_mask, neg)
        probs_hard = masked_scores.softmax(-1)                    # exact sparse probs

        # Straight-through: hard mask forward, dense softmax weights backward.
        indexer_logits = indexer_scores.masked_fill(~causal.unsqueeze(0), neg)
        indexer_probs = indexer_logits.softmax(-1)                # [B,S,S], sums to 1
        probs = probs_hard + indexer_probs.unsqueeze(1) - indexer_probs.detach().unsqueeze(1)

        y = (probs @ vv).transpose(1, 2).reshape(B, S, H * dh)
        y = self.w_o(y)

        valid = causal.float().sum().clamp_min(1.0)
        sparsity = selected.float().sum() / (valid * B)

        recall = torch.zeros((), device=x.device, dtype=x.dtype)
        if self.measure_recall:
            with torch.no_grad():
                dense = scores.masked_fill(~causal, neg).softmax(-1)
                recall = (dense * selected.unsqueeze(1)).sum(-1).mean()

        aux: dict[str, torch.Tensor] = {
            "attn/indexer_scores": indexer_masked,
            "attn/indexer_weights": indexer_weights,
            "attn/selected_mask": selected.unsqueeze(1).to(x.dtype),
            "attn/scores": masked_scores,
            "attn/probs": probs,
            "attn/sparsity": sparsity.to(x.dtype),
            "attn/recall_proxy": recall,
        }
        return y, aux


class SparseBlock(ArchModule):
    AUX_KEYS = (
        "attn/probs",
        "attn/indexer_scores",
        "attn/selected_mask",
        "attn/sparsity",
        "attn/recall_proxy",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = RMSNorm(d)
        self.attn = SparseCausalSelfAttention(
            d,
            cfg["n_heads"],
            cfg.get("n_kv_heads", cfg["n_heads"]),
            cfg["d_head"],
            top_k=cfg.get("top_k", 8),
            n_index_heads=cfg.get("n_index_heads", 2),
            d_index=cfg.get("d_index", 16),
            measure_recall=cfg.get("measure_recall", True),
        )
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, cfg["d_ff"])

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, aux = self.attn(self.norm1(x))
        x = x + a
        m = self.mlp(self.norm2(x))
        x = x + m
        aux["mlp/out"] = m
        return x, aux


@register("sparse_attn")
class SparseAttentionTransformer(ArchModule):
    """Pre-norm decoder-only Transformer with learned top-k sparse attention."""

    AUX_KEYS = (
        "attn/probs",
        "attn/indexer_scores",
        "attn/selected_mask",
        "attn/sparsity",
        "attn/recall_proxy",
        "lm/logits",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(SparseBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.lm_head.weight = self.embed.weight  # tied

    def forward(self, tokens: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.embed(tokens)
        aux: dict[str, torch.Tensor] = {"embed/out": x}
        for i, blk in enumerate(self.blocks):
            x, a = blk(x)
            for key, val in a.items():
                aux[f"layer{i}/{key}"] = val
        x = self.norm_f(x)
        logits = self.lm_head(x)
        aux["lm/logits"] = logits
        aux["hidden/final"] = x
        return logits, aux


def tiny_config(**over) -> ArchConfig:
    p = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        top_k=8,
        n_index_heads=2,
        d_index=16,
        n_layers=2,
        d_ff=128,
        vocab=256,
        seq=32,
    )
    p.update(over)
    return ArchConfig("sparse-attn-tiny", "sparse-attention", p)


@register_spec("sparse_attn")
def sparse_attn_spec() -> MathSpec:
    return MathSpec(
        name="sparse_attn",
        family="sparse-attention",
        summary=(
            "DeepSeek Sparse Attention / Native Sparse Attention: a low-cost "
            "lightning indexer selects top-k past positions and exact attention "
            "runs only on that support."
        ),
        equations=[
            ("indexer", r"I_{tj}=\sum_{a=1}^{H_I}w_{t,a}\,\mathrm{ReLU}(q_{t,a}^\top k_j^I),\quad q_{t,a}=W_a^Qx_t,\ k_j^I=W^Kx_j"),
            ("indexer_weights", r"w_{t,a}=\sigma(W_wx_t)_a\ge 0"),
            ("selection", r"\mathcal{S}_t=\mathrm{TopK}_{j\le t}(I_{tj}),\quad |\mathcal{S}_t|=\min(k,t+1)"),
            ("mask", r"M_{tj}=\mathbf{1}[j\in\mathcal{S}_t]"),
            ("sparse_attention", r"p_{tj}=\frac{M_{tj}\exp(s_{tj})}{\sum_{\ell\le t}M_{t\ell}\exp(s_{t\ell})},\qquad y_t=\sum_{j\in\mathcal{S}_t}p_{tj}v_j"),
            ("dense_complexity", r"\mathrm{FLOPs}_{\mathrm{dense}}=4n_hd_hS^2"),
            ("exact_sparse_complexity", r"\mathrm{FLOPs}_{\mathrm{exact}}=4n_hd_hSk+O(S\log k)"),
            ("indexer_complexity", r"\mathrm{FLOPs}_{\mathrm{indexer}}=2H_Id_IS^2"),
            ("sparsity", r"\mathrm{sparsity}=\frac{\sum_{tj}M_{tj}}{\sum_{tj}\mathbf{1}[j\le t]}\approx\frac{k}{S}"),
            ("recall", r"\mathrm{recall}=\sum_{j\in\mathcal{S}_t}p^{(\mathrm{dense})}_{tj}=1-\varepsilon_k"),
            ("missed_mass", r"\varepsilon_k=1-\sum_{i=1}^{k}p_{(i)}=\sum_{i>k}p_{(i)}"),
            ("power_law_bound", r"p_{(i)}\le Ci^{-\alpha},\ \alpha>1\ \Rightarrow\ \varepsilon_k\le\frac{C}{\alpha-1}k^{1-\alpha}"),
            ("tv_bound", r"\min_{|S|=k}\|p-p_S\|_1=2\varepsilon_k"),
            ("straight_through", r"\tilde M_{tj}=M_{tj}+q_{tj}-\mathrm{sg}(q_{tj}),\quad q_{tj}=\mathrm{softmax}_{j\le t}(I_{tj})"),
            ("expressivity", r"\mathcal{A}_k=\{P\in\Delta^{S\times S}:P_{tj}=0\ \text{if}\ j>t\ \text{or}\ j\notin\mathcal{S}_t,\ |\mathcal{S}_t|\le k\}"),
            ("exact_condition", r"p_t\ \text{representable}\iff|\mathrm{supp}(p_t)|\le k\ \text{and}\ \mathrm{supp}(p_t)\subseteq\mathcal{S}_t"),
            ("approx_condition", r"\min_{|S_t|=k}\|p_t-p_t^{(S_t)}\|_1=2\varepsilon_k"),
        ],
        assumptions=[
            "Causal: S_t is a subset of {1,...,t}; the query's own position is always eligible.",
            "The indexer is shared across main attention heads: one scalar I_tj per query/key pair.",
            "The forward mask is hard; the straight-through surrogate only changes gradients.",
            "The indexer costs O(S) per query (O(S^2) per sequence), but H_I d_I << n_h d_h; the exact attention is O(S k). Strict O(S k) total needs a sublinear indexer.",
            "If top_k >= S, the mask is all causal and the block reduces exactly to dense attention.",
            "The heavy-tail bound assumes a power-law tail p_(i) <= C i^{-alpha} with alpha > 1; alpha <= 1 has a non-summable tail and no finite bound.",
            "Sparse attention alone reduces attention FLOPs, not the KV cache; combine it with MLA for cache compression.",
            "The recall proxy is a diagnostic that reads dense attention; it is disabled with measure_recall=False.",
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
            "DeepSeek-AI 2025 (arXiv:2512.02556)",
            "Yuan et al. 2025 (arXiv:2502.11089)",
            "Baek et al. 2026 (arXiv:2608.30181)",
            "DeepSeek-AI 2026 (arXiv:2609.19969)",
        ],
    )
