# Sparse Attention (Native / DeepSeek Sparse)

## 1. The object being computed

Exact attention over a *selected* subset of positions, chosen per query by a
cheap learned indexer.  Quadratic $\to$ linear in $S$ for the heavy branch,
with a recall trade-off governed by how heavy-tailed the attention is.

## 2. Derivation

**Indexer** (lightning indexer): for query $t$,
$$
I_{t,j} = \sum_{r=1}^{H_I} w_r\,\mathrm{ReLU}(q_{t,r}\cdot k_{j,r})
$$
with few heads $H_I$ and low dimension.  Cost $O(S\cdot d_I)$.

**Selection**: keep top-$k$ positions $S_t=\mathrm{TopK}_j(I_{t,j},k)$.

**Sparse attention**: $\mathrm{Attn}_t = \mathrm{softmax}_{j\in S_t}(q_t\cdot k_j/\sqrt{d})\,v_j$.

**Recall trade-off**: probability mass missed
$=1-\sum_{j\in S_t} p_{t,j}$.  If the attention distribution is heavy-tailed
(power law), the top-$k$ captures most of the mass and recall is high; for
uniform attention, recall $\approx k/S$.

**Straight-through top-k**: forward uses a hard mask; backward flows through
the soft indexer scores, so the selector trains.

## 3. Token-level walkthrough

For query $t$: indexer scores all $S$ past positions in $O(S d_I)$, pick $k$,
run exact attention over those $k$.  The selected set changes per query and
per head, so the sparsity pattern is input-dependent.

## 4. Hidden-space consequences

- `attn/sparsity` = $k/S$ (constant by construction) but the *actual* number
  of distinct positions used varies.
- `attn/recall_proxy` = sum of top-k attention weights under dense attention.
- Indexer scores should correlate with true attention; the tracer can measure
  this correlation.

## 5. Complexity and failure modes

Time: $O(S\cdot d_I + S\cdot k\cdot d)$ vs $O(S^2 d)$.
Failure: indexer can miss a sharp attention spike (low recall) if its
features are too weak; training instability from the straight-through estimator.

## 6. References

- Native Sparse Attention (arXiv:2502.11089)
- DeepSeek-V3.2 / DSA (arXiv:2512.02556)
- DeepSeek-V4.1-Flash (arXiv:2609.19969)
- A.X K2 (arXiv:2608.30181)
EOF
echo done