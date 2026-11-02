# Mixture of Experts (Sparse MoE)

## 1. The object being computed

A block where each token is routed to $k$ of $N$ expert FFNs.  Compute is
$O(k)$ per token but capacity is $O(N)$: only the selected experts run, yet
all $N$ must reside in memory.

## 2. Derivation

**Router**: $\ell = x W_r \in \mathbb{R}^N$, $p=\mathrm{softmax}(\ell)$.

**Top-k selection**: $S=\mathrm{TopK}(p,k)$, gate weights
$g_i = p_i / \sum_{j\in S} p_j$ for $i\in S$.

**Output**: $y = \sum_{i\in S} g_i\, E_i(x) + E_{\mathrm{shared}}(x)$
(fine-grained experts + one always-on shared expert, DeepSeekMoE).

**Load balancing (classic)**:
$$
\mathcal{L}_{\mathrm{aux}}=\alpha\,N\sum_{i=1}^N f_i\,P_i,\quad
f_i=\frac{\#\{t: i\in S_t\}}{T},\quad P_i=\frac1T\sum_t p_{t,i}
$$
At the fixed point $f_i=P_i=1/N$ (uniform), because $\sum P_i^2$ is
minimised at uniformity by convexity (Jensen).  Gradient pushes $f_i$ down
where $P_i$ is high.

**Aux-loss-free (DeepSeek-V3)**: add a per-expert bias $b_i$ to the logits
before top-k, update $b_i \leftarrow b_i + \eta\,(\bar f - f_i)$ with no
gradient.  The bias reaches a fixed point where load is balanced, decoupling
balancing from the learning signal.

**Capacity factor**: an expert processes at most $C\cdot T/N$ tokens; overflow
tokens are dropped (or routed to the next expert).

## 3. Token-level walkthrough

For token $x_t$: compute $N$ router logits, pick $k$, renormalise their
probabilities, run $k$ small FFNs, sum with the shared expert.  The router
logit for an expert that never wins for this token still receives gradient
through softmax (all $N$ logits are differentiable).

## 4. Hidden-space consequences

- Router entropy collapses early in training if one expert dominates.
- Load histogram drifts to a few experts without balancing; with balancing it
  stays flat.
- Gate values become bimodal (near 0 or 1) as routing sharpens.
- The tracer flags `attention_collapse`-style alarms on `route/entropy`.

## 5. Complexity and failure modes

Params: $N\cdot d\cdot d_{ff}$ (all experts resident) + router.
Time per token: $k\cdot d\cdot d_{ff}$ (only $k$ experts run).
Memory: all $N$ experts must be in memory (or sharded across devices).

Failure modes: expert collapse (all tokens to one expert), load imbalance
causing stragglers in expert parallelism, capacity dropping hurting
rare tokens.

## 6. References

- Shazeer et al. 2017 (arXiv:1701.06538) -- MoE
- DeepSeekMoE (arXiv:2401.04088)
- Aux-loss-free balancing (arXiv:2408.15664)
- Kimi K2 (arXiv:2507.20534)
EOF
echo done