# Looped / Recurrent-Depth Transformer

## 1. The object being computed

A core block applied $n$ times to the same hidden state, with loop-specific
low-rank adapters, deep supervision at every loop, and an adaptive halting
gate.  Effective depth $= n\cdot L_{\mathrm{core}}$ at parameters
$\approx L_{\mathrm{core}}$.

## 2. Derivation

**Loop as fixed-point iteration**: $h_{k+1}=F(h_k,x)$.
Unrolled: $h_n = F^n(x)$.  If $F$ is contractive ($\rho(\partial F/\partial h)<1$),
$h_n\to h^*=(I-J)^{-1}b$ and marginal information gain decays -- this is why
more loops stop helping.

**Parameter efficiency**: $P_{\mathrm{looped}}=P_{\mathrm{core}}+L_{\mathrm{unique}}P_{\mathrm{layer}}$
instead of $n\,P_{\mathrm{layer}}$.

**Loop-specific LoRA** (rank $r$): $F_k = F + A_k B_k^\top$ breaks exact
weight sharing, sustaining diversity across loops at cost $2 r d$ per loop.

**Deep supervision**: $\mathcal{L}=\sum_{k=1}^n \lambda^{n-k}\mathcal{L}_k$.
Every loop must be a plausible exit point, which stabilises training.

**Halting**: $p_{\mathrm{halt},k}=\sigma(w^\top h_k)$; expected loops
$=\sum_k k\,p_k\prod_{j<k}(1-p_j)$.

**Saturation**: repeated application contracts to a fixed point, so marginal
gain decays.  Loop-specific LoRA counteracts this by making each loop a
different map.

**Parallel Loop Transformer**: pass hidden states across adjacent token
positions between loop stages; requires *separate* normalisation of the
current-token embedding and the previous-token hidden state.

## 3. Token-level walkthrough

Token $t$ enters the loop with hidden $h_0=x_t$; each iteration refines it
using the (possibly position-passed) previous state.  The halting gate decides
whether to stop early for this token.

## 4. Hidden-space consequences

- Loop delta $\|h_{k+1}-h_k\|$ shrinks (contraction evidence).
- Loop hidden states become increasingly similar (cosine $\to 1$).
- Halting probability saturates (often to "always run all loops" early on).

## 5. Complexity and failure modes

Serial decode cost: $O(n_{\mathrm{loops}}\cdot L\cdot d^2)$.
Failure: without LoRA, loops collapse to a fixed point and extra loops add
nothing; decoding slows linearly with loops.

## 6. References

- "How to Loop Your Transformer" (arXiv:2610.40325)
- HRM (arXiv:2506.21734), TRM (arXiv:2510.04871)
- Probabilistic TRM (arXiv:2605.19943)
- Looped Transformers under the Jacobian Lens (arXiv:2609.01924)
EOF
echo done