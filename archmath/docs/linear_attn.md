# Linear Attention / Gated DeltaNet

## 1. The object being computed

Attention whose state is a fixed-size matrix $S_t\in\mathbb{R}^{d_k\times d_v}$,
updated per token in $O(d_k d_v)$ instead of attending over the whole
history.  Gated DeltaNet adds a per-channel decay and a delta-rule correction.

## 2. Derivation

**Recurrent form**:
$$
S_t = \alpha_t \odot S_{t-1} + \beta_t\,(v_t - S_{t-1}k_t)\,k_t^\top,\quad
o_t = S_t^\top q_t
$$
where $\alpha_t=\exp(-\mathrm{softplus}(a_t))$ is a per-channel decay and
$\beta_t=\sigma(b_t)$ is a write gate.

**Delta rule as online regression**: $S_t$ is the minimiser of
$\|S k_t - v_t\|^2$ starting from $S_{t-1}$, taking one gradient step with
learning rate $1/\|k_t\|^2$:
$$
S_t = S_{t-1} - \frac1{\|k_t\|^2}(S_{t-1}k_t-v_t)k_t^\top
$$
which is exactly the formula above with $\alpha=1,\beta=1$.

**Chunkwise parallel form**: split the sequence into chunks of size $C$.
Within a chunk, compute a causal attention matrix (the "intra" term); carry
the state from the previous chunk (the "inter" term).  The two are mathematically
identical to the recurrent form; this implementation asserts equality.

**Closed form**: unrolling,
$$
S_n = \sum_{i\le n}\left(\prod_{i<j\le n}\alpha_j\right) k_i v_i^\top
$$
a weighted sum of outer products -- linear attention is a kernel smoother
with input-dependent (gated) weights.

**Negative eigenvalues** (Solar Open 2): allow $\alpha_t<0$ so the state can
flip sign, increasing expressivity without positional encodings.

## 3. Token-level walkthrough

For token $t$: decay the state channel-wise, compute the prediction error
$v_t - S_{t-1}k_t$, project it onto $k_t$, write it into $S_t$, then read
$o_t = S_t^\top q_t$.  The state is the entire "memory" of the prefix; its
norm shows how much has been stored.

## 4. Hidden-space consequences

- State norm saturates under decay; without decay it grows unbounded.
- Decay gate collapsing to 1 = "memory never forgets" (interference);
  collapsing to 0 = "memory never writes".
- Delta correction magnitude shrinks as the state learns the prefix.

## 5. Complexity and failure modes

Time: $O(S\,d\,d_{\mathrm{state}})$ vs $O(S^2 d)$.
Memory at inference: $O(d\,d_{\mathrm{state}})$ constant (no KV cache).

Failure: pure linear attention cannot solve parity-conditioned retrieval
(provably, arXiv:2605.16640); needs hybridisation with softmax attention.

## 6. References

- Gated Delta Networks (arXiv:2412.06464)
- Mamba-2 / SSD (arXiv:2405.21060)
- Solar Open 2 (arXiv:2607.20062)
- Expressivity of hybrid DeltaNet+attention (arXiv:2605.16640, arXiv:2602.01763)
EOF
echo done