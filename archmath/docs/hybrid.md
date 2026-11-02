# Hybrid Stack (Layer-Ratio Recipe)

## 1. The object being computed

A Transformer stack where different layers use different attention mechanisms
(softmax, linear, SSM) according to a ratio recipe.  The softmax fraction $f$
controls KV-cache and decode-FLOP savings.

## 2. Derivation

**Savings**: KV cache $\propto f$ (only softmax layers cache).  Decode FLOPs
$\propto f\,S + (1-f)\cdot\mathrm{const}$.  Hybrid wins in decode FLOPs when
$S>1$ and $f<1$.

**Recipes**:
- Solar Open 2: 1 softmax per 3 linear (gated delta rule), no positional
  encoding, 1M context.
- Nemotron-H: majority Mamba-2 layers, sparse attention, up to 3x faster.
- MiniMax: lightning attention with a softmax layer every 8.

**Crossover**: hybrid wins once $S$ is large enough that the softmax
$O(S)$ term dominates; the constant (linear) term wins for small $S$.

**Expressivity caveat**: hybrid is provably weaker than full attention on some
sequential-composition tasks (arXiv:2602.01763).  The softmax layers are
what carry the hard retrieval; linear/SSM layers carry the bulk compute.

## 3. Token-level walkthrough

Each layer applies its designated mechanism; softmax layers cache KV, linear
and SSM layers carry a constant-size state.  The ratio is fixed by the recipe.

## 4. Hidden-space consequences

- `hybrid/layer_types` shows the pattern (0=softmax,1=linear,2=ssm).
- `hybrid/ratio_actual` reports the achieved softmax fraction.
- Attention entropy only defined for softmax layers.

## 5. Complexity and failure modes

KV savings: $1-f$.  Decode-FLOP savings: $(1-f)(S-1)/S$.

Failure: too few softmax layers hurt retrieval; too many remove the savings.

## 6. References

- Solar Open 2 (arXiv:2607.20062)
- Nemotron-H (arXiv:2504.03624)
- Hybrid attention on NPUs (arXiv:2609.32114)
- Expressivity hierarchy (arXiv:2602.01763)
EOF
echo done