# Dense decoder-only Transformer

## 1. The object being computed

A causal language model: given tokens $x_1,\dots,x_S$, produce logits
$\ell_t \in \mathbb{R}^V$ for $x_{t+1}$.  The whole model is a stack of
identical residual blocks over a $d$-dimensional residual stream, with
weight-tied input/output embeddings.

## 2. Derivation

**RMSNorm** (Zhang & Sennrich 2019):
$$
\mathrm{RMSNorm}(x)=g\odot \frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}
$$
Scale invariant for $\epsilon=0$: $\mathrm{RMSNorm}(c x)=\mathrm{RMSNorm}(x)$
for $c>0$.  The tiny $\epsilon$ breaks exact invariance, hence $\epsilon=10^{-6}$.

**RoPE** (Su et al. 2021): rotate pairs of dimensions by position-dependent angles
so that $\langle \mathrm{RoPE}(q,t),\mathrm{RoPE}(k,s)\rangle$ depends only on
$t-s$.  This is the property that makes attention *relative* without a learned
position bias.

**GQA** (Shazeer 2019): $h$ query heads, $h_{kv}<h$ key/value heads, each KV
head is broadcast to $h/h_{kv}$ query heads.  Parameter count
$4dh$ (MHA) $\to d h + 2 d h_{kv} + d h$ (GQA); KV cache $\to h_{kv}/h$.

**Attention**:
$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V,\quad
M_{ij}=-\infty\ (j>i)
$$

**SwiGLU** (Shazeer 2020): $\mathrm{SwiGLU}(x)=(\mathrm{SiLU}(xW_g)\odot xW_u)W_d$,
with expansion $8/3\,d$.

**Residual stream** (pre-norm):
$$
x_{l+1}=x_l+\mathrm{Attn}(\mathrm{LN}(x_l)),\quad
x_{l+1}\leftarrow x_{l+1}+\mathrm{MLP}(\mathrm{LN}(x_{l+1}))
$$

## 3. Token-level walkthrough

For a single new token $x_t$: embed $\to$ residual stream $\to$ for each layer:
LN normalises the stream, attention mixes information from $x_{\le t}$ (causal
mask), the residual adds it back, LN again, MLP transforms, residual adds back.
Finally LN + tied unembedding gives logits over $V$.  Every operation is
local in the sequence dimension except attention, which is the only place tokens
talk to each other.

## 4. Hidden-space consequences (what the tracer should see)

- Residual-stream norm grows with depth (pre-norm stacks add, not replace).
- Effective rank of the stream is high early, may collapse in deep layers.
- Attention entropy: sharp (low) for induction heads, diffuse (high) for
  aggregation heads.  A head that is always diffuse is a candidate for pruning.
- Attention scores contain $-\infty$ on the masked (future) positions; the
  tracer masks these to a large finite number so telemetry stays finite.

## 5. Complexity and failure modes

Params: $V d + L(4 d h d_h + 3 d d_{ff}) + d$.
Time per token: $O(L(d^2 + d d_{ff} + h d_h S))$.
KV cache: $O(L\,h_{kv}\,d_h\,S)$.

Failure modes: attention sink (first token gets most mass) in long contexts;
rank collapse in very deep stacks; gradient vanishing if residual paths are
not well-conditioned.

## 6. References

- Vaswani et al. 2017 (arXiv:1706.03762)
- Shazeer 2019 (GQA), Su et al. 2021 (RoPE), Zhang & Sennrich 2019 (RMSNorm)
EOF
echo done