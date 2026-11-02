# Multi-Head Latent Attention (MLA)

## 1. The object being computed

Standard attention materialises $K,V\in\mathbb{R}^{S\times h d_h}$ in the KV
cache.  MLA compresses them to a latent $c\in\mathbb{R}^{d_c}$ with
$d_c\ll h d_h$, and at inference never expands them back -- the up-projection
matrices are *absorbed* into $W_Q$ and $W_O$.

## 2. Derivation

**Compression**: $c = W_{DKV} x\in\mathbb{R}^{d_c}$, then
$K = W_{UK} c$, $V = W_{UV} c$.

**Absorption**: the attention score
$$
q^\top k = (W_Q x)^\top (W_{UK} c) = (W_{UK}^\top W_Q x)^\top c
$$
so $W_{UK}$ folds into $W_Q$: define $\tilde Q = W_{UK}^\top W_Q x$ and
score $= \tilde Q^\top c$.  Similarly $V$'s contribution folds into $W_O$:
output $= W_O (P V) = (W_O W_{UV}) (P c)$.  The cache stores only $c$.

**Decoupled RoPE**: RoPE is not compatible with absorption (it would have to
be applied to $c$, breaking the linear fold).  So MLA keeps a small separate
rotary key $k^R$ outside the latent, concatenated to the (absorbed) main key.

**Rank argument**: $K=W_{UK}C$ has rank $\le d_c$ regardless of $S$.  By
Eckart-Young, the best rank-$d_c$ approximation to the true $K$ is its
truncated SVD; MLA *is* that factorisation, learned end to end.  Hence
compression is lossless up to rank $d_c$.

## 3. Token-level walkthrough

For token $t$: compress $x_t$ to $c_t$ (cached), compute the rotary key
$k^R_t$, and the (absorbed) query.  Attention score to position $s$ is
$\tilde q_t^\top c_s + (q^R_t)^\top k^R_s$.  Value aggregation uses the
absorbed $W_O W_{UV}$ applied to $c_s$.  Only $c_s$ and $k^R_s$ are cached.

## 4. Hidden-space consequences

- Latent norm $\|c\|$ shows how much information survives compression.
- Compression ratio $d_c / (2 h d_h)$ is the headline number (e.g. 4x-8x).
- The tracer can compare the naive (expand) and absorbed paths: their
  outputs must agree to floating point.

## 5. Complexity and failure modes

KV cache: $O(L\,d_c\,S)$ instead of $O(L\,2 h d_h\,S)$.

Failure modes: if $d_c$ is too small, rank bottleneck loses information;
decoupled RoPE dimensions must stay small or they dominate the cache.

## 6. References

- DeepSeek-V2 (arXiv:2405.04434)
- DeepSeek-V3.2 / DSA (arXiv:2512.02556)
- DeepSeek-V4.1-Flash (arXiv:2609.19969)
EOF
echo done