# Byte Latent Transformer (BLT)

## 1. The object being computed

A byte-level LM with no tokenizer.  Bytes are grouped into *patches* whose
boundaries are placed where the next byte is surprising; only patches reach
the global Transformer.  Cost is $O(P^2)$ over patches plus $O(N)$ locally.

## 2. Derivation

**Entropy patching**: compute next-byte entropy
$H_i = -\sum_v p_i(v)\log p_i(v)$ from a small local model.  Start a new
patch when $H_i$ exceeds a global threshold (with a patch-size cap).  Entropy
is the right signal because it is the local surprise: high surprise = the
current patch cannot predict the next byte, so cut here.

**Three modules**:
- Local encoder: $h_i=\mathrm{CausalConv}(\mathrm{Embed}(x_i))$,
  pooled to $z_p=W_e\,\mathrm{mean}_{i\in p}h_i$.
- Global latent Transformer over $P$ patch latents (causal in patch order).
- Local decoder: $p(x_i\mid x_{<i})=\mathrm{softmax}(W_d[u_{p(i)-1};h^{\mathrm{ctx}}_i])$,
  reading the *previous* patch's global state (keeps autoregressivity).

## 3. Token-level walkthrough

For byte $i$: local conv embeds it, entropy decides whether a new patch starts,
pooling aggregates the patch, the global Transformer (over patches) produces
$u_p$, and the local decoder predicts the next byte from $u_{p-1}$ and the
local context.  The global attention never sees individual bytes.

## 4. Hidden-space consequences

- `patch/boundaries` shows where cuts happen; `patch/sizes` shows the
  distribution; `patch/count` is $P$.
- Patch collapse to single bytes on noisy data (entropy always high).
- The tracer can watch patch count vs. sequence length.

## 5. Complexity and failure modes

Global attention is over $P$ patches, not $N$ bytes: $O(P^2 + N d_{\mathrm{local}})$.
Failure: patch collapse on noisy data; too few patches loses local detail.

## 6. References

- BLT (arXiv:2412.09871)
- Fast BLT (arXiv:2605.08044)
EOF
echo done