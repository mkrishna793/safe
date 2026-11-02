# State-Space Model (Mamba-2 SSD)

## 1. The object being computed

A sequence model with a fixed-size hidden state $h_t$ updated by a
diagonal (scalar-identity) linear recurrence, plus a width-4 convolution and
input-dependent gating.  No attention, no KV cache.

## 2. Derivation

**Continuous LTI system**: $\dot h(t)=A h(t)+B x(t)$, $y(t)=C^\top h(t)$.

**Discretisation** (zero-order hold): $h_t = \bar A h_{t-1} + \bar B x_t$,
$\bar A=\exp(\Delta A)$, $\bar B=(\bar A-I)A^{-1}B$.

**Mamba-2 (SSD)**: $A$ is scalar-identity (diagonal), so $\bar A_t$ is a
scalar per token: $h_t = \alpha_t h_{t-1} + B_t x_t$, $y_t = C_t^\top h_t$,
with $\alpha_t=\exp(-\Delta_t \exp(A))$.  The scalar-identity structure is
what makes the chunked (parallel) scan possible.

**Chunked scan**: within a chunk, the recurrence is a matrix multiply; across
chunks, only the final state carries over.  Cost $O(S\,d\,d_{\mathrm{state}})$
with high parallelism.

**Convolution**: a width-4 depthwise conv before the SSM provides local mixing
that the recurrence alone lacks.

## 3. Token-level walkthrough

For token $t$: conv mixes $x_{t-3..t}$, project to $B_t,C_t,\Delta_t$, update
$h_t=\alpha_t h_{t-1}+B_t x_t$, read $y_t=C_t^\top h_t$.  The state is the
entire prefix compressed into $d_{\mathrm{state}}$ dimensions per channel.

## 4. Hidden-space consequences

- State norm tracks how much history is retained; under strong decay it
  saturates.
- $\Delta_t$ is the per-token "step size": large $\Delta$ = reset memory,
  small $\Delta$ = accumulate.
- The tracer exposes `state/A_bar` to watch the decay schedule.

## 5. Complexity and failure modes

Time: $O(S\,d\,d_{\mathrm{state}})$.
Memory at inference: $O(d\,d_{\mathrm{state}})$ constant.

Failure: diagonal SSMs are provably weaker than attention on some
sequential-composition tasks; the conv helps but does not close the gap.

## 6. References

- Mamba (arXiv:2312.00752)
- Mamba-2 / SSD (arXiv:2405.21060)
- Nemotron-H (arXiv:2504.03624)
EOF
echo done