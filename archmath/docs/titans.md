# Titans (Neural Test-Time Memory)

## 1. The object being computed

A Transformer where attention is short-term memory and a matrix-valued fast
weight $M$ is long-term memory, updated at test time by a surprise-driven
learning rule.  A gate mixes the two reads.

## 2. Derivation

**Memory update** (delta rule with forgetting and momentum):
$$
M_t = \gamma M_{t-1} - \alpha\, m_t,\quad
m_t = \beta\, m_{t-1} + (1-\beta)\,(M_{t-1}k_t-v_t)k_t^\top
$$
with surprise $=\|M_{t-1}k_t-v_t\|^2$.

**Read**: $r_t = M_{t-1} q_t$ (before the update, so the memory predicts
the current token from past context).

**Gate**: $g_t=\sigma(w_g^\top[a_t;r_t])$,
$y_t = g_t\,a_t + (1-g_t)\,r_t$.

**Connection to linear attention**: linear attention's state is updated by
the input ($S_t = S_{t-1}+v_t k_t^\top$); Titans' state is updated by a
*learning rule* on surprise.  Same shape, different semantics.

## 3. Token-level walkthrough

Token $t$: attention reads the short context, memory reads the long context,
the gate decides which to trust, residual carries the mix forward.  The
memory updates using the prediction error on $t$.

## 4. Hidden-space consequences

- Surprise decays over a sequence as the memory learns the prefix.
- Memory norm grows then stabilises under forgetting.
- Gate shifts from memory to attention as context accumulates.

## 5. Complexity and failure modes

Memory state: $O(d^2)$ per layer.  Time per token: $O(d^2)$.

Failure: if forgetting is too weak, memory saturates; if too strong, it
forgets too fast.  The gate can collapse to "always attention".

## 6. References

- Titans (arXiv:2501.00663)
- "Memory Layers at Scale" (product-key memory)
EOF
echo done