# Latent Reasoning (HRM / TRM)

## 1. The object being computed

Reasoning happens in a latent space, not in chain-of-thought tokens.  A small
core is applied recursively to a latent state, with deep supervision, and the
answer is read out from the final latent.  No CoT tokens are generated.

## 2. Derivation

**HRM** (Hierarchical Reasoning Model): two modules at different timescales.
- Slow high-level planner $H$: updated every $N$ steps.
- Fast low-level executor $L$: updated every step, conditioned on $H$.

**TRM** (Tiny Recursive Model): a single 2-layer core, two latent tokens
(answer $z$, reasoning $y$):
$$
y \leftarrow f(y, z, x),\quad z \leftarrow g(z, y)
$$
repeated $T$ times, with deep supervision over $T$ steps.

**Deep supervision**: $\mathcal{L}=\sum_{t=1}^T \lambda^{T-t}\ell_t$ where
$\ell_t$ is the loss at recursion step $t$.  This is a variational bound /
weighted sum that forces every intermediate state to be decodable.

**Why no CoT**: the latent state can hold exponentially more information
per step than a token (continuous vs. discrete), and rollout is in latent
space without decoding.

## 3. Token-level walkthrough

Input $x$ is encoded once.  The latent $(y,z)$ is refined over $T$ steps;
at each step the core reads $x$ and the current latents.  The answer head
reads $z_T$.  No tokens are emitted during reasoning.

## 4. Hidden-space consequences

- Latent state norm grows then stabilises as reasoning converges.
- Deep-supervision losses decrease across steps (the model learns to
  reason in fewer effective steps).
- The tracer can watch `state/answer` and `state/reasoning` drift.

## 5. Complexity and failure modes

Params: tiny (7M in TRM).  Compute: $T$ core applications per example.
Failure: latent collapse if supervision is too weak; no interpretability of
the reasoning trace (by design).

## 6. References

- HRM (arXiv:2506.21734)
- TRM (arXiv:2510.04871)
- Probabilistic TRM (arXiv:2605.19943)
EOF
echo done