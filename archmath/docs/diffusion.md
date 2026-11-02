# Diffusion Language Model (LLaDA)

## 1. The object being computed

A generative model over discrete tokens via a forward masking process and a
learned reverse (denoising) process.  Generation is any-order and iterative.

## 2. Derivation

**Forward process**: $q(x_t\mid x_0)=\prod_i q(x_{t,i}\mid x_{0,i})$ with
$q(\mathrm{mask}\mid x_0)=t$, $q(x_0\mid x_0)=1-t$.  Each token is masked
independently with probability $t$.

**Reverse**: $p_\theta(x_0\mid x_t)$ predicts all masked tokens in parallel
using bidirectional attention (the objective is a marginal likelihood over
the whole sequence, not a chain rule, so bidirectional is valid).

**ELBO**: 
$$
\mathcal{L}=\mathbb{E}_t\,\mathbb{E}_{x_t}\left[
\sum_{i:\,\text{masked}} \frac{-\log p_\theta(x_{0,i}\mid x_t)}{1-t}
\right]
$$
The $1/(1-t)$ weight equalises the contribution of every noise level: a
token masked with probability $t$ is predicted with weight $1/(1-t)$.

**Sampling**: iterative unmasking, $k$ tokens per step, ordered by confidence
(highest predicted probability first).

## 3. Token-level walkthrough

At step $t$: a fraction $t$ of tokens are [MASK].  The model predicts
distributions for all masked positions in parallel.  Unmask the top-$k$ by
confidence.  Repeat until no masks remain.

## 4. Hidden-space consequences

- Confidence rises monotonically across denoising steps.
- Mask fraction follows the schedule $t$.
- Attention becomes less diffuse as $t\to 0$ (the model commits).

## 5. Complexity and failure modes

Time per sample: $T\cdot O(S^2 d)$ (T denoising steps).  Parallelism within
a step is full (all positions at once).

Failure: low-quality samples if too few steps; the any-order property means
the model can revise earlier decisions, which helps coherence but can
oscillate.

## 6. References

- LLaDA (arXiv:2502.09992)
- iLLaDA (arXiv:2606.25331)
- Mercury (arXiv:2506.17298)
EOF
echo done