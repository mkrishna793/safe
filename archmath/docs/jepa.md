# JEPA (Joint-Embedding Predictive Architecture)

## 1. The object being computed

A world model that predicts in *latent* space, not pixel/token space.  An
online encoder $f_\theta$, an EMA target encoder $f_\xi$, and a predictor
$g_\phi$; the loss is $\|\hat s_y - \mathrm{sg}(s_y)\|^2$ plus a variance
regulariser.

## 2. Derivation

**Encoders**: $s_x=f_\theta(x)$, $s_y=f_\xi(y)$ (target is EMA of online).

**Predictor**: $\hat s_y = g_\phi(s_x, a)$ where $a$ is an action/context.

**Loss**: $\mathcal{L}=\|\hat s_y-\mathrm{sg}(s_y)\|^2+\lambda\,\mathrm{VarReg}(s_x)$.

**Collapse proof**: a constant encoder achieves $\mathcal{L}=0$ (predictor
outputs the constant, target is the constant).  Hence the regulariser is
*necessary*, not cosmetic.  Stop-gradient + EMA target + variance regulariser
break the collapse.

**Why latent space**: predicting in latent space removes the pixel/token
reconstruction bias; a latent world model can be rolled out without decoding.

## 3. Token-level walkthrough

Observations are encoded to latents; the predictor rolls the latent state
forward under an action; the target encoder provides the (stop-grad) target.
The variance regulariser keeps the latent distribution spread out.

## 4. Hidden-space consequences

- `world/latent_std` is the collapse detector: it must stay $>0$.
- `world/loss` should decrease; if it hits 0 with std $\to 0$, collapse.
- The tracer can watch the EMA target drift slowly.

## 5. Complexity and failure modes

Params: encoder + predictor.  Rollout is in latent space (cheap).

Failure: collapse to a constant if regularisation is removed; mode collapse
of the predictor.

## 6. References

- LeCun 2022 -- autonomous machine intelligence
- I-JEPA (arXiv:2301.08243)
- V-JEPA (arXiv:2304.07193)
- DT-JEPA (arXiv:2610.32657)
EOF
echo done