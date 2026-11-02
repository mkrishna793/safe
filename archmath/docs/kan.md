# KAN (Kolmogorov-Arnold Network)

## 1. The object being computed

A network where learnable univariate functions sit on *edges* (not nodes):
$y_j = \sum_i \phi_{ij}(x_i)$, with each $\phi_{ij}$ a B-spline.  Based on
the Kolmogorov-Arnold representation theorem.

## 2. Derivation

**KA theorem**: any continuous $f:[0,1]^n\to\mathbb{R}$ can be written as
$f(x)=\sum_{q=1}^{2n+1}\Phi_q(\sum_{p=1}^n \phi_{q,p}(x_p))$.  This is an
*existence* result: it says nothing about learnability or sample complexity.

**KAN layer**: $\phi_{ij}(x)=w_b\,\mathrm{SiLU}(x)+w_s\sum_l c_l B_{l,k}(x)$
(residual base + spline).

**B-spline recursion** (Cox-de Boor):
$$
B_{i,0}(u)=\mathbf{1}_{[t_i,t_{i+1})}(u),\quad
B_{i,k}(u)=\frac{u-t_i}{t_{i+k}-t_i}B_{i,k-1}(u)
+\frac{t_{i+k+1}-u}{t_{i+k+1}-t_{i+1}}B_{i+1,k-1}(u)
$$

**Parameter count**: $n\,m\,(G+k+1)$ vs $4\,n\,m$ for a same-width MLP.
The ratio is $(G+k+1)/4$; for $k=3$ the break-even grid is $G=0$, so a KAN
is *never* cheaper than a same-width two-layer MLP by raw parameter count.

## 3. Token-level walkthrough

Each input dimension $x_i$ is transformed by $m$ learned univariate functions
(one per output), summed per output.  The spline basis is local, so gradients
are sparse in the coefficient space.

## 4. Hidden-space consequences

- `kan/basis` shows the evaluated basis functions (should be smooth).
- `kan/edge_energy` shows which edges matter.
- The tracer can watch coefficient drift.

## 5. Complexity, scaling and honest limits

Params per layer: $d^2(G+k+1)$.  Time: same (spline eval is the cost).

**Honest scaling**: spline evaluation is memory-bound on GPUs; KANs have not
beaten MLPs at LLM scale as of 2026.  The advantage, where it exists, is
interpretability and accuracy on low-dimensional smooth targets.

## 6. References

- KAN (arXiv:2404.19756)
- Cox 1972 / de Boor 1972 -- B-spline recursion
EOF
echo done