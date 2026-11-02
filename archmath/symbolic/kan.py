"""Symbolic spec for the KAN architecture."""
from __future__ import annotations

from ..core import MathSpec, register_spec


@register_spec("kan")
def kan_spec() -> MathSpec:
    return MathSpec(
        name="kan",
        family="kolmogorov-arnold",
        summary="KAN: learnable univariate B-spline functions on edges, from the Kolmogorov-Arnold representation theorem.",
        equations=[
            ("ka_theorem", r"f(x_1,\dots,x_n)=\sum_{q=1}^{2n+1}\Phi_q\!\left(\sum_{p=1}^n \phi_{q,p}(x_p)\right),\quad \phi_{q,p}:[0,1]\to\mathbb{R}"),
            ("kan_layer", r"y_j = \sum_{i=1}^{n} \phi_{ij}(x_i),\quad \phi_{ij}(x)=w_b\,\mathrm{SiLU}(x)+w_s\sum_{l} c_l\,B_{l,k}(x)"),
            ("bspline_basis", r"B_{i,0}(u)=\mathbf{1}_{[t_i,t_{i+1})}(u),\quad B_{i,k}(u)=\frac{u-t_i}{t_{i+k}-t_i}B_{i,k-1}(u)+\frac{t_{i+k+1}-u}{t_{i+k+1}-t_{i+1}}B_{i+1,k-1}(u)"),
            ("param_count", r"P_{\text{KAN}}=n\,m\,(G+k+1),\quad P_{\text{MLP}}=4\,n\,m"),
            ("crossover", r"\frac{P_{\text{KAN}}}{P_{\text{MLP}}}=\frac{G+k+1}{4};\ \text{break-even }G=3-k\ (\text{negative for }k=3)"),
        ],
        assumptions=[
            "The KA theorem is an EXISTENCE result for continuous f on [0,1]^n; it says nothing about learnability or sample complexity.",
            "B-spline basis functions form a partition of unity on the clamped domain, so the residual base (SiLU) is optional but stabilises training.",
            "Inputs must be normalised into the spline domain [0,1]; extrapolation outside the grid is linear (order 1) and unreliable.",
        ],
        complexity={
            "params_per_layer": "d_model*d_model*(kan_grid+kan_order+1)",
            "time_per_token": "d_model*d_model*(kan_grid+kan_order+1)",
            "break_even_grid": "3 - kan_order",
        },
        invariants=[
            "B-spline basis sums to 1 (partition of unity) on [0, grid]",
            "output shape [B, S, m] for input [B, S, n]",
            "gradients finite for finite inputs",
        ],
        references=[
            "Liu et al. 2024 (arXiv:2404.19756) -- KAN",
            "Cox 1972 / de Boor 1972 -- B-spline recursion",
        ],
    )
