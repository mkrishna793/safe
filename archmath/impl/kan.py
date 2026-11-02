"""Kolmogorov--Arnold Network (KAN) block, with a correct B-spline basis.

The layer here is the one from Liu et al. (2024), written out rather than
hidden behind ``pykan``:

    y_j = sum_i phi_{ij}(x_i)

    phi_{ij}(x) = [ w_b,ij * SiLU(x) ]  +  sum_{r=0}^{G+k-1} c_{ijr} B_{r,k}(u)
    u           = (clamp(x, -1, 1) + 1) / 2

where ``B_{r,k}`` is the degree-``k`` B-spline basis on ``G`` uniform intervals
of ``[0, 1]``, evaluated by the Cox--de Boor recursion (see
:func:`bspline_basis`).  ``w_b`` is the "KAN 2.0" residual base branch; it is
optional.  The spline scale ``w_s`` is absorbed into the coefficients ``c``,
because a per-edge scalar multiplying an otherwise free spline is not
identifiable.

Two things this file is deliberately honest about:

* The Kolmogorov--Arnold theorem is an *existence* statement.  It says every
  continuous ``f`` on ``[0,1]^n`` can be written as a sum of continuous
  univariate functions; it says nothing about whether gradient descent will
  find them, how smooth they are, or how many samples are needed.
* The B-spline implementation is the mathematical content.  It is a direct
  Cox--de Boor evaluation, not a kernel approximation, and it satisfies the
  partition-of-unity invariant to float precision.

The module accepts token ids ``[B, S]`` and returns logits ``[B, S, vocab]``.
``KANLayer`` itself accepts an arbitrary leading shape ``[..., n]``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm, rope


# --------------------------------------------------------------------------- #
# B-spline basis
# --------------------------------------------------------------------------- #

def bspline_knots(grid: int, order: int, *, dtype: torch.dtype = torch.float32,
                  device: torch.device | None = None) -> torch.Tensor:
    """Clamped uniform knot vector for ``grid`` intervals and degree ``order``.

    There are ``grid + 2*order + 1`` knots: ``order+1`` copies of 0, the
    ``grid-1`` interior knots ``1/grid, ..., (grid-1)/grid``, and ``order+1``
    copies of 1.  The number of degree-``order`` basis functions is therefore

        (#knots) - order - 1 = grid + order.

    The clamped ends make the basis interpolatory at u=0 and u=1, which is what
    lets a KAN represent functions on a closed interval without extrapolating.
    """
    if grid < 1:
        raise ValueError("grid must be >= 1")
    if order < 0:
        raise ValueError("order must be >= 0")
    interior = torch.arange(1, grid, dtype=dtype, device=device) / grid
    return torch.cat(
        [
            torch.zeros(order + 1, dtype=dtype, device=device),
            interior,
            torch.ones(order + 1, dtype=dtype, device=device),
        ]
    )


def bspline_basis(u: torch.Tensor, grid: int, order: int) -> torch.Tensor:
    """Evaluate every degree-``order`` B-spline basis function at ``u``.

    Parameters
    ----------
    u:
        Tensor of any shape with values in ``[0, 1]``.  Values outside are not
        clamped here; ``KANLayer`` clamps before calling.
    grid, order:
        Number of knot intervals and polynomial degree.

    Returns
    -------
    Tensor of shape ``u.shape + (grid + order,)``.  Entry ``[..., r]`` is
    ``B_{r,order}(u)``.

    The recursion (Cox--de Boor)
    ----------------------------
    With knot vector ``t`` and ``N_{i,0}(u) = 1[t_i <= u < t_{i+1}]``,

        N_{i,p}(u) = (u - t_i) / (t_{i+p} - t_i) * N_{i,p-1}(u)
                   + (t_{i+p+1} - u) / (t_{i+p+1} - t_{i+1}) * N_{i+1,p-1}(u)

    where a term with a zero-width knot span is defined to be zero.  The
    implementation keeps all ``#knots - 1`` degree-zero spans, then contracts
    one span per degree until ``grid + order`` functions remain.  This is the
    standard triangular scheme; it costs ``O(grid + order)`` per point and is
    differentiable in ``u``.
    """
    if u.numel() and (u.min() < 0 or u.max() > 1):
        raise ValueError("bspline_basis expects u in [0, 1]")

    t = bspline_knots(grid, order, dtype=u.dtype, device=u.device)
    n_knots = t.numel()

    # Degree 0: one indicator per knot span.  There are n_knots - 1 spans.
    n = (u.unsqueeze(-1) >= t[:-1]) & (u.unsqueeze(-1) < t[1:])
    n = n.to(u.dtype)
    # The last non-degenerate span is [ (grid-1)/grid, 1 ).  The final
    # order+1 spans are zero-length (repeated endpoint knots); close the
    # right endpoint on the last real span so that u=1 is represented.
    last_real_span = order + grid - 1
    n[..., last_real_span] = n[..., last_real_span] + (u == 1).to(u.dtype)

    for p in range(1, order + 1):
        m = n_knots - 1 - p  # number of degree-p basis functions
        den_l = t[p:p + m] - t[:m]
        den_r = t[p + 1:p + 1 + m] - t[1:1 + m]
        # Safe division: when a knot span has zero width the corresponding
        # basis function is zero on the support of the other term.  Using a
        # safe denominator instead of torch.where keeps the backward pass
        # finite (0 * inf would otherwise produce NaN gradients).
        den_l = torch.where(den_l.abs() > 0, den_l, torch.ones_like(den_l))
        den_r = torch.where(den_r.abs() > 0, den_r, torch.ones_like(den_r))
        left = (u.unsqueeze(-1) - t[:m]) / den_l
        right = (t[p + 1:p + 1 + m] - u.unsqueeze(-1)) / den_r
        n = left * n[..., :m] + right * n[..., 1:m + 1]

    return n


def bspline_eval(u: torch.Tensor, coeff: torch.Tensor, grid: int, order: int) -> torch.Tensor:
    """Evaluate ``sum_r coeff[..., r] B_r(u)`` for a batch of points."""
    basis = bspline_basis(u, grid, order)
    return torch.einsum("...r,...r->...", basis, coeff)


# --------------------------------------------------------------------------- #
# KAN layer
# --------------------------------------------------------------------------- #

class KANLayer(ArchModule):
    """One KAN layer: ``n`` inputs, ``m`` outputs, one spline per edge.

    Parameters per edge: ``grid + order`` spline coefficients, plus one base
    weight if the residual SiLU branch is enabled.  Total parameters:
    ``n * m * (grid + order + 1)`` with the base, ``n * m * (grid + order)``
    without it.

    Inputs are expected in ``[-1, 1]``.  The layer clamps outside that range,
    matching the standard KAN convention of a fixed spline grid; this is a real
    limitation, discussed in ``docs/kan.md``.  The transformer block applies a
    ``tanh`` before every KAN layer so that clamping is normally inactive.
    """

    AUX_KEYS = ("kan/basis", "kan/coeff", "kan/edge_energy")

    def __init__(self, n: int, m: int, grid: int = 5, order: int = 3,
                 use_base: bool = True) -> None:
        super().__init__()
        if n < 1 or m < 1:
            raise ValueError("n and m must be positive")
        self.n = n
        self.m = m
        self.grid = grid
        self.order = order
        self.n_basis = grid + order
        self.use_base = use_base

        # Small random splines at init: the residual base branch carries the
        # initial signal, and the coefficients grow only if the data asks.
        self.coeff = nn.Parameter(torch.randn(n, m, self.n_basis) * 0.05)
        if use_base:
            self.base = nn.Parameter(torch.empty(n, m))
            nn.init.kaiming_uniform_(self.base, a=math.sqrt(5))
        else:
            self.register_parameter("base", None)

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.shape[-1] != self.n:
            raise ValueError(f"expected last dim {self.n}, got {x.shape[-1]}")
        u = 0.5 * (x.clamp(-1.0, 1.0) + 1.0)
        basis = bspline_basis(u, self.grid, self.order)          # [..., n, G+k]
        spline = torch.einsum("...ik,ijk->...j", basis, self.coeff)
        if self.use_base:
            y = spline + F.silu(x) @ self.base
            energy = self.coeff.pow(2).mean(-1) + self.base.pow(2)
        else:
            y = spline
            energy = self.coeff.pow(2).mean(-1)

        aux = {
            "kan/basis": basis,
            "kan/coeff": self.coeff,
            "kan/edge_energy": energy,
        }
        return y, aux

    def extra_repr(self) -> str:
        return f"n={self.n}, m={self.m}, grid={self.grid}, order={self.order}, base={self.use_base}"


class KANMLP(ArchModule):
    """Two KAN layers as a drop-in replacement for the transformer MLP.

    ``d_model -> d_ff -> d_model``.  A ``tanh`` bounds each layer's input to the
    spline domain.  The first layer's aux keys are exposed as ``kan/*``; the
    second layer's as ``kan/*_out``.
    """

    AUX_KEYS = (
        "kan/basis", "kan/coeff", "kan/edge_energy",
        "kan/basis_out", "kan/coeff_out", "kan/edge_energy_out",
    )

    def __init__(self, d: int, d_ff: int, grid: int = 5, order: int = 3,
                 use_base: bool = True) -> None:
        super().__init__()
        self.l1 = KANLayer(d, d_ff, grid, order, use_base)
        self.l2 = KANLayer(d_ff, d, grid, order, use_base)

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h, a1 = self.l1(torch.tanh(x))
        y, a2 = self.l2(torch.tanh(h))
        aux: dict[str, torch.Tensor] = dict(a1)
        aux.update({f"{k}_out": v for k, v in a2.items()})
        aux["mlp/out"] = y
        return y, aux


# --------------------------------------------------------------------------- #
# Attention (same shape as the reference dense block)
# --------------------------------------------------------------------------- #

class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv_heads: int, d_head: int,
                 theta: float = 10_000.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.theta = theta
        self.group = n_heads // n_kv_heads
        self.w_q = nn.Linear(d, n_heads * d_head, bias=False)
        self.w_k = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.w_v = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.w_o = nn.Linear(n_heads * d_head, d, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, _ = x.shape
        q = self.w_q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        pos = torch.arange(S, device=x.device)
        q = rope(q, pos, self.theta)
        k = rope(k, pos, self.theta)

        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, -1e9)
        probs = scores.softmax(-1)
        y = probs @ v
        y = y.transpose(1, 2).reshape(B, S, self.n_heads * self.d_head)
        return self.w_o(y), {"attn/probs": probs, "attn/scores": scores}


class KANBlock(ArchModule):
    AUX_KEYS = ("attn/probs", "mlp/out", "kan/basis", "kan/coeff", "kan/edge_energy")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = nn.Parameter(torch.ones(d))
        self.norm2 = nn.Parameter(torch.ones(d))
        self.attn = CausalSelfAttention(
            d, cfg["n_heads"], cfg.get("n_kv_heads", cfg["n_heads"]), cfg["d_head"]
        )
        self.kan = KANMLP(
            d=d,
            d_ff=cfg["d_ff"],
            grid=cfg.get("kan_grid", 5),
            order=cfg.get("kan_order", 3),
            use_base=cfg.get("kan_base", True),
        )

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, aux = self.attn(rms_norm(x, self.norm1))
        x = x + a
        m, kaux = self.kan(rms_norm(x, self.norm2))
        x = x + m
        aux.update(kaux)
        aux["mlp/out"] = m
        return x, aux


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

@register("kan")
class KANTransformer(ArchModule):
    """Decoder-only transformer whose MLP is a two-layer KAN.

    Input: token ids ``[B, S]``.  Output: logits ``[B, S, vocab]``.
    """

    AUX_KEYS = (
        "attn/probs", "mlp/out", "kan/basis", "kan/coeff", "kan/edge_energy",
        "lm/logits",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(KANBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = nn.Parameter(torch.ones(d))
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.embed(tokens)
        aux: dict[str, torch.Tensor] = {"embed/out": x}
        last: dict[str, torch.Tensor] = {}
        for i, blk in enumerate(self.blocks):
            x, a = blk(x)
            last = a
            for k, v in a.items():
                aux[f"layer{i}/{k}"] = v
        x = rms_norm(x, self.norm_f)
        logits = self.lm_head(x)
        aux["lm/logits"] = logits
        aux["hidden/final"] = x
        # Also expose the last block's KAN internals under the canonical keys.
        for k in ("kan/basis", "kan/coeff", "kan/edge_energy"):
            if k in last:
                aux[k] = last[k]
        return logits, aux


def tiny_config(**over) -> ArchConfig:
    p = dict(
        d_model=64, n_heads=4, n_kv_heads=2, d_head=16,
        n_layers=2, d_ff=128, vocab=256, seq=16,
        kan_grid=5, kan_order=3, kan_base=True,
    )
    p.update(over)
    return ArchConfig("kan-tiny", "kan", p)


# --------------------------------------------------------------------------- #
# Parameter accounting and the fitting demo
# --------------------------------------------------------------------------- #

def kan_param_count(n: int, m: int, grid: int, order: int, use_base: bool = True) -> int:
    """Exact parameter count of one KAN layer: ``n*m*(G+k+[base])``."""
    return n * m * (grid + order + (1 if use_base else 0))


def crossover_report(n: int, m: int, grid: int = 5, order: int = 3) -> dict[str, float]:
    """Compare a KAN layer with the ``4*n*m`` two-layer MLP reference.

    ``4*n*m`` is the parameter count of an MLP ``n -> 2n -> m`` when ``n = m``
    (more generally ``2*n*(n+m)``).  The KAN/MLP ratio is
    ``(G + k + 1) / 4``.  Solving for the grid at which the two are equal gives
    ``G* = 3 - k``; for the usual ``k = 3`` that is ``G* = 0``.  In other words,
    **there is no grid size at which a KAN is cheaper than a same-width
    two-layer MLP with hidden width 2n**.  The interesting crossover is in
    accuracy per parameter and depends on the target's smoothness and intrinsic
    dimension, not on raw parameter count.  This function returns both the raw
    ratio and the width an MLP can afford at the KAN's budget.
    """
    kan = kan_param_count(n, m, grid, order, use_base=True)
    mlp = 4 * n * m
    k = grid + order + 1
    return {
        "kan_params": float(kan),
        "mlp_params": float(mlp),
        "ratio": k / 4.0,
        "break_even_grid": float(4 - order - 1),
        # An MLP n -> h -> m has h*(n+m) params.  At the KAN's budget,
        # h = n*m*(G+k+1)/(n+m); for n=m=d this is d*(G+k+1)/2, which is
        # 4.5*d at G=5,k=3 -- the fair comparison is a much wider MLP.
        "equal_budget_mlp_hidden": float(n * m * k / max(n + m, 1)),
    }


class _TinyKAN(nn.Module):
    """Two-layer KAN used by the demo (2 -> hidden -> 1)."""

    def __init__(self, hidden: int = 16, grid: int = 5, order: int = 3) -> None:
        super().__init__()
        self.l1 = KANLayer(2, hidden, grid, order, use_base=True)
        self.l2 = KANLayer(hidden, 1, grid, order, use_base=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.l1(x)
        y, _ = self.l2(torch.tanh(h))
        return y


def fit_sin_square_demo(
    steps: int = 300,
    n_grid: int = 32,
    hidden: int = 16,
    grid: int = 5,
    order: int = 3,
    lr: float = 1e-2,
    seed: int = 0,
) -> dict[str, float]:
    """Fit ``y = sin(x1) + x2**2`` on a 32x32 grid and report the final MSE.

    The inputs lie in ``[-0.9, 0.9]^2``, inside the spline domain.  This is a
    deliberately easy, low-dimensional, smooth target: it demonstrates that the
    basis and the optimizer work, not that KANs scale.  The same demo with a
    high-dimensional noisy target is a different story (see ``docs/kan.md``).
    """
    torch.manual_seed(seed)
    g = torch.linspace(-0.9, 0.9, n_grid)
    x1, x2 = torch.meshgrid(g, g, indexing="ij")
    x = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1)
    y = (torch.sin(x[:, 0]) + x[:, 1].pow(2)).unsqueeze(-1)

    model = _TinyKAN(hidden=hidden, grid=grid, order=order)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()

    return {
        "mse": float(loss.detach()),
        "steps": float(steps),
        "n_params": float(sum(p.numel() for p in model.parameters())),
        "grid": float(grid),
        "order": float(order),
    }


# The symbolic specification (contract file 2) lives in
# archmath/symbolic/kan.py.  Importing the implementation registers it
# too, so get_spec("kan") works after either import -- the same
# convenience the dense reference provides by defining its spec inline.
from ..symbolic import kan as _symbolic_kan  # noqa: E402,F401
