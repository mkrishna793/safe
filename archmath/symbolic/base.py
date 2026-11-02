"""Symbolic utilities: complexity algebra, shape flow, numeric Jacobians."""

from __future__ import annotations

from typing import Callable

import sympy as sp
import torch

from ..core import ArchConfig, MathSpec


_SAFE = {
    "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs,
    "max": sp.Max, "min": sp.Min, "ceil": sp.ceiling, "floor": sp.floor,
}

#: Dimensional shorthands.  Mathematical writing uses d, L, h, V, S; configs use
#: d_model, n_layers, n_heads, vocab, seq.  Both must work, otherwise every spec
#: author has to remember which universe they are in -- and they will not.
ALIASES: dict[str, str] = {
    "d": "d_model",
    "L": "n_layers",
    "h": "n_heads",
    "V": "vocab",
    "S": "seq",
    "N": "n_experts",
    "k": "top_k",
    "C": "chunk",
    "H": "n_heads",
    "dk": "d_head",
    "dv": "d_head",
}


def symbols_for(cfg: ArchConfig) -> dict[str, sp.Symbol]:
    """Turn every numeric config value into a sympy symbol (or constant).

    Aliases share the *same* symbol object as their target, so an expression may
    freely mix ``d`` and ``d_model``.
    """
    syms: dict[str, sp.Symbol] = {}
    for k, v in cfg.params.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        syms[k] = sp.Symbol(k, positive=True, integer=isinstance(v, int))
    for alias, target in ALIASES.items():
        if target in syms and alias not in syms:
            syms[alias] = syms[target]
    return syms


def evaluate(expr: str, cfg: ArchConfig, extra: dict[str, float] | None = None) -> float:
    """Evaluate a complexity expression against a concrete config.

    Unknown identifiers become free symbols and are reported, not silently
    treated as zero -- a wrong complexity number is worse than a missing one.
    """
    table = symbols_for(cfg)
    if extra:
        table.update({k: sp.Symbol(k) for k in extra})
    local = {**_SAFE, **table}
    parsed = sp.sympify(expr, locals=local)
    free = parsed.free_symbols - set(table.values())
    if free:
        raise ValueError(f"unbound symbols in {expr!r}: {sorted(map(str, free))}")
    subs = {s: cfg.params.get(str(s), 0) for s in parsed.free_symbols}
    if extra:
        subs.update({sp.Symbol(k): v for k, v in extra.items()})
    return float(parsed.subs(subs).evalf())


def complexity_report(spec: MathSpec, cfg: ArchConfig) -> dict[str, float]:
    return {k: evaluate(v, cfg) for k, v in spec.complexity.items()}


def numeric_jacobian(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Finite-difference Jacobian of fn at x.  x: [d], fn: [d] -> [m]."""
    x = x.detach().float()
    base = fn(x)
    J = torch.zeros(base.numel(), x.numel())
    for i in range(x.numel()):
        xp = x.clone(); xp[i] += eps
        xm = x.clone(); xm[i] -= eps
        J[:, i] = ((fn(xp) - fn(xm)) / (2 * eps)).reshape(-1)
    return J


def spectral_radius(J: torch.Tensor) -> float:
    """Largest |eigenvalue| -- the stability knob for recurrent/looped blocks."""
    if J.numel() == 0:
        return float("nan")
    return float(torch.linalg.eigvals(J).abs().max())


def participation_ratio(J: torch.Tensor) -> float:
    s = torch.linalg.svdvals(J)
    s = s[s > 0]
    return float((s.sum() ** 2) / (s.pow(2).sum() + 1e-12)) if s.numel() else 0.0


def contractivity_report(fn: Callable[[torch.Tensor], torch.Tensor], d: int, seed: int = 0) -> dict[str, float]:
    """Measure whether a recurrent map is contractive (stable) or expansive."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(d, generator=g)
    J = numeric_jacobian(fn, x)
    return {
        "spectral_radius": spectral_radius(J),
        "frobenius_norm": float(J.norm()),
        "participation_ratio": participation_ratio(J),
    }
