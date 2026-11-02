"""Core contracts: config, math spec, module base, registry."""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class ArchConfig:
    """A single point in an architecture's hyperparameter space.

    `params` must contain everything needed to instantiate the block AND to
    evaluate the symbolic complexity expressions.  Keep the names canonical:

        d_model, n_heads, n_kv_heads, d_head, n_layers, d_ff, vocab,
        n_experts, top_k, chunk, n_loops, ...
    """

    name: str
    family: str
    params: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ArchConfig.name is required")
        if not isinstance(self.params, dict):
            raise TypeError("ArchConfig.params must be a dict")

    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def merged(self, **overrides: Any) -> "ArchConfig":
        return ArchConfig(self.name, self.family, {**self.params, **overrides})

    def fingerprint(self) -> str:
        items = sorted(self.params.items())
        return f"{self.name}|" + "|".join(f"{k}={v}" for k, v in items)


# --------------------------------------------------------------------------- #
# Mathematics
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class MathSpec:
    """The mathematical identity of an architecture.

    equations : list of (label, expression).  Expressions are LaTeX-flavoured
                strings; they are documentation *and* the source for the Sage
                derivation scripts, which re-derive them symbolically.
    invariants: properties that MUST hold.  Each is checked numerically in the
                test suite (e.g. "attention rows sum to 1").
    complexity: symbolic expressions in the config params, e.g.
                {"time": "L * (4*d^2 + 2*d*d_ff)", "memory": "L*d*S"}
    """

    name: str
    family: str
    summary: str
    equations: list[tuple[str, str]]
    assumptions: list[str] = dataclasses.field(default_factory=list)
    complexity: dict[str, str] = dataclasses.field(default_factory=dict)
    invariants: list[str] = dataclasses.field(default_factory=list)
    references: list[str] = dataclasses.field(default_factory=list)

    def symbols(self) -> set[str]:
        """Rough symbol extraction for the Sage layer."""
        import re

        out: set[str] = set()
        for _, expr in self.equations:
            out.update(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\b", expr))
        for expr in self.complexity.values():
            out.update(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\b", expr))
        return out


# --------------------------------------------------------------------------- #
# Module contract
# --------------------------------------------------------------------------- #

class ArchModule(nn.Module, abc.ABC):
    """Base class for every architecture block.

    Subclasses MUST return ``(y, aux)``.  ``aux`` carries the intermediates that
    make the block explainable: attention weights, router logits, gate values,
    loop states, latent memory, patch boundaries, and so on.

    Naming convention for aux keys (the tracer groups by prefix):

        attn/*      attention probabilities, scores, masks
        route/*     router logits, expert assignments, load
        state/*     recurrent state, memory slots, loop hidden states
        gate/*      scalar gates (decay, halting, mixture weights)
        latent/*    anything in a compressed or latent space
        patch/*     byte/patch boundaries
        world/*     JEPA predictions, targets
    """

    #: canonical aux keys this block promises to expose
    AUX_KEYS: tuple[str, ...] = ()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""


def unwrap(output: Any) -> tuple[Any, dict[str, torch.Tensor]]:
    """Normalise a block output into (tensor, aux)."""
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
        return output[0], output[1]
    return output, {}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, type[ArchModule]] = {}
_SPECS: dict[str, Callable[[], MathSpec]] = {}


def register(name: str) -> Callable[[type[ArchModule]], type[ArchModule]]:
    """Register an executable architecture block."""

    def deco(cls: type[ArchModule]) -> type[ArchModule]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"architecture {name!r} already registered")
        _REGISTRY[name] = cls
        cls.arch_name = name  # type: ignore[attr-defined]
        return cls

    return deco


def register_spec(name: str) -> Callable[[Callable[[], MathSpec]], Callable[[], MathSpec]]:
    """Register the symbolic spec provider for an architecture."""

    def deco(fn: Callable[[], MathSpec]) -> Callable[[], MathSpec]:
        _SPECS[name] = fn
        return fn

    return deco


def get_arch(name: str) -> type[ArchModule]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown architecture {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def get_spec(name: str) -> MathSpec:
    if name not in _SPECS:
        raise KeyError(f"no symbolic spec for {name!r}; available: {sorted(_SPECS)}")
    return _SPECS[name]()


def list_archs() -> list[str]:
    return sorted(_REGISTRY)


def list_specs() -> list[str]:
    return sorted(_SPECS)


# --------------------------------------------------------------------------- #
# Numeric helpers shared by implementations
# --------------------------------------------------------------------------- #

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def softmax_entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """Shannon entropy of a probability tensor, in nats."""
    return -(p * (p + eps).log()).sum(dim=dim)


def effective_rank(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Participation ratio of the singular spectrum: (sum s)^2 / sum s^2.

    A cheap, differentiable-ish proxy for the number of directions carrying
    variance.  Collapses to 1 for rank-1, equals min(m,n) for isotropic.
    """
    x2 = x.reshape(-1, x.shape[-1]).float()
    s = torch.linalg.svdvals(x2)
    s = s.clamp_min(eps)
    return (s.sum() ** 2) / (s.pow(2).sum())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10_000.0) -> torch.Tensor:
    """Rotary position embedding applied to the last dimension (even split)."""
    d = x.shape[-1]
    half = d // 2
    freqs = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions.float().unsqueeze(-1) * freqs
    cos, sin = angles.cos(), angles.sin()
    cos = torch.cat([cos, cos], dim=-1).to(x.dtype)
    sin = torch.cat([sin, sin], dim=-1).to(x.dtype)
    return x * cos + rotate_half(x) * sin
