"""ArchMath: a mathematical decoding + real-time tracing framework for AI architectures.

Two halves, deliberately separated:

  symbolic/  -- what the architecture *is*, as mathematics (equations, dims, complexity, invariants)
  impl/      -- what the architecture *does*, as executable PyTorch with exposed intermediates
  trace/     -- what the architecture *becomes* during training, measured in the hidden space

The contract every architecture obeys:

    block(x, **kw) -> (y, aux)

`aux` is a dict of named tensors that the tracer reads.  If an architecture cannot
expose an intermediate, the tracer cannot explain it -- so exposing aux is part of
implementing the architecture, not an afterthought.
"""

from __future__ import annotations

from .core import ArchConfig, ArchModule, MathSpec, get_arch, list_archs, register
from .trace import TraceSession, TensorStats, HiddenSpaceReport
from .manifold import RealTimeMathDecoder, ManifoldMetrics

__all__ = [
    "ArchConfig",
    "ArchModule",
    "MathSpec",
    "register",
    "get_arch",
    "list_archs",
    "TraceSession",
    "TensorStats",
    "HiddenSpaceReport",
    "RealTimeMathDecoder",
    "ManifoldMetrics",
]

__version__ = "0.1.0"
