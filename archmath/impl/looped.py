"""Looped / recurrent-depth Transformer.

A stack of transformer blocks is applied ``n_loops`` times.  The weights of the
repeated part are shared, so depth grows without growing the parameter count:

    h_0 = Embed(tokens)
    h_{k+1} = F_k(h_k, e),        k = 0 .. K-1
    F_k = Core_k with loop-specific low-rank corrections  W + B_k A_k

Three placements of the recurrent part (``mode``):

    full    the whole ``n_layers`` stack is the core; K full passes
    middle  the first and last ``n_unique/2`` blocks run once; the contiguous
            middle slice is repeated K times
    core    a deliberately small core (``n_core`` blocks) is repeated K times,
            with unique entry/exit blocks around it

Three ways of feeding a loop (``loop_connection``):

    recurrent  b = h                      textbook fixed-point iteration
    inject     b = g * RMS(e) + (1-g) * RMS(h)          (normalized unshifted)
    shifted    b = g * RMS(e_n) + (1-g) * RMS(h_{n-1})  (Parallel Loop Transformer)

``shifted`` is the Parallel Loop Transformer of Ekbote et al. (2026): the state
of loop k-1 at the *previous token position* is used, which makes all K loop
stages of token n available at the same time during decoding.  The two streams
are normalized separately before they are mixed -- with a direct sum the larger
of the two silently erases the other, and the paper shows this single detail is
what makes parallel looping competitive.  The current-token embedding is always
kept, otherwise prediction errors accumulate along the token axis.

Every loop also produces logits (deep supervision), and a halting head emits
``p_halt`` per loop; the returned logits are the halt-weighted mixture.

    aux  state/loop_h      [K, B, S, d]   hidden state after each loop
         state/loop_delta  [K]            mean ||h_k - h_{k-1}||_2 per token
         state/loop_rel_delta [K]         the same, divided by ||h_k||
         state/loop_cosine [K-1]          mean cos(h_k, h_{k+1})
         gate/halt         [K]            mean p_halt per loop
         gate/halt_token   [K, B, S]      per-token halt probability
         gate/halt_cum     [K]            mean cumulative halt mass c_k
         gate/mix          [B, S]         mean of the input mixture gate
         loss/loop_aux     [K]            per-loop cross entropy
         loss/deep_sup     []             mean of loss/loop_aux
         loss/aux          []             halt-weighted deep supervision
         lm/logits         [B, S, vocab]  halt-weighted output
         lm/logits_loop    [K, B, S, vocab]

Input is ``tokens`` of shape [B, S] (long).  A float tensor [B, S, d_model] is
also accepted and treated as a pre-embedded sequence.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm, rope


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class LoopLoRALinear(nn.Module):
    """Linear map with one low-rank correction per loop index.

        y = x W^T + (alpha / r) * (x A_k^T) B_k^T

    ``B`` is zero-initialised, so every loop starts as the *same* map and
    specialises only if the task asks for it.  With ``rank == 0`` no adapter
    parameters exist at all: the loops are then identical by construction, not
    merely by initialisation.
    """

    def __init__(self, d_in: int, d_out: int, n_loops: int, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError("LoRA rank must be >= 0")
        self.base = nn.Linear(d_in, d_out, bias=False)
        self.rank = rank
        self.scale = (float(alpha) if alpha is not None else float(rank)) / max(rank, 1)
        if rank > 0:
            self.a = nn.Parameter(torch.empty(n_loops, rank, d_in))
            self.b = nn.Parameter(torch.zeros(n_loops, d_out, rank))
            nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        else:
            self.register_parameter("a", None)
            self.register_parameter("b", None)

    def forward(self, x: torch.Tensor, loop: int) -> torch.Tensor:
        y = self.base(x)
        if self.rank == 0:
            # No adapter exists, therefore F_0 = F_1 = ... = F_{K-1}.
            return y
        return y + self.scale * ((x @ self.a[loop].t()) @ self.b[loop].t())


class LoopSwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int, n_loops: int, rank: int) -> None:
        super().__init__()
        self.w_gate = LoopLoRALinear(d, d_ff, n_loops, rank)
        self.w_up = LoopLoRALinear(d, d_ff, n_loops, rank)
        self.w_down = LoopLoRALinear(d_ff, d, n_loops, rank)

    def forward(self, x: torch.Tensor, loop: int) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x, loop)) * self.w_up(x, loop), loop)


class LoopAttention(nn.Module):
    """Causal multi-head attention whose projections carry loop adapters."""

    def __init__(self, cfg: ArchConfig, n_loops: int, rank: int) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.n_kv_heads = cfg.get("n_kv_heads", cfg["n_heads"])
        self.d_head = cfg["d_head"]
        self.group = self.n_heads // self.n_kv_heads
        self.theta = cfg.get("rope_theta", 10_000.0)
        self.w_q = LoopLoRALinear(d, self.n_heads * self.d_head, n_loops, rank)
        self.w_k = LoopLoRALinear(d, self.n_kv_heads * self.d_head, n_loops, rank)
        self.w_v = LoopLoRALinear(d, self.n_kv_heads * self.d_head, n_loops, rank)
        self.w_o = LoopLoRALinear(self.n_heads * self.d_head, d, n_loops, rank)

    def forward(self, x: torch.Tensor, loop: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, _ = x.shape
        q = self.w_q(x, loop).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x, loop).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x, loop).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        pos = torch.arange(S, device=x.device)
        q = rope(q, pos, self.theta)
        k = rope(k, pos, self.theta)

        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, -1e9)
        probs = scores.softmax(-1)
        y = (probs @ v).transpose(1, 2).reshape(B, S, self.n_heads * self.d_head)
        return self.w_o(y, loop), {"attn/probs": probs, "attn/scores": scores}


class LoopBlock(ArchModule):
    """Pre-norm transformer block with per-loop low-rank adapters."""

    AUX_KEYS = ("attn/probs", "mlp/out")

    def __init__(self, cfg: ArchConfig, n_loops: int, rank: int) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = RMSNorm(d)
        self.attn = LoopAttention(cfg, n_loops, rank)
        self.norm2 = RMSNorm(d)
        self.mlp = LoopSwiGLU(d, cfg["d_ff"], n_loops, rank)

    def forward(self, x: torch.Tensor, loop: int = 0, **kw: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, aux = self.attn(self.norm1(x), loop)
        x = x + a
        m = self.mlp(self.norm2(x), loop)
        x = x + m
        aux["mlp/out"] = m
        return x, aux


class GatedMix(nn.Module):
    """Separate RMS normalization + low-rank gate for the two loop inputs.

        b = g  * RMS(e)  +  (1 - g) * RMS(h),     g = sigmoid(W2 SiLU(W1 [e; h]))

    The gate is shared across loops (the paper finds the gate form matters much
    less than the separate normalization), and adds < 0.02% parameters.
    """

    def __init__(self, d: int, rank: int = 8) -> None:
        super().__init__()
        self.norm_e = RMSNorm(d)
        self.norm_h = RMSNorm(d)
        self.down = nn.Linear(2 * d, rank, bias=False)
        self.up = nn.Linear(rank, d)

    def forward(self, e: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eh = self.norm_e(e)
        hh = self.norm_h(h)
        g = torch.sigmoid(self.up(F.silu(self.down(torch.cat([eh, hh], dim=-1)))))
        return g * eh + (1.0 - g) * hh, g


# --------------------------------------------------------------------------- #
# Layer placement
# --------------------------------------------------------------------------- #

def split_layers(
    n_layers: int,
    mode: str = "full",
    n_unique: int | None = None,
    n_core: int | None = None,
) -> tuple[int, int, int]:
    """Return (prefix, core, suffix) counts of stored blocks.

    ``middle`` follows the paper: the first and last floor(B/3) blocks are not
    looped.  ``core`` keeps the same idea but with an explicitly small core.
    Degenerate shallow stacks (n_layers <= 2) collapse ``middle`` into ``full``;
    use ``n_layers >= 3`` to see the difference.
    """
    if mode == "full":
        return 0, n_layers, 0
    if mode == "middle":
        u = n_layers // 3 if n_unique is None else int(n_unique)
        u = max(0, min(u, (n_layers - 1) // 2))
        return u, n_layers - 2 * u, u
    if mode == "core":
        c = max(1, n_layers // 2) if n_core is None else int(n_core)
        c = max(1, min(c, n_layers))
        rest = n_layers - c
        p = rest // 2
        return p, c, rest - p
    raise ValueError(f"unknown loop mode {mode!r}; expected 'full', 'middle' or 'core'")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

@register("looped")
class LoopedTransformer(ArchModule):
    """Recurrent-depth decoder: one block stack applied K times with loop LoRA."""

    AUX_KEYS = (
        "state/loop_h",
        "state/loop_delta",
        "gate/halt",
        "loss/loop_aux",
        "lm/logits",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.d_model = d
        self.n_loops = int(cfg.get("n_loops", 4))
        self.mode = str(cfg.get("mode", "full"))
        self.loop_connection = str(cfg.get("loop_connection", "recurrent"))
        self.rank = int(cfg.get("lora_rank", 0))
        self.use_halt = bool(cfg.get("halt", True))
        self.gate_rank = int(cfg.get("gate_rank", 8))

        if self.n_loops < 1:
            raise ValueError("n_loops must be >= 1")
        if self.loop_connection not in ("recurrent", "inject", "shifted"):
            raise ValueError(f"unknown loop_connection {self.loop_connection!r}")

        prefix, core, suffix = split_layers(
            int(cfg["n_layers"]), self.mode, cfg.get("n_unique"), cfg.get("n_core")
        )
        self.n_prefix, self.n_core, self.n_suffix = prefix, core, suffix

        self.embed = nn.Embedding(cfg["vocab"], d)
        self.prefix = nn.ModuleList(LoopBlock(cfg, self.n_loops, self.rank) for _ in range(prefix))
        self.core = nn.ModuleList(LoopBlock(cfg, self.n_loops, self.rank) for _ in range(core))
        self.suffix = nn.ModuleList(LoopBlock(cfg, self.n_loops, self.rank) for _ in range(suffix))
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.lm_head.weight = self.embed.weight  # tied

        self.halt_norm = RMSNorm(d)
        self.halt_head = nn.Linear(d, 1, bias=False)
        self.halt_bias = nn.Parameter(torch.zeros(self.n_loops))

        self.mix = GatedMix(d, self.gate_rank) if self.loop_connection != "recurrent" else None

        # rank 0 is a structural promise: no adapter parameters exist, hence
        # every loop is literally the same function.
        if self.rank == 0:
            for m in self.modules():
                if isinstance(m, LoopLoRALinear):
                    assert m.a is None and m.b is None, "rank 0 must not allocate loop adapters"

    # -- topology --------------------------------------------------------- #

    @property
    def effective_depth(self) -> int:
        """Number of transformer block evaluations per token."""
        return self.n_prefix + self.n_loops * self.n_core + self.n_suffix

    def _loop_input(self, e: torch.Tensor, h: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Feed loop k.  Returns (input, mixture gate or None)."""
        if k == 0 or self.mix is None:
            return h, None
        if self.loop_connection == "inject":
            return self.mix(e, h)
        # shifted: loop k-1's state at token n-1.  Causal by construction.
        prev = F.pad(h[:, :-1], (0, 0, 1, 0))
        return self.mix(e, prev)

    def _run_core(self, h: torch.Tensor, loop: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        y = h
        for i, blk in enumerate(self.core):
            y, a = blk(y, loop=loop)
            for key, val in a.items():
                aux[f"loop{loop}/layer{i}/{key}"] = val
        return y, aux

    # -- forward ---------------------------------------------------------- #

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None, **kw: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tokens.dim() == 3:
            e = tokens if tokens.shape[-1] == self.d_model else self.embed(tokens.long())
        else:
            e = self.embed(tokens.long())
        B, S, _ = e.shape
        V = self.cfg["vocab"]
        aux: dict[str, torch.Tensor] = {"embed/out": e}

        h = e
        for blk in self.prefix:
            h, _ = blk(h, loop=0)
        entry = h

        states: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        last_mix: torch.Tensor | None = None

        for k in range(self.n_loops):
            inp, g = self._loop_input(e, h, k)
            if g is not None:
                last_mix = g
            h, a = self._run_core(inp, k)
            for key, val in a.items():
                aux[key] = val
            states.append(h)

            p_halt = torch.sigmoid(self.halt_head(self.halt_norm(h)).squeeze(-1) + self.halt_bias[k])
            gates.append(p_halt)

        for blk in self.suffix:
            h, _ = blk(h, loop=self.n_loops - 1)

        # -- halting distribution (ACT) ----------------------------------- #
        K = self.n_loops
        p = torch.stack(gates, dim=0)                       # [K, B, S]
        c: list[torch.Tensor] = []
        remaining = torch.ones_like(p[0])
        for k in range(K):
            ck = p[k] * remaining
            c.append(ck)
            remaining = remaining * (1.0 - p[k])
        c[-1] = c[-1] + remaining                           # last loop absorbs the rest
        cum = torch.stack(c, dim=0)                         # proper distribution over loops

        logits_loop = torch.stack(
            [self.lm_head(self.norm_f(s)) for s in (states if K > 0 else [h])], dim=0
        )                                                   # [K, B, S, V]
        if self.use_halt:
            logits = torch.einsum("kbs,kbsv->bsv", cum, logits_loop)
        else:
            logits = logits_loop[-1]

        # -- deep supervision --------------------------------------------- #
        tgt = tokens if targets is None else targets
        losses: list[torch.Tensor] = []
        if tgt.dim() == 2 and tgt.shape[1] > 1:
            for k in range(K):
                losses.append(
                    F.cross_entropy(
                        logits_loop[k][:, :-1].reshape(-1, V),
                        tgt[:, 1:].reshape(-1),
                    )
                )
        else:
            zero = logits_loop.sum() * 0.0
            losses = [zero for _ in range(K)]
        loss_loop = torch.stack(losses)
        deep_sup = loss_loop.mean()
        halt_weighted = (cum.detach().mean(dim=(1, 2)) * loss_loop).sum()

        # -- hidden-space diagnostics ------------------------------------- #
        hs = torch.stack([s.detach() for s in states], dim=0)          # [K, B, S, d]
        prev_states = torch.cat([entry.detach().unsqueeze(0), hs[:-1]], dim=0)
        deltas = (hs - prev_states).norm(dim=-1).mean(dim=(1, 2))   # [K]
        rel = deltas / (hs.norm(dim=-1).mean(dim=(1, 2)) + 1e-8)
        if K > 1:
            cos = F.cosine_similarity(hs[:-1].flatten(2), hs[1:].flatten(2), dim=-1).mean(dim=1)
        else:
            cos = torch.zeros(0, device=e.device)

        aux.update(
            {
                "state/loop_h": hs,
                "state/loop_delta": deltas,
                "state/loop_rel_delta": rel,
                "state/loop_cosine": cos,
                "state/final": h,
                "gate/halt": p.detach().mean(dim=(1, 2)),
                "gate/halt_token": p.detach(),
                "gate/halt_cum": cum.detach().mean(dim=(1, 2)),
                "gate/halt_entropy": -(cum.detach().clamp_min(1e-9).log() * cum.detach()).sum(0).mean(),
                "gate/mix": (last_mix.detach().mean(-1) if last_mix is not None else torch.zeros(B, S, device=e.device)),
                "loss/loop_aux": loss_loop.detach(),
                "loss/deep_sup": deep_sup.detach(),
                "loss/aux": halt_weighted.detach(),
                "lm/logits": logits.detach(),
                "lm/logits_loop": logits_loop.detach(),
                "lm/logits_last": logits_loop[-1].detach(),
                "hidden/final": self.norm_f(h),
            }
        )
        return logits, aux

    # -- diagnostics ------------------------------------------------------ #

    @torch.no_grad()
    def loop_map_deviation(self, tokens: torch.Tensor) -> float:
        """max_k ||F_k(h) - F_0(h)||_inf on a shared input.

        With ``lora_rank=0`` this MUST be exactly 0.0: the loops are the same
        mathematical map, even though their iterates differ.
        """
        e = self.embed(tokens.long())
        h = e
        for blk in self.prefix:
            h, _ = blk(h, loop=0)
        outs = [self._run_core(h, k)[0] for k in range(self.n_loops)]
        return max(float((o - outs[0]).abs().max()) for o in outs)

    def assert_rank0_identity(self, tokens: torch.Tensor, tol: float = 0.0) -> float:
        """Assert the rank-0 contract and return the measured deviation."""
        if self.rank != 0:
            raise ValueError("assert_rank0_identity requires lora_rank=0")
        dev = self.loop_map_deviation(tokens)
        assert dev <= tol, f"rank-0 loops are not identical maps: max deviation {dev}"
        return dev

    def measure_spectral_radius(self, tokens: torch.Tensor, position: int | None = None, loop: int = 1) -> float:
        """Largest |eigenvalue| of the full loop map dF_k/dh at one position.

        ``F_k`` includes the loop-input connection (re-injection / shift) and
        the core stack.  This is the rho of the contraction argument: rho < 1
        predicts that loop deltas decay geometrically and that extra loops stop
        buying information.  Loop 0 is skipped because it is a function of the
        embedding only, not of a previous state.

        Diagnostic only -- never called from forward.  Returns ``nan`` when
        ``n_loops == 1``.
        """
        if self.n_loops < 2:
            return float("nan")
        k = max(1, min(int(loop), self.n_loops - 1))
        was_training = self.training
        self.eval()
        try:
            e = self.embed(tokens.long())
            with torch.no_grad():
                h = e
                for blk in self.prefix:
                    h, _ = blk(h, loop=0)
            h = h.detach()
            B, S, d = h.shape
            pos = S // 2 if position is None else position

            def fn(vec: torch.Tensor) -> torch.Tensor:
                v = vec.reshape(B, 1, d)
                hh = torch.cat([h[:, :pos], v, h[:, pos + 1:]], dim=1)
                inp, _ = self._loop_input(e, hh, k)
                y = self._run_core(inp, k)[0]
                return y[:, pos, :].reshape(-1)

            x0 = h[:, pos, :].reshape(-1).detach()
            J = torch.autograd.functional.jacobian(fn, x0, vectorize=True)
            return float(torch.linalg.eigvals(J).abs().max())
        finally:
            self.train(was_training)

    def param_breakdown(self) -> dict[str, int]:
        """Parameters by role, for checking the efficiency claim numerically."""
        out = {"core": 0, "unique_layers": 0, "lora": 0, "embed_head": 0, "other": 0}
        for name, p in self.named_parameters():
            n = p.numel()
            if name.endswith(".a") or name.endswith(".b"):
                out["lora"] += n
            elif name.startswith("core."):
                out["core"] += n
            elif name.startswith("prefix.") or name.startswith("suffix."):
                out["unique_layers"] += n
            elif name.startswith("embed.") or name.startswith("lm_head."):
                out["embed_head"] += n
            else:
                out["other"] += n
        out["total"] = sum(out.values())
        return out


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def tiny_config(**over: Any) -> ArchConfig:
    p: dict[str, Any] = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        n_layers=2,
        d_ff=128,
        vocab=256,
        seq=16,
        n_loops=4,
        mode="full",
        loop_connection="inject",
        lora_rank=4,
        halt=True,
        gate_rank=8,
    )
    p.update(over)
    prefix, core, suffix = split_layers(int(p["n_layers"]), str(p["mode"]), p.get("n_unique"), p.get("n_core"))
    p["n_unique"] = prefix + suffix
    p["n_core"] = core
    return ArchConfig("looped-tiny", "looped", p)


# The symbolic specification lives in archmath/symbolic/looped.py.
from ..symbolic import looped as _symbolic_looped  # noqa: E402,F401
