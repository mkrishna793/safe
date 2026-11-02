"""Gated DeltaNet: linear attention with the delta rule and per-channel decay.

Full LM over token ids ``[B, S]`` (embedding -> 2 pre-norm blocks -> tied head).

The layer keeps a recurrent state ``S in R^{d_state x d_head}`` per head.  With
``k_t, q_t in R^{d_state}``, ``v_t in R^{d_head}``, a per-channel decay
``alpha_t in R^{d_state}`` and a scalar write gate ``beta_t``::

    S_t = alpha_t (.) S_{t-1} + beta_t k_t (v_t - S_{t-1}^T k_t)^T
    o_t = S_t^T q_t

This is the contract's "decay-after-read" form.  Setting
``decay_before_read=True`` moves the decay in front of the read, which is the
canonical Gated DeltaNet recurrence
``S_t = alpha_t S_{t-1} + beta_t k_t (v_t - (alpha_t S_{t-1})^T k_t)^T``.
Both are implemented and both chunkwise paths are asserted against the
recurrent path in :func:`selfcheck`.

The chunked trainer uses the exact linear-algebra identity derived in
``docs/linear_attn.md``: a change of variables ``T_t = Delta_t^{-1} S_t``
(``Delta_t = prod_{j<=t} alpha_j``) turns the gated recurrence into a plain
delta rule with read keys ``x_t = Delta_{t-1} k_t`` (or ``Delta_t k_t`` for the
canonical form) and write keys ``k_t / Delta_t``.  The intra-chunk delta
residuals then solve one lower-triangular system of size ``chunk x chunk``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def decay_from_logits(a: torch.Tensor, negative: bool = False) -> torch.Tensor:
    """Per-channel gate in (0, 1) or (-1, 1).

    Positive branch: ``alpha = exp(-softplus(a))``.  The softplus keeps
    ``log alpha`` finite, so the chunked form never divides by zero.
    Negative branch (Solar Open 2 style): ``alpha = tanh(a)``, a signed
    eigenvalue that can flip the sign of a state direction from one token to
    the next.  The magnitude is clamped away from 0 only for the log-product
    bookkeeping; the recurrence uses the unclamped value.
    """
    if negative:
        return torch.tanh(a)
    return torch.exp(-F.softplus(a))


def _log_abs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.abs().clamp_min(eps).log()


def _sign(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))


def _split_pad(x: torch.Tensor, pad: int, value: float) -> torch.Tensor:
    if pad == 0:
        return x
    return F.pad(x, (0, 0, 0, pad), value=value)


# --------------------------------------------------------------------------- #
# Recurrent reference form
# --------------------------------------------------------------------------- #

def recurrent_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    decay_before_read: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plain O(S) recurrence, one step at a time.

    Shapes: ``q,k,alpha`` -> ``[B,H,S,d_state]``, ``v`` -> ``[B,H,S,d_head]``,
    ``beta`` -> ``[B,H,S]``.  Returns ``(o, states, deltas)`` with ``states``
    ``[B,H,S,d_state,d_head]`` and ``deltas`` ``[B,H,S,d_head]``.
    """
    B, H, S, dk = k.shape
    dv = v.shape[-1]
    state = q.new_zeros(B, H, dk, dv)
    outs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    for t in range(S):
        kt, vt = k[:, :, t], v[:, :, t]
        at, bt = alpha[:, :, t], beta[:, :, t]
        if decay_before_read:
            state = at.unsqueeze(-1) * state
            pred = torch.einsum("bhk,bhkv->bhv", kt, state)
            delta = vt - pred
            state = state + bt[:, :, None, None] * kt.unsqueeze(-1) * delta.unsqueeze(-2)
        else:
            pred = torch.einsum("bhk,bhkv->bhv", kt, state)
            delta = vt - pred
            state = at.unsqueeze(-1) * state
            state = state + bt[:, :, None, None] * kt.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], state))
        states.append(state)
        deltas.append(delta)
    return torch.stack(outs, 2), torch.stack(states, 2), torch.stack(deltas, 2)


# --------------------------------------------------------------------------- #
# Chunkwise parallel form
# --------------------------------------------------------------------------- #

def chunkwise_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk: int = 8,
    decay_before_read: bool = False,
    return_coeff: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Chunkwise exact form: intra-chunk triangular solve + inter-chunk carry.

    Returns ``(o, final_state, chunk_states, deltas, coeff)`` where
    ``chunk_states`` is ``[B,H,n_chunks,d_state,d_head]`` (state after each
    chunk) and ``coeff`` is the signed intra-chunk kernel ``[B,H,S,chunk]``.

    The identity (derivation in the docs): with ``Delta_i = prod_{r<=i} alpha_r``
    inside the chunk and ``kbar_j = k_j / Delta_j``,

        d_i = v_i - S_0^T x_i - sum_{j<i} beta_j (kbar_j . x_i) d_j
        o_i = S_0^T (Delta_i . q_i) + sum_{j<=i} beta_j (kbar_j . (Delta_i . q_i)) d_j
        S_C = Delta_C . S_0 + sum_{j<=C} beta_j (Delta_C/Delta_j) . k_j d_j^T

    with ``x_i = Delta_{i-1} . k_i`` (contract form) or ``Delta_i . k_i``
    (canonical Gated DeltaNet form).  The first two lines are one masked
    ``C x C`` solve plus two matmuls; only ``S_C`` is carried across chunks.
    """
    B, H, S, dk = k.shape
    dv = v.shape[-1]
    n_chunks = (S + chunk - 1) // chunk
    pad = n_chunks * chunk - S
    # Pad with alpha=1, beta=0, q=k=v=0 so padded steps are exact no-ops.
    q = _split_pad(q, pad, 0.0)
    k = _split_pad(k, pad, 0.0)
    v = _split_pad(v, pad, 0.0)
    alpha = _split_pad(alpha, pad, 1.0)
    beta = F.pad(beta, (0, pad), value=0.0) if pad else beta

    state = q.new_zeros(B, H, dk, dv)
    out = q.new_zeros(B, H, n_chunks * chunk, dv)
    coeff_all = q.new_zeros(B, H, n_chunks * chunk, chunk)
    delta_all = q.new_zeros(B, H, n_chunks * chunk, dv)
    chunk_states: list[torch.Tensor] = []

    tril = torch.ones(chunk, chunk, device=q.device, dtype=torch.bool).tril()
    strict = torch.ones(chunk, chunk, device=q.device, dtype=torch.bool).tril(-1)
    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)

    for c in range(n_chunks):
        sl = slice(c * chunk, (c + 1) * chunk)
        qq, kk, vv = q[:, :, sl], k[:, :, sl], v[:, :, sl]
        aa, bb = alpha[:, :, sl], beta[:, :, sl]

        # Signed cumulative decay inside the chunk: Delta_i = sign_i * exp(g_i).
        g = _log_abs(aa).cumsum(dim=2)                  # [B,H,C,dk]
        sgn = _sign(aa).cumprod(dim=2)                  # [B,H,C,dk]
        delta_i = sgn * g.exp()                         # [B,H,C,dk]

        # R[i,j] = Delta_i / Delta_j for i >= j, zero elsewhere.
        diff = g.unsqueeze(3) - g.unsqueeze(2)          # [B,H,C,C,dk]
        ratio = (sgn.unsqueeze(3) * sgn.unsqueeze(2)) * diff.exp()
        ratio = ratio * tril[None, None, :, :, None]

        # Read keys x_i and the ratio Delta_{i-1}/Delta_j used by the solve.
        one = q.new_ones(B, H, 1, dk)
        zero = q.new_zeros(B, H, 1, dk)
        delta_prev = torch.cat([one, delta_i[:, :, :-1]], dim=2)
        g_prev = torch.cat([zero, g[:, :, :-1]], dim=2)
        sgn_prev = torch.cat([one, sgn[:, :, :-1]], dim=2)
        x = (delta_i if decay_before_read else delta_prev) * kk

        gi, si = (g, sgn) if decay_before_read else (g_prev, sgn_prev)
        diff_p = gi.unsqueeze(3) - g.unsqueeze(2)
        ratio_p = (si.unsqueeze(3) * sgn.unsqueeze(2)) * diff_p.exp()
        ratio_p = ratio_p * strict[None, None, :, :, None]

        # L[i,j] = beta_j * (k_i . (Delta_{i-1}/Delta_j . k_j)),  j < i.
        lower = (kk.unsqueeze(3) * ratio_p * kk.unsqueeze(2)).sum(-1)
        lower = lower * bb.unsqueeze(2)
        lower = lower * strict[None, None, :, :]

        rhs = vv - torch.einsum("bhcd,bhdk->bhck", x, state)
        d = torch.linalg.solve_triangular(lower + eye, rhs, upper=False, unitriangular=True)

        # Signed intra-chunk kernel C[i,j] = beta_j * (k_j . (Delta_i/Delta_j . q_i)).
        coeff = (kk.unsqueeze(2) * ratio * qq.unsqueeze(3)).sum(-1)
        coeff = coeff * bb.unsqueeze(2)
        coeff = coeff * tril[None, None, :, :]

        qbar = delta_i * qq
        inter = torch.einsum("bhcd,bhdk->bhck", qbar, state)
        intra = torch.einsum("bhcj,bhjv->bhcv", coeff, d)
        out[:, :, sl] = inter + intra
        coeff_all[:, :, sl] = coeff
        delta_all[:, :, sl] = d

        # Carry S_C = Delta_C . S_0 + sum_j beta_j (Delta_C/Delta_j) . k_j d_j^T.
        delta_c = delta_i[:, :, -1]
        kbar_c = bb.unsqueeze(-1) * ratio[:, :, -1] * kk
        state = delta_c.unsqueeze(-1) * state + torch.einsum("bhjd,bhjv->bhdv", kbar_c, d)
        chunk_states.append(state)

    out = out[:, :, :S]
    coeff_all = coeff_all[:, :, :S]
    delta_all = delta_all[:, :, :S]
    return out, state, torch.stack(chunk_states, 2), delta_all, (coeff_all if return_coeff else None)


# --------------------------------------------------------------------------- #
# Layer + model
# --------------------------------------------------------------------------- #

class GatedDeltaNetBlock(ArchModule):
    """Pre-norm Gated DeltaNet block followed by a SwiGLU MLP."""

    AUX_KEYS = (
        "state/S", "state/S_norm", "state/delta_norm",
        "gate/alpha", "gate/alpha_channels", "gate/beta",
        "attn/probs_intra", "attn/coeff_intra",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.d_state = cfg.get("d_state", cfg["d_head"])
        self.d_head = cfg["d_head"]
        self.chunk = cfg.get("chunk", 8)
        self.negative = bool(cfg.get("negative_decay", False))
        self.decay_before_read = bool(cfg.get("decay_before_read", False))
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.w_q = nn.Linear(d, self.n_heads * self.d_state, bias=False)
        self.w_k = nn.Linear(d, self.n_heads * self.d_state, bias=False)
        self.w_v = nn.Linear(d, self.n_heads * self.d_head, bias=False)
        self.w_o = nn.Linear(self.n_heads * self.d_head, d, bias=False)
        self.w_alpha = nn.Linear(d, self.n_heads * self.d_state)
        self.w_beta = nn.Linear(d, self.n_heads)
        # Start with slow decay (~0.98 positive / ~0.96 signed) so the state
        # has time to accumulate structure inside a 32-token sequence.
        with torch.no_grad():
            self.w_alpha.bias.fill_(2.0 if self.negative else -4.0)
        self.mlp = SwiGLU(d, cfg["d_ff"])

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, _ = x.shape
        h = self.norm1(x)
        q = self.w_q(h).view(B, S, self.n_heads, self.d_state).transpose(1, 2)
        k = self.w_k(h).view(B, S, self.n_heads, self.d_state).transpose(1, 2)
        v = self.w_v(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        # Unit-norm keys are what make the delta rule a contraction: the
        # transition is I - beta k k^T with ||k||_2 = 1, so its eigenvalue
        # along k is 1 - beta in (0, 1) for beta in (0, 1).  Without this a
        # random key of norm sqrt(d_state) makes 1 - beta||k||^2 ~ -7 and the
        # state diverges (this is the stability condition beta||k||^2 < 2).
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        alpha = decay_from_logits(self.w_alpha(h), self.negative)
        alpha = alpha.view(B, S, self.n_heads, self.d_state).transpose(1, 2)
        beta = torch.sigmoid(self.w_beta(h)).transpose(1, 2)

        o, state, chunk_states, deltas, coeff = chunkwise_scan(
            q, k, v, alpha, beta, self.chunk, self.decay_before_read
        )
        o = o.transpose(1, 2).reshape(B, S, self.n_heads * self.d_head)
        y = self.w_o(o)
        x = x + y

        m = self.mlp(self.norm2(x))
        x = x + m

        probs = coeff.softmax(-1)
        aux = {
            "state/S": state,
            "state/S_norm": chunk_states.norm(dim=(-1, -2)),
            "state/delta_norm": deltas.norm(dim=-1),
            "gate/alpha": alpha.mean(-1),
            "gate/alpha_channels": alpha,
            "gate/beta": beta,
            "attn/probs_intra": probs,
            "attn/coeff_intra": coeff,
            "mlp/out": m,
        }
        return x, aux


@register("linear_attn")
class GatedDeltaNetLM(ArchModule):
    """Gated DeltaNet decoder: embedding -> N blocks -> RMSNorm -> tied logits."""

    AUX_KEYS = GatedDeltaNetBlock.AUX_KEYS + ("lm/logits",)

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(GatedDeltaNetBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.embed(tokens)
        aux: dict[str, torch.Tensor] = {"embed/out": x}
        for i, blk in enumerate(self.blocks):
            x, a = blk(x)
            for key, val in a.items():
                aux[f"layer{i}/{key}"] = val
        x = self.norm_f(x)
        logits = self.lm_head(x)
        aux["lm/logits"] = logits
        aux["hidden/final"] = x
        return logits, aux


def tiny_config(**over) -> ArchConfig:
    p = dict(
        d_model=64, n_heads=4, n_kv_heads=4, d_head=16, d_state=16,
        n_layers=2, d_ff=128, vocab=256, seq=32, chunk=8,
    )
    p.update(over)
    return ArchConfig("linear_attn-tiny", "linear_attn", p)


# --------------------------------------------------------------------------- #
# Self-check: the invariant that matters
# --------------------------------------------------------------------------- #

def selfcheck(chunk: int = 8, steps: int = 32, seed: int = 0, verbose: bool = True) -> dict[str, float]:
    """Assert recurrent == chunkwise for both gate signs and both decay orders."""
    torch.manual_seed(seed)
    B, H, dk, dv = 2, 4, 16, 16
    errs: dict[str, float] = {}
    for negative in (False, True):
        for before in (False, True):
            for length in (steps, steps - 3):  # also exercises the pad path
                q = torch.randn(B, H, length, dk)
                k = F.normalize(torch.randn(B, H, length, dk), dim=-1)
                v = torch.randn(B, H, length, dv)
                if negative:
                    alpha = torch.tanh(torch.randn(B, H, length, dk) + 2.0)
                else:
                    alpha = torch.exp(-F.softplus(torch.randn(B, H, length, dk) - 4.0))
                beta = torch.sigmoid(torch.randn(B, H, length))
                o1, s1, d1 = recurrent_scan(q, k, v, alpha, beta, before)
                o2, s2, cs2, d2, _ = chunkwise_scan(q, k, v, alpha, beta, chunk, before)
                key = f"{'negative' if negative else 'positive'}_{'before' if before else 'after'}_{length}"
                e = float((o1 - o2).abs().max())
                es = float((s1[:, :, -1] - s2).abs().max())
                ed = float((d1 - d2).abs().max())
                errs[f"{key}/out"] = e
                errs[f"{key}/state"] = es
                errs[f"{key}/delta"] = ed
                assert e < 2e-4, (key, "output", e)
                assert es < 2e-4, (key, "state", es)
                assert ed < 2e-4, (key, "delta", ed)
    if verbose:
        worst = max(errs.values())
        print(f"linear_attn selfcheck: worst abs error {worst:.3e} over {len(errs)} checks")
    return errs


@register_spec("linear_attn")
def linear_attn_spec() -> MathSpec:
    return MathSpec(
        name="linear_attn",
        family="linear_attention",
        summary="Gated DeltaNet: per-channel decayed linear attention with a delta-rule (online least-squares) state update, trained by an exact chunkwise form.",
        equations=[
            ("state", r"S_t \in \mathbb{R}^{d_k \times d_v},\quad o_t = S_t^\top q_t"),
            ("recurrence", r"S_t = \alpha_t \odot S_{t-1} + \beta_t\,k_t\,\big(v_t - S_{t-1}^\top k_t\big)^\top,\qquad o_t = S_t^\top q_t"),
            ("canonical", r"S_t = \alpha_t \odot S_{t-1} + \beta_t\,k_t\,\big(v_t - (\alpha_t \odot S_{t-1})^\top k_t\big)^\top"),
            ("gate_alpha", r"\alpha_t = \exp(-\mathrm{softplus}(a_t)) \in (0,1)^{d_k}\quad\text{or}\quad \alpha_t = \tanh(a_t) \in (-1,1)^{d_k}"),
            ("gate_beta", r"\beta_t = \sigma(b_t) \in (0,1)"),
            ("unrolled", r"S_t = \Big(\prod_{j=1}^{t} D_j\Big) S_0 + \sum_{j=1}^{t} \Big(\prod_{r=j+1}^{t} D_r\Big)\, \beta_j k_j \big(v_j - S_{j-1}^\top k_j\big)^\top,\quad D_j = \mathrm{diag}(\alpha_j)"),
            ("linear_map", r"S_t = A_t S_{t-1} + u_t,\quad A_t = D_t - \beta_t k_t k_t^\top,\quad u_t = \beta_t k_t v_t^\top"),
            ("least_squares", r"\mathcal{L}_t(S) = \tfrac12\|S^\top k_t - v_t\|_2^2,\qquad \nabla_S \mathcal{L}_t(S) = k_t (S^\top k_t - v_t)^\top"),
            ("delta_step", r"S_t = D_t S_{t-1} - \beta_t \nabla_S \mathcal{L}_t(S_{t-1}) = D_t S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top"),
            ("closed_form", r"S_t = \arg\min_{S} \tfrac12\|S^\top k_t - v_t\|^2 + \tfrac{1}{2\beta_t}(S - D_t S_{t-1})^\top (S - D_t S_{t-1})"),
            ("chunk_transform", r"T_t = \Delta_t^{-1} S_t,\quad \Delta_t = \prod_{j=1}^{t}\alpha_j,\quad T_t = T_{t-1} + \beta_t \bar k_t (v_t - T_{t-1}^\top x_t)^\top"),
            ("chunk_keys", r"\bar k_t = k_t / \Delta_t,\quad x_t = \Delta_{t-1} k_t\ \text{(contract) or } \Delta_t k_t\ \text{(canonical)}"),
            ("chunk_solve", r"(I + L)D = V - X S_0,\quad L_{ij} = \beta_j\, \bar k_j^\top x_i\ (j<i),\quad D_i = v_i - T_{i-1}^\top x_i"),
            ("chunk_output", r"O = \bar Q S_0 + \mathrm{tril}(C)\,D,\quad C_{ij} = \beta_j\, \bar k_j^\top \bar q_i,\quad \bar q_i = \Delta_i \odot q_i"),
            ("chunk_carry", r"S_C = \Delta_C \odot S_0 + \sum_{j=1}^{C} \beta_j\,(\Delta_C/\Delta_j)\odot k_j\,D_j^\top"),
            ("output", r"y_t = W_o\,\mathrm{concat}_h(o_t^{(h)}) + x_t,\qquad x_{t+1} = x_t + \mathrm{MLP}(\mathrm{RMSNorm}(y_t))"),
        ],
        assumptions=[
            "Causality: the state at t depends only on (k, v, alpha, beta) at or before t.",
            "Per-channel decay is diagonal in the key/state dimension; the write gate beta is a scalar per head so the transition stays D_t - beta_t k_t k_t^T (a rank-one update of a diagonal).",
            "The chunked form is an algebraic identity, not an approximation: it holds for any alpha (including negative eigenvalues) because the cumulative decay is tracked as sign * exp(sum log|alpha|).",
            "The delta rule is one SGD step on the per-token regression loss with a proximal/decay term; beta_t is the step size.",
            "No positional encoding is used: position enters only through the causal state and the decay product.",
        ],
        complexity={
            "params": "vocab*d_model + n_layers*(2*d_model*n_heads*d_state + 2*d_model*n_heads*d_head + d_model*n_heads*d_state + d_model*n_heads + 3*d_model*d_ff) + d_model",
            "time": "n_layers*(S*n_heads*(4*d_state*d_head + 4*d_state + 2*d_head) + 6*S*d_model*d_ff) + 2*S*d_model*vocab",
            "time_attention_equivalent": "n_layers*n_heads*S*S*d_head",
            "state_per_token": "n_layers*n_heads*d_state*d_head",
            "chunk_workspace": "n_layers*n_heads*chunk*chunk*d_state",
        },
        invariants=[
            "recurrent form equals chunkwise form to < 2e-4 max abs error",
            "state norm is finite and (for positive alpha, beta in (0,1)) non-expansive: ||S_t|| <= ||S_{t-1}|| + ||v_t||",
            "output shape equals input shape for every block",
            "token t's output depends only on tokens at or before t",
            "alpha in (0,1) for the positive parameterisation; |alpha| < 1 for the signed parameterisation",
            "the intra-chunk kernel is lower triangular (strictly causal)",
        ],
        references=[
            "Yang, Kautz, Hatamizadeh 2024 (arXiv:2412.06464) Gated Delta Networks",
            "Yang, Wang, Zhang, Shen, Kim 2024 (arXiv:2406.06484) Parallelizing DeltaNet",
            "Dao, Gu 2024 (arXiv:2405.21060) Transformers are SSMs (SSD)",
            "Park et al. 2026 (arXiv:2607.20062) Solar Open 2: gated delta rule with negative eigenvalues",
            "Steifer 2026 (arXiv:2605.16640) Shorter scratchpads in hybrid DeltaNet-attention decoders",
            "Ye, He, Liao, Wu, Lu 2026 (arXiv:2602.01763) Expressiveness hierarchy in hybrid linear-full attention",
            "Qwen Team 2025 Qwen3-Next (3:1 Gated DeltaNet : gated attention hybrid)",
        ],
    )
