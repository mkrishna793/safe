"""Mamba-2 style state-space model (SSD: state-space duality).

Full LM over token ids ``[B, S]``.  Each block is::

    u_t = conv1d_{width 4}(W_in LN(x))_t              # causal local mixer
    Delta_t = softplus(W_dt LN(x)_t + b_dt)            # per-head step size
    A_bar_t = exp(-Delta_t exp(A_log))                 # scalar per head, in (0,1)
    h_t = A_bar_t h_{t-1} + B_t x_t^T                  # h in R^{d_state x d_head}
    y_t = C_t^T h_t + D x_t
    out = W_out( (y_t * SiLU(W_z LN(x)_t))_heads )

``A_bar`` is a scalar times the identity in the state dimension: this is the
"scalar-identity" structure that makes the SSD chunked scan possible.  The
recurrence is implemented twice, naively and chunkwise, and the two are
asserted equal in :func:`selfcheck`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm


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


class CausalConv1d(nn.Module):
    """Depthwise causal convolution of width ``kernel`` (Mamba-2 uses 4)."""

    def __init__(self, channels: int, kernel: int = 4) -> None:
        super().__init__()
        self.kernel = kernel
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C] -> causal conv: left pad k-1, then trim.
        y = F.conv1d(x.transpose(1, 2), self.weight, self.bias, padding=self.kernel - 1, groups=x.shape[-1])
        return y[..., : x.shape[1]].transpose(1, 2)


# --------------------------------------------------------------------------- #
# The scan, twice
# --------------------------------------------------------------------------- #

def naive_scan(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """h_t = exp(dt_t A) h_{t-1} + B_t x_t^T, y_t = C_t^T h_t.

    Shapes: ``x`` -> ``[Bz,H,S,d_head]``, ``B,C`` -> ``[Bz,H,S,d_state]``,
    ``dt`` -> ``[Bz,H,S]``, ``A`` -> ``[H]`` (negative).  Returns
    ``(y, states, abar, h_final)`` with ``states`` ``[Bz,H,S,d_state,d_head]``.
    """
    Bz, H, S, ds = B.shape
    dv = x.shape[-1]
    h = x.new_zeros(Bz, H, ds, dv)
    ys: list[torch.Tensor] = []
    hs: list[torch.Tensor] = []
    abars: list[torch.Tensor] = []
    for t in range(S):
        abar = torch.exp(dt[:, :, t] * A.view(1, H))
        h = abar[:, :, None, None] * h + B[:, :, t].unsqueeze(-1) * x[:, :, t].unsqueeze(-2)
        ys.append(torch.einsum("bhd,bhdv->bhv", C[:, :, t], h))
        hs.append(h)
        abars.append(abar)
    return torch.stack(ys, 2), torch.stack(hs, 2), torch.stack(abars, 2), h


def chunked_scan(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    chunk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """SSD chunked scan: intra-chunk causal matrix + inter-chunk state carry.

    With ``G_i = sum_{r<=i} log A_bar_r`` inside a chunk::

        y_i = e^{G_i} C_i^T h_0 + sum_{j<=i} e^{G_i-G_j}(C_i.B_j) x_j
        h_C = e^{G_C} h_0 + sum_j e^{G_C-G_j} B_j x_j^T

    Returns ``(y, h_final, chunk_states, abar)`` with ``chunk_states``
    ``[Bz,H,n_chunks,d_state,d_head]``.
    """
    Bz, H, S, ds = B.shape
    dv = x.shape[-1]
    n_chunks = (S + chunk - 1) // chunk
    pad = n_chunks * chunk - S
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, pad))
        dt = F.pad(dt, (0, pad), value=0.0)  # softplus(0 * A) -> A_bar = 1

    h = x.new_zeros(Bz, H, ds, dv)
    y = x.new_zeros(Bz, H, n_chunks * chunk, dv)
    abars = torch.exp(dt * A.view(1, H, 1))              # [Bz,H,S]
    tril = torch.ones(chunk, chunk, device=x.device, dtype=torch.bool).tril()
    chunk_states: list[torch.Tensor] = []

    for c in range(n_chunks):
        sl = slice(c * chunk, (c + 1) * chunk)
        xc, Bc, Cc = x[:, :, sl], B[:, :, sl], C[:, :, sl]
        ab = abars[:, :, sl]
        G = ab.clamp_min(1e-12).log().cumsum(2)           # [Bz,H,C], <= 0
        eG = G.exp()

        inter = eG.unsqueeze(-1) * torch.einsum("bhcd,bhdv->bhcv", Cc, h)
        diff = G.unsqueeze(3) - G.unsqueeze(2)            # [i,j]
        decay = diff.exp() * tril[None, None, :, :]
        scores = torch.einsum("bhcd,bhjd->bhcj", Cc, Bc) * decay
        intra = torch.einsum("bhcj,bhjv->bhcv", scores, xc)
        y[:, :, sl] = inter + intra

        g_c = G[:, :, -1]
        Bscaled = (g_c.unsqueeze(-1) - G).exp().unsqueeze(-1) * Bc
        h = eG[:, :, -1].unsqueeze(-1).unsqueeze(-1) * h + torch.einsum("bhjd,bhjv->bhdv", Bscaled, xc)
        chunk_states.append(h)

    return y[:, :, :S], h, torch.stack(chunk_states, 2), abars[:, :, :S]


# --------------------------------------------------------------------------- #
# Block + model
# --------------------------------------------------------------------------- #

class Mamba2Block(ArchModule):
    AUX_KEYS = (
        "state/h", "state/h_norm", "state/A_bar",
        "gate/delta", "gate/z", "conv/out", "conv/out_norm",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.d_head = cfg["d_head"]
        self.d_state = cfg.get("d_state", 16)
        self.chunk = cfg.get("chunk", 8)
        self.kernel = int(cfg.get("conv_width", 4))
        self.d_inner = self.n_heads * self.d_head

        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.w_in = nn.Linear(d, self.d_inner, bias=False)
        self.conv = CausalConv1d(self.d_inner, self.kernel)
        self.w_z = nn.Linear(d, self.d_inner, bias=False)
        self.w_b = nn.Linear(d, self.n_heads * self.d_state, bias=False)
        self.w_c = nn.Linear(d, self.n_heads * self.d_state, bias=False)
        self.w_dt = nn.Linear(d, self.n_heads, bias=True)
        self.w_out = nn.Linear(self.d_inner, d, bias=False)
        # Mamba init: A = -(1 .. H), Delta ~ softplus(-1) so the fastest head
        # decays with A_bar ~ exp(-0.31 * H).  D is a learnable skip.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.n_heads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        with torch.no_grad():
            self.w_dt.bias.fill_(-1.0)
        self.mlp = SwiGLU(d, cfg["d_ff"])

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        Bz, S, _ = x.shape
        h_norm = self.norm1(x)
        u = self.w_in(h_norm)
        u = self.conv(u).view(Bz, S, self.n_heads, self.d_head).transpose(1, 2)
        z = self.w_z(h_norm).view(Bz, S, self.n_heads, self.d_head).transpose(1, 2)
        Bt = self.w_b(h_norm).view(Bz, S, self.n_heads, self.d_state).transpose(1, 2)
        Ct = self.w_c(h_norm).view(Bz, S, self.n_heads, self.d_state).transpose(1, 2)
        dt = F.softplus(self.w_dt(h_norm)).transpose(1, 2)   # [Bz,H,S]
        A = -torch.exp(self.A_log)                           # [H]

        y, h, chunk_states, abars = chunked_scan(u, Bt, Ct, dt, A, self.chunk)
        y = y + self.D.view(1, self.n_heads, 1, 1) * u
        y = y * F.silu(z)
        y = y.transpose(1, 2).reshape(Bz, S, self.d_inner)
        x = x + self.w_out(y)

        m = self.mlp(self.norm2(x))
        x = x + m

        aux = {
            "state/h": h,
            "state/h_norm": chunk_states.norm(dim=(-1, -2)),
            "state/A_bar": abars,
            "gate/delta": dt,
            "gate/z": z,
            "conv/out": u,
            "conv/out_norm": u.norm(dim=-1),
            "mlp/out": m,
        }
        return x, aux


@register("ssm")
class Mamba2LM(ArchModule):
    AUX_KEYS = Mamba2Block.AUX_KEYS + ("lm/logits",)

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(Mamba2Block(cfg) for _ in range(cfg["n_layers"]))
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
        n_layers=2, d_ff=128, vocab=256, seq=32, chunk=8, conv_width=4,
    )
    p.update(over)
    return ArchConfig("ssm-tiny", "ssm", p)


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #

def selfcheck(chunk: int = 8, steps: int = 32, seed: int = 0, verbose: bool = True) -> dict[str, float]:
    """Assert naive recurrence == chunked SSD scan."""
    torch.manual_seed(seed)
    Bz, H, ds, dv = 2, 4, 16, 16
    errs: dict[str, float] = {}
    for length in (steps, steps - 3):
        x = torch.randn(Bz, H, length, dv)
        Bt = torch.randn(Bz, H, length, ds)
        Ct = torch.randn(Bz, H, length, ds)
        dt = F.softplus(torch.randn(Bz, H, length) - 1.0)
        A = -torch.exp(torch.log(torch.arange(1, H + 1, dtype=torch.float32)))
        y1, s1, ab1, h1 = naive_scan(x, Bt, Ct, dt, A)
        y2, h2, cs2, ab2 = chunked_scan(x, Bt, Ct, dt, A, chunk)
        e = float((y1 - y2).abs().max())
        eh = float((h1 - h2).abs().max())
        ea = float((ab1 - ab2).abs().max())
        errs[f"{length}/y"] = e
        errs[f"{length}/state"] = eh
        errs[f"{length}/A_bar"] = ea
        assert e < 2e-5, ("y", length, e)
        assert eh < 2e-5, ("h", length, eh)
        assert ea < 1e-6, ("A_bar", length, ea)
    if verbose:
        print(f"ssm selfcheck: worst abs error {max(errs.values()):.3e}")
    return errs


@register_spec("ssm")
def ssm_spec() -> MathSpec:
    return MathSpec(
        name="ssm",
        family="state_space",
        summary="Mamba-2 SSD: selective diagonal SSM with scalar-identity transition, causal width-4 convolution, and an exact chunked scan.",
        equations=[
            ("conv", r"u_t = \sum_{r=0}^{3} w_r \odot x_{t-r}\quad(\text{depthwise, causal, width }4)"),
            ("delta", r"\Delta_t = \mathrm{softplus}(W_\Delta \mathrm{LN}(x)_t + b_\Delta) \in \mathbb{R}^{H}"),
            ("A_bar", r"\bar A_t = \exp(\Delta_t A),\qquad A = -\exp(A_{\log}) \in \mathbb{R}^{H}_{<0},\qquad \bar A_t \in (0,1)"),
            ("recurrence", r"h_t = \bar A_t \odot h_{t-1} + B_t\,u_t^\top,\qquad y_t = C_t^\top h_t + D\odot u_t"),
            ("shapes", r"h_t \in \mathbb{R}^{d_s \times d_v},\quad B_t,C_t \in \mathbb{R}^{d_s},\quad u_t \in \mathbb{R}^{d_v}"),
            ("unrolled", r"h_t = \Big(\prod_{j=1}^{t}\bar A_j\Big) h_0 + \sum_{j=1}^{t}\Big(\prod_{r=j+1}^{t}\bar A_r\Big) B_j u_j^\top"),
            ("linear_map", r"h_t = \bar A_t I\, h_{t-1} + B_t u_t^\top,\qquad y_t = (C_t^\top \otimes I)\,\mathrm{vec}(h_t)"),
            ("scalar_identity", r"\bar A_t = a_t I_{d_s}\ \Longrightarrow\ \prod_{r=j+1}^{t}\bar A_r = \exp\Big(\sum_{r=j+1}^{t}\Delta_r A\Big) I"),
            ("chunk_decay", r"G_i = \sum_{r=1}^{i}\log \bar A_r,\qquad M_{ij} = e^{G_i-G_j}\ \ (j\le i)"),
            ("chunk_output", r"y_i = e^{G_i}\,C_i^\top h_0 + \sum_{j\le i} e^{G_i-G_j}(C_i\cdot B_j)\,u_j"),
            ("chunk_carry", r"h_C = e^{G_C} h_0 + \sum_{j=1}^{C} e^{G_C-G_j} B_j u_j^\top"),
            ("block_matrix", r"Y = \mathrm{tril}\big((C B^\top)\odot M\big)U + \big(e^{G}\mathbf{1}^\top\big)\odot(C h_0),\quad h_C = e^{G_C}h_0 + B^\top \mathrm{diag}(e^{G_C-G})U"),
            ("output", r"y_t \leftarrow y_t \odot \mathrm{SiLU}(z_t),\qquad x_{t+1} = x_t + W_o y_t + \mathrm{MLP}(\mathrm{RMSNorm}(\cdot))"),
        ],
        assumptions=[
            "The transition is a scalar times the identity in the state dimension (Mamba-2's restriction of the diagonal SSM).  This is what turns the matrix scan into a scalar-decay times rank-one update.",
            "A < 0 and Delta > 0, hence A_bar in (0,1): the recurrence is a convex combination plus a write, so the state norm is non-expansive in the absence of writes.",
            "The convolution is causal: u_t depends on x_{t-3..t} only.",
            "B_t, C_t and Delta_t are input-dependent (selective), so the model is linear in the state but not in the input.",
        ],
        complexity={
            "params": "vocab*d_model + n_layers*(3*d_model*d_head*n_heads + 2*d_model*n_heads*d_state + d_model*n_heads + d_model*d_head*n_heads + 3*d_model*d_ff + n_heads*4 + n_heads) + d_model",
            "time": "n_layers*(S*n_heads*(4*d_state*d_head + 4*d_state + 2*d_head) + S*4*d_model*d_head*n_heads + 6*S*d_model*d_ff) + 2*S*d_model*vocab",
            "time_attention_equivalent": "n_layers*n_heads*S*S*d_head",
            "state_per_token": "n_layers*n_heads*d_state*d_head",
            "conv_state_per_token": "n_layers*3*n_heads*d_head",
            "chunk_workspace": "n_layers*n_heads*chunk*chunk",
        },
        invariants=[
            "naive recurrence equals chunked scan to < 2e-5 max abs error",
            "A_bar in (0,1) for every token and head",
            "state norm is finite and bounded by ||h_0|| + sum_t ||B_t|| ||u_t||",
            "output shape equals input shape for every block",
            "token t's output depends only on tokens at or before t",
            "convolution is causal (no future leakage)",
        ],
        references=[
            "Dao, Gu 2024 (arXiv:2405.21060) Transformers are SSMs: SSD",
            "Gu, Dao 2023 (arXiv:2312.00752) Mamba",
            "Yang, Kautz, Hatamizadeh 2024 (arXiv:2412.06464) Gated Delta Networks (Mamba-2 + delta rule)",
            "Qwen Team 2025 Qwen3-Next (hybrid linear-attention stack)",
        ],
    )
