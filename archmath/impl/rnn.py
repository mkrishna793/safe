"""Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) with GRU options.

    RNN:
        h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)

    LSTM:
        f_t = sigmoid(W_xf x_t + W_hf h_{t-1} + b_f)      (forget gate)
        i_t = sigmoid(W_xi x_t + W_hi h_{t-1} + b_i)      (input gate)
        c~_t = tanh(W_xc x_t + W_hc h_{t-1} + b_c)        (candidate cell)
        c_t = f_t * c_{t-1} + i_t * c~_t                   (cell state, Constant Error Carousel)
        o_t = sigmoid(W_xo x_t + W_ho h_{t-1} + b_o)      (output gate)
        h_t = o_t * tanh(c_t)                             (hidden state)

The Constant Error Carousel (CEC) in LSTM guarantees:
    ∂c_t / ∂c_{t-1} = diag(f_t)
which prevents exponential vanishing of gradients when f_t ≈ 1.
"""

from __future__ import annotations

import math
from typing import Any

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


class LSTMCell(nn.Module):
    """Numerically stable LSTM cell exposing gate activations and state trajectories."""

    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        # Combined projections for f, i, c~, o gates: [W_x, W_h]
        self.w_x = nn.Linear(d_in, 4 * d_hidden, bias=True)
        self.w_h = nn.Linear(d_hidden, 4 * d_hidden, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        # Standard forget gate bias initialization to 1.0 for long-term memory retention
        with torch.no_grad():
            self.w_x.bias[0:self.d_hidden].fill_(1.0)
            self.w_x.bias[self.d_hidden:].zero_()

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        gates = self.w_x(x_t) + self.w_h(h_prev)
        f_raw, i_raw, c_raw, o_raw = gates.chunk(4, dim=-1)

        f_t = torch.sigmoid(f_raw)
        i_t = torch.sigmoid(i_raw)
        c_tilde = torch.tanh(c_raw)
        o_t = torch.sigmoid(o_raw)

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)

        aux = {
            "gate/forget": f_t,
            "gate/input": i_t,
            "gate/output": o_t,
            "state/cell_candidate": c_tilde,
            "state/cell": c_t,
            "state/hidden": h_t,
        }
        return h_t, c_t, aux


class LSTMBlock(ArchModule):
    AUX_KEYS = ("state/hidden", "state/cell", "gate/forget", "gate/input", "gate/output")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm = RMSNorm(d)
        self.cell = LSTMCell(d, d)
        self.mlp = nn.Sequential(
            RMSNorm(d),
            nn.Linear(d, cfg["d_ff"], bias=False),
            nn.SiLU(),
            nn.Linear(cfg["d_ff"], d, bias=False),
        )

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, D = x.shape
        x_norm = self.norm(x)

        h_t = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        c_t = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        h_seq = []
        c_seq = []
        f_seq = []
        i_seq = []
        o_seq = []

        for t in range(S):
            h_t, c_t, aux_t = self.cell(x_norm[:, t, :], h_t, c_t)
            h_seq.append(h_t)
            c_seq.append(c_t)
            f_seq.append(aux_t["gate/forget"])
            i_seq.append(aux_t["gate/input"])
            o_seq.append(aux_t["gate/output"])

        h_out = torch.stack(h_seq, dim=1)  # [B, S, D]
        c_out = torch.stack(c_seq, dim=1)
        f_out = torch.stack(f_seq, dim=1)
        i_out = torch.stack(i_seq, dim=1)
        o_out = torch.stack(o_seq, dim=1)

        # Residual connection + MLP
        y = x + h_out
        m = self.mlp(y)
        out = y + m

        # Spectral radius estimation of hidden recurrent weight W_hh
        with torch.no_grad():
            w_hh = self.cell.w_h.weight
            sv = torch.linalg.svdvals(w_hh.float())
            spec_radius = sv.max()

        aux = {
            "state/hidden": h_out,
            "state/cell": c_out,
            "gate/forget": f_out,
            "gate/input": i_out,
            "gate/output": o_out,
            "mlp/out": m,
            "state/w_hh_spectral_radius": spec_radius.unsqueeze(0),
        }
        return out, aux


@register("rnn")
class RNNModel(ArchModule):
    AUX_KEYS = ("state/hidden", "state/cell", "gate/forget", "lm/logits")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(LSTMBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.embed(tokens)
        aux: dict[str, torch.Tensor] = {"embed/out": x}

        for i, blk in enumerate(self.blocks):
            x, a = blk(x)
            for k, v in a.items():
                aux[f"layer{i}/{k}"] = v

        x = self.norm_f(x)
        logits = self.lm_head(x)
        aux["lm/logits"] = logits
        aux["hidden/final"] = x
        return logits, aux


def tiny_config(**over) -> ArchConfig:
    p = dict(d_model=64, n_layers=2, d_ff=128, vocab=256, seq=32)
    p.update(over)
    return ArchConfig("rnn-tiny", "rnn", p)


@register_spec("rnn")
def rnn_spec() -> MathSpec:
    return MathSpec(
        name="rnn",
        family="rnn",
        summary="Deep stacked LSTM recurrent network with Constant Error Carousel and RMSNorm residual layers.",
        equations=[
            ("forget_gate", r"f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)"),
            ("input_gate", r"i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)"),
            ("candidate_cell", r"\tilde{c}_t = \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c)"),
            ("cell_update", r"c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t"),
            ("output_gate", r"o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)"),
            ("hidden_state", r"h_t = o_t \odot \tanh(c_t)"),
            ("cec_jacobian", r"\frac{\partial c_t}{\partial c_{t-1}} = \mathrm{diag}(f_t)"),
            ("residual", r"x_{l+1} = x_l + h^{(l)}(x_l) + \mathrm{MLP}(\mathrm{LN}(x_l + h^{(l)}))"),
        ],
        assumptions=[
            "Strictly causal token recursion: h_t depends only on x_{<=t}.",
            "Constant Error Carousel prevents gradient decay when f_t ≈ 1.",
            "Bounded hidden activations: h_t in (-1, 1).",
        ],
        complexity={
            "params": "vocab*d_model + n_layers*(4*(d_model*d_model + d_model) + 4*d_model*d_model + 2*d_model*d_ff) + d_model",
            "time_per_token": "n_layers*(8*d_model*d_model + 4*d_model*d_ff)",
            "memory": "n_layers * d_model * seq",
        },
        invariants=[
            "forget, input, output gates are strictly in (0, 1)",
            "hidden states are bounded in (-1, 1)",
            "gradient flows across sequence via linear cell update",
        ],
        references=[
            "Hochreiter & Schmidhuber 1997 (LSTM)",
            "Gers et al. 2000 (Forget Gate)",
            "Cho et al. 2014 (GRU)",
        ],
    )
