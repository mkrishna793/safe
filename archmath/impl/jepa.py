"""JEPA: Joint-Embedding Predictive Architecture (latent world model)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm

# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class TinyEncoder(nn.Module):
    def __init__(self, vocab: int, d: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            for _ in range(n_layers)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for layer in self.layers:
            x = x + layer(x)
        return x  # [B,S,d]


class Predictor(nn.Module):
    def __init__(self, d: int, action_dim: int = 4) -> None:
        super().__init__()
        self.action = nn.Embedding(action_dim, d)
        self.net = nn.Sequential(
            nn.Linear(2 * d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)
        )

    def forward(self, s: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
        a = self.action(a_idx)  # [B,S,d] or [B,d]
        if a.dim() == 2:
            a = a.unsqueeze(1).expand_as(s)
        return self.net(torch.cat([s, a], dim=-1))


class JEPAWorld(ArchModule):
    AUX_KEYS = ("world/pred", "world/target", "world/loss", "world/latent_std")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        vocab = cfg["vocab"]
        n_layers = cfg["n_layers"]
        self.online = TinyEncoder(vocab, d, n_layers)
        self.target = TinyEncoder(vocab, d, n_layers)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(d, action_dim=cfg.get("action_dim", 4))
        self.ema = cfg.get("ema", 0.99)
        self.var_weight = cfg.get("var_weight", 1.0)

    def _ema_update(self) -> None:
        with torch.no_grad():
            for po, pt in zip(self.online.parameters(), self.target.parameters()):
                pt.data.mul_(self.ema).add_(po.data, alpha=1 - self.ema)

    def forward(self, tokens: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # tokens: [B, S] observations. Predict next latent under action = step index.
        B, S = tokens.shape
        s_online = self.online(tokens)  # [B,S,d]
        with torch.no_grad():
            s_target = self.target(tokens)  # [B,S,d]; treat as target for NEXT position

        # predictor takes s_t and predicts s_{t+1}; action = (t mod action_dim)
        actions = torch.arange(S, device=tokens.device) % self.predictor.action.num_embeddings
        actions = actions.unsqueeze(0).expand(B, S)
        # shift targets: target for position t is s_target[:, t+1]
        target = torch.roll(s_target, shifts=-1, dims=1)  # [B,S,d]

        pred = self.predictor(s_online, actions)  # [B,S,d]

        # variance regulariser on online latents (collapse guard)
        if s_online.shape[0] > 1:
            std = s_online.std(dim=0).mean()
        elif s_online.shape[1] > 1:
            std = s_online.std(dim=1).mean()
        else:
            std = torch.tensor(1.0, device=tokens.device, dtype=s_online.dtype)
        var_loss = F.relu(1.0 - std)

        pred_loss = F.mse_loss(pred, target.detach())
        loss = pred_loss + self.var_weight * var_loss

        if self.training:
            self._ema_update()

        aux = {
            "world/pred": pred,
            "world/target": target.detach(),
            "world/loss": loss,
            "world/latent_std": std.detach(),
        }
        # expose a pseudo-logits for the tracer/verify (project latent to vocab)
        # by returning pred norm as output proxy is not good; instead return pred
        # reshaped? verify expects tokens. Return pred as the "output" tensor.
        return pred, aux


@register("jepa")
class JEPATransformer(ArchModule):
    AUX_KEYS = ("world/pred", "world/target", "world/loss", "world/latent_std", "lm/logits")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.world = JEPAWorld(cfg)
        d = cfg["d_model"]
        self.head = nn.Linear(d, cfg["vocab"], bias=False)

    def forward(self, tokens: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred, aux = self.world(tokens)
        # expose a logits proxy so the tracer/verify can compute CE if asked
        logits = self.head(pred)
        aux["lm/logits"] = logits
        return logits, aux


def tiny_config(**over) -> ArchConfig:
    p = dict(d_model=64, n_heads=4, n_kv_heads=2, d_head=16, n_layers=2, d_ff=128,
             vocab=256, seq=16, action_dim=4, ema=0.99, var_weight=1.0)
    p.update(over)
    return ArchConfig("jepa-tiny", "jepa", p)


@register_spec("jepa")
def jepa_spec() -> MathSpec:
    return MathSpec(
        name="jepa",
        family="jepa",
        summary="JEPA: predict in latent space, not pixel/token space; EMA target encoder + stop-gradient + variance regulariser prevent collapse.",
        equations=[
            ("online", r"s_x = f_\theta(x)"),
            ("target", r"s_y = f_\xi(y),\quad \xi \leftarrow \tau\xi+(1-\tau)\theta\ \text{(EMA, stop-grad)}"),
            ("predictor", r"\hat s_y = g_\phi(s_x, a)"),
            ("loss", r"\mathcal{L}=\|\hat s_y-\mathrm{sg}(s_y)\|^2 + \lambda\,\mathrm{VarReg}(s_x)"),
            ("collapse", r"\text{constant }f_\theta\text{ achieves }\mathcal{L}=0 \Rightarrow \text{regularisation is necessary}"),
        ],
        assumptions=[
            "Predicting in latent space removes the pixel/token reconstruction bias of generative models.",
            "The EMA target is a slowly moving reference distribution; stop-gradient prevents the predictor from collapsing the online encoder.",
            "Variance (and covariance) regularisation keeps the latent representation informative.",
        ],
        complexity={
            "params": "d_model*d_model*n_layers*4 + d_model*d_model",
            "time_per_token": "d_model*d_model",
        },
        invariants=[
            "latent std > 0 (no collapse)",
            "EMA target changes slowly",
            "prediction loss finite",
        ],
        references=[
            "LeCun 2022 -- a path towards autonomous machine intelligence",
            "Assran et al. 2023 (arXiv:2301.08243) -- I-JEPA",
            "Bardes et al. 2024 (arXiv:2304.07193) -- V-JEPA",
        ],
    )


def make_world_sequence(steps: int = 32, vocab: int = 16, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    # a simple walk: token t = (t*3 + 7) mod vocab
    return torch.tensor([(t * 3 + 7) % vocab for t in range(steps)], dtype=torch.long).unsqueeze(0)


def jepa_demo(steps: int = 100, seq: int = 16, vocab: int = 16, seed: int = 0) -> dict[str, float]:
    torch.manual_seed(seed)
    cfg = tiny_config(d_model=32, n_layers=1, vocab=vocab, seq=seq, action_dim=4)
    model = JEPATransformer(cfg)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    stds = []
    for _ in range(steps):
        toks = make_world_sequence(seq, vocab, seed=seed).repeat(4, 1)[:, :seq]
        toks = toks.clamp(0, vocab - 1)
        opt.zero_grad()
        out, aux = model(toks)
        loss = aux["world/loss"]
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        stds.append(float(aux["world/latent_std"].detach()))
    return {"final_loss": losses[-1] if losses else float("nan"),
            "final_latent_std": stds[-1] if stds else float("nan"),
            "steps": float(steps)}
