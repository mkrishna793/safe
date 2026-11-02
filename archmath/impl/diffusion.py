"""Masked diffusion language model -- LLaDA-style discrete diffusion.

The architecture is *not* a chain rule.  It is a denoiser: a fully
bidirectional Transformer that maps a partially masked sequence ``x_t`` to a
distribution over the clean sequence ``x_0``.

    forward :  q_t(x_t | x_0) = prod_i [ t * delta(x_t^i, x_0^i)
                                         + (1-t) * delta(x_t^i, M) ]
    reverse :  p_theta(x_0 | x_t) = prod_i softmax(W h_theta(x_t)^i)_{x_0^i}
    loss    :  L = E_t E_{x_t} [ 1/(1-t) * sum_{i: x_t^i = M}
                                 -log p_theta(x_0^i | x_t) ]

``t`` is the *retention* probability (the fraction of tokens left clean); the
mask rate is ``1 - t``.  LLaDA writes the same objective with ``t_LLaDA``
equal to our ``1 - t``, so their weight ``1/t_LLaDA`` is our ``1/(1-t)``.

Two passes live in ``forward``:

* the training pass -- one random retention level, differentiable, returns the
  logits and the ELBO estimate that you should call ``.backward()`` on;
* the reverse pass  -- ``diffusion_steps`` iterative unmaskings under
  ``no_grad``, purely to expose the denoising trajectory to the tracer
  (``denoise/*_trace``).

Input is ``[B, S]`` long token ids in ``[0, vocab)``.  The mask token lives at
index ``vocab``, outside the vocabulary, so it can never be predicted.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm, rope


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    """SwiGLU(x) = (x W_gate . SiLU(x W_up)) W_down."""

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class BidirectionalAttention(nn.Module):
    """Multi-head attention with **no causal mask**.

    Every position may attend to every other position, including the ones it is
    about to denoise.  This is legal because the objective is the marginal
    likelihood of the whole sequence, not a left-to-right factorization: the
    target at position ``i`` is never part of the input ``x_t`` (it is either
    still masked or already revealed *by the forward process*, not by the
    model), so there is no leakage -- only context.
    """

    def __init__(self, d: int, n_heads: int, n_kv_heads: int, d_head: int, theta: float = 10_000.0) -> None:
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
        probs = scores.softmax(-1)                    # <- no mask, full square
        y = (probs @ v).transpose(1, 2).reshape(B, S, self.n_heads * self.d_head)
        mask = torch.ones(S, S, device=x.device, dtype=torch.bool)
        return self.w_o(y), {"attn/probs": probs, "attn/scores": scores, "attn/mask": mask}


class DiffusionBlock(ArchModule):
    AUX_KEYS = ("attn/probs", "mlp/out")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = RMSNorm(d)
        self.attn = BidirectionalAttention(d, cfg["n_heads"], cfg.get("n_kv_heads", cfg["n_heads"]), cfg["d_head"])
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, cfg["d_ff"])

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, aux = self.attn(self.norm1(x))
        x = x + a
        m = self.mlp(self.norm2(x))
        x = x + m
        aux["mlp/out"] = m
        return x, aux


@register("diffusion")
class MaskedDiffusionLM(ArchModule):
    """LLaDA-style masked diffusion LM over ``[B, S]`` token ids."""

    AUX_KEYS = (
        "attn/probs",
        "lm/logits",
        "denoise/mask",
        "denoise/confidence",
        "denoise/step",
        "denoise/elbo",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.vocab = cfg["vocab"]
        self.mask_id = cfg["vocab"]                    # extra embedding row
        self.n_steps = int(cfg.get("diffusion_steps", 8))
        self.schedule = cfg.get("mask_schedule", "uniform")
        self.embed = nn.Embedding(cfg["vocab"] + 1, d)
        self.blocks = nn.ModuleList(DiffusionBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = RMSNorm(d)
        # small init: at step 0 the denoiser is uniform over the vocabulary, so
        # the ELBO starts at its information-theoretic floor log(vocab).
        nn.init.normal_(self.embed.weight, std=0.02)

    # ------------------------------------------------------------------ #
    # core
    # ------------------------------------------------------------------ #
    def _run(self, x_t: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.embed(x_t)
        aux: dict[str, torch.Tensor] = {"embed/out": x}
        for i, blk in enumerate(self.blocks):
            x, a = blk(x)
            for k, v in a.items():
                aux[f"layer{i}/{k}"] = v
        aux["attn/probs"] = aux[f"layer{len(self.blocks) - 1}/attn/probs"]
        aux["attn/mask"] = aux[f"layer{len(self.blocks) - 1}/attn/mask"]
        x = self.norm_f(x)
        # tied head: rows 0..vocab-1 of the embedding matrix; the mask row is
        # never a prediction target and is sliced away.
        logits = F.linear(x, self.embed.weight[: self.vocab])
        aux["lm/logits"] = logits
        aux["hidden/final"] = x
        return logits, aux

    def _sample_t(self, batch: int, seq: int, t: float | torch.Tensor | None, device: torch.device) -> torch.Tensor:
        """Retention probability per sequence, always in (0, 1)."""
        if t is None:
            u = torch.rand(batch, device=device)
            if self.schedule == "cosine":                 # more mass near t=0/1
                t_b = torch.cos(0.5 * math.pi * u)
            else:
                t_b = 1.0 - u
        else:
            t_b = torch.as_tensor(t, device=device, dtype=torch.float32).expand(batch).clone()
        # keep at least ~one masked position per sequence and 1-t > 0
        return t_b.clamp(min=1.0 / (2 * seq), max=1.0 - 1.0 / seq)

    def _corrupt(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(tokens, dtype=torch.float32) >= t.unsqueeze(-1)
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            rows = empty.nonzero().squeeze(-1)
            cols = torch.randint(0, tokens.shape[1], (rows.numel(),), device=tokens.device)
            mask[rows, cols] = True
        return mask

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        tokens: torch.Tensor,
        t: float | torch.Tensor | None = None,
        steps: int | None = None,
        trajectory: bool = True,
        **kw,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tokens.dim() != 2:
            raise ValueError(f"diffusion expects [B, S] token ids, got {tuple(tokens.shape)}")
        B, S = tokens.shape
        device = tokens.device

        # ---- training pass: one random retention level ---------------- #
        t_b = self._sample_t(B, S, t, device)
        mask = self._corrupt(tokens, t_b)
        x_t = torch.where(mask, torch.full_like(tokens, self.mask_id), tokens)
        logits, aux = self._run(x_t)

        logp = logits.log_softmax(-1)
        nll = -logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)      # [B, S]
        weight = 1.0 / (1.0 - t_b)                                    # the ELBO weight
        elbo = (weight.unsqueeze(-1) * nll * mask.float()).sum() / (B * S)

        probs = logp.exp()
        conf = probs.max(-1).values
        step_idx = ((1.0 - t_b) * self.n_steps).floor().clamp(0, self.n_steps - 1).long()

        aux.update(
            {
                "denoise/t": t_b,
                "denoise/mask": mask,
                "denoise/mask_fraction": mask.float().mean(),
                "denoise/elbo": elbo,
                "denoise/elbo_unweighted": (nll * mask.float()).sum() / mask.float().sum().clamp_min(1.0),
                "denoise/confidence": conf * mask.float(),
                "denoise/pred": logits.detach().argmax(-1),
                "denoise/step": step_idx,
            }
        )

        # ---- reverse pass: confidence-ordered parallel unmasking ------- #
        if trajectory:
            sample, trace = self.sample(B, S, steps=steps)
            aux.update(trace)
            aux["denoise/sample"] = sample

        return logits, aux

    # ------------------------------------------------------------------ #
    # reverse / sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        batch: int,
        seq: int,
        steps: int | None = None,
        prompt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Iterative unmasking: k = ceil(masked / steps_left) tokens per step.

        At each step we evaluate p_theta(x_0 | x_t) once, take the ``k`` masked
        positions with the highest per-token confidence ``max_v p``, and commit
        their argmax.  This is the confidence-based (any-order) sampler: the
        order is chosen by the model, not fixed left-to-right.
        """
        steps = int(steps or self.n_steps)
        device = self.embed.weight.device
        x = torch.full((batch, seq), self.mask_id, dtype=torch.long, device=device)
        if prompt is not None:
            p = prompt.shape[1]
            x[:, :p] = prompt.to(device)

        conf_trace: list[torch.Tensor] = []
        mask_trace: list[torch.Tensor] = []
        step_conf: list[torch.Tensor] = []
        order = torch.full((batch, seq), steps, dtype=torch.long, device=device)
        rank = torch.arange(seq, device=device).unsqueeze(0).expand(batch, seq)

        for s in range(steps):
            logits, _ = self._run(x)
            probs = logits.softmax(-1)
            conf, pred = probs.max(-1)                       # [B, S]
            masked = x == self.mask_id
            score = conf.masked_fill(~masked, -1.0)

            left = masked.sum(dim=1)
            k_row = torch.minimum(
                torch.full_like(left, int(math.ceil(int(left.max()) / max(1, steps - s)))),
                left,
            )
            order_idx = score.argsort(dim=1, descending=True)
            take = (rank < k_row.unsqueeze(-1)) & masked.gather(1, order_idx)
            fresh = pred.gather(1, order_idx)

            # writes are indexed by rank, so they must be scattered back to
            # the sequence positions through order_idx
            x = x.scatter(1, order_idx, torch.where(take, fresh, x.gather(1, order_idx)))

            committed = torch.zeros_like(conf)
            committed.scatter_(1, order_idx, torch.where(take, conf.gather(1, order_idx), torch.zeros_like(conf)))
            conf_trace.append(committed)
            mask_trace.append(x == self.mask_id)
            step_conf.append(committed.sum() / take.sum().clamp_min(1))
            new_rank = torch.where(take, torch.full_like(order_idx, s), order.gather(1, order_idx))
            order.scatter_(1, order_idx, new_rank)

        trace = {
            "denoise/conf_trace": torch.stack(conf_trace),        # [steps, B, S]
            "denoise/mask_trace": torch.stack(mask_trace),        # [steps, B, S]
            "denoise/step_confidence": torch.stack(step_conf),    # [steps]
            "denoise/order": order,                               # [B, S] commit step
            "denoise/steps": torch.tensor(float(steps)),
        }
        return x, trace


def tiny_config(**over) -> ArchConfig:
    p = dict(
        d_model=64, n_heads=4, n_kv_heads=2, d_head=16, n_layers=2, d_ff=128,
        vocab=256, seq=32, diffusion_steps=8, mask_schedule="uniform",
    )
    p.update(over)
    return ArchConfig("diffusion-tiny", "diffusion", p)


@register_spec("diffusion")
def diffusion_spec() -> MathSpec:
    return MathSpec(
        name="diffusion",
        family="diffusion",
        summary="Masked (absorbing-state) discrete diffusion LM: a bidirectional denoiser trained on an ELBO with 1/(1-t) weighting.",
        equations=[
            ("forward_process",
             r"q_t(x_t \mid x_0) = \prod_{i=1}^{S} \big[\, t\,\delta(x_t^i, x_0^i) + (1-t)\,\delta(x_t^i, \mathrm{M}) \,\big],\quad t \in (0,1]"),
            ("mask_marginal",
             r"q_t(x_t^i = \mathrm{M} \mid x_0) = 1-t, \qquad q_t(x_t^i = x_0^i \mid x_0) = t"),
            ("reverse_model",
             r"p_\theta(x_0 \mid x_t) = \prod_{i=1}^{S} p_\theta(x_0^i \mid x_t),\qquad p_\theta(x_0^i = v \mid x_t) = \mathrm{softmax}_v\!\big(W h_\theta(x_t)^i\big)"),
            ("bidirectional_attention",
             r"\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_h})}{\sum_{j'=1}^{S}\exp(q_i \cdot k_{j'} / \sqrt{d_h})}\ \ \forall j \quad (\text{no causal mask})"),
            ("elbo",
             r"\mathcal{L}(\theta) = \mathbb{E}_{t\sim\mathcal{U}[0,1]}\ \mathbb{E}_{x_0}\ \mathbb{E}_{x_t\sim q_t(\cdot\mid x_0)} \left[ \frac{1}{1-t} \sum_{i:\,x_t^i=\mathrm{M}} -\log p_\theta(x_0^i \mid x_t) \right]"),
            ("likelihood_bound",
             r"\log p_\theta(x_0) \;\ge\; -\mathcal{L}(\theta)"),
            ("weight_cancels_mask_prob",
             r"\mathbb{E}_{x_t\sim q_t}\!\left[\frac{\mathbf{1}[x_t^i=\mathrm{M}]}{1-t}\right] = \frac{q_t(x_t^i=\mathrm{M}\mid x_0)}{1-t} = \frac{1-t}{1-t} = 1"),
            ("any_order_bound",
             r"-\mathcal{L}(\theta) = \mathbb{E}_{\pi}\!\left[\sum_{i=1}^{S}\log p_\theta\!\big(x_0^{\pi_i} \,\big|\, \{x_0^{\pi_j}\}_{j<i}\big)\right] \;\le\; \log p_\theta(x_0)"),
            ("confidence",
             r"c_i = \max_{v}\, p_\theta(x_i = v \mid x_t),\qquad \mathcal{U}_k = \operatorname{TopK}_{i\,:\,x_t^i=\mathrm{M}}\,(c_i)"),
            ("reverse_step",
             r"x^{(s+1)}_i = \begin{cases}\arg\max_v p_\theta(x_i=v\mid x^{(s)}) & i\in\mathcal{U}_k\\ x^{(s)}_i & \text{otherwise}\end{cases},\qquad k=\left\lceil \frac{|\{i:x^{(s)}_i=\mathrm{M}\}|}{\text{steps}-s}\right\rceil"),
        ],
        assumptions=[
            "The forward process is absorbing: a masked token never changes back, so q_t(x_t|x_0) is exact and needs no learned transition kernel.",
            "The reverse model factorizes over positions *conditionally on x_t*; it is not a chain rule over the sequence.",
            "All positions are conditionally independent given x_t (mean-field reverse), which is what makes parallel decoding exact under the model.",
            "The Transformer has no causal mask: the target x_0^i is never in the input, and the objective is the marginal likelihood of x_0.",
            "t is the retention probability; the mask rate is 1-t. LLaDA's t is our 1-t, so their 1/t weight is our 1/(1-t).",
        ],
        complexity={
            "params": "(vocab+1)*d_model + n_layers*(2*d_model*d_model + 2*d_model*n_kv_heads*d_head + 3*d_model*d_ff + 2*d_model) + d_model",
            "time_per_token": "n_layers*(2*d_model*d_model + 4*d_model*n_kv_heads*d_head + 4*d_model*seq + 6*d_model*d_ff) + 2*d_model*vocab",
            "attention_flops_per_token": "2*n_layers*n_heads*d_head*seq",
            "train_flops_per_sequence": "seq*(n_layers*(2*d_model*d_model + 4*d_model*n_kv_heads*d_head + 4*d_model*seq + 6*d_model*d_ff) + 2*d_model*vocab)",
            "sampling_forwards": "diffusion_steps",
            "sampling_flops_per_sequence": "diffusion_steps*seq*(n_layers*(2*d_model*d_model + 4*d_model*n_kv_heads*d_head + 4*d_model*seq + 6*d_model*d_ff) + 2*d_model*vocab)",
        },
        invariants=[
            "attention rows sum to 1 and the mask is all-ones (bidirectional)",
            "0 <= t <= 1 and the ELBO weight is 1/(1-t) > 0",
            "E[1/(1-t) * 1[masked]] = 1 for every position, so the loss is an unbiased per-token NLL estimate",
            "the mask token id (vocab) is outside the predicted vocabulary",
            "at initialization the per-token ELBO is approximately log(vocab)",
            "sampling unmasks every position after exactly `diffusion_steps` steps",
            "final sampled tokens are always in [0, vocab)",
        ],
        references=[
            "Nie et al. 2025 (arXiv:2502.09992) LLaDA",
            "Nie et al. 2026 (arXiv:2606.25331) iLLaDA",
            "Khanna et al. 2025 (arXiv:2506.17298) Mercury",
            "Gwak et al. 2026 (arXiv:2607.12829) dLLM inference survey",
            "Kim et al. 2026 (arXiv:2608.06628) block-hybrid attention retrofit",
        ],
    )
