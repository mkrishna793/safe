"""Byte Latent Transformer -- entropy-patched byte-level LM.

No tokenizer.  Bytes are grouped into *patches* whose boundaries are placed
where the next byte is surprising, and only the patches reach the global
Transformer:

    bytes  --entropy-->  boundaries  --pool-->  z_p  --global attn-->  u_p
      |                                                               |
      +------------------ local decoder (u_{p-1}, byte context) ------+--> logits

Three modules, exactly as in the paper:

* **local encoder**   ``h_i = CausalConv(Embed(x_i))``, pooled inside a patch
  to ``z_p = W_e * mean_{i in patch p} h_i``;
* **global latent Transformer** over the ``P`` patch latents, causal in patch
  order;
* **local decoder**   ``p(x_i | x_{<i}) = softmax(W_d [u_{p(i)-1} ; h^ctx_i])``.

The decoder reads the *previous* patch's global state, never the current one,
so the factorization stays autoregressive even though patch ``p``'s latent was
built from all of patch ``p``'s bytes.

Entropy patching is a stop-gradient, vectorised computation: a comparison, a
running maximum and a cumulative sum.  Nothing is looped over bytes.

Input is ``[B, N]`` long byte ids in ``[0, 256)``; the vocabulary has 260 rows
(256 bytes + BOS / MASK / EOS / PAD).
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
    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class PatchAttention(nn.Module):
    """Causal multi-head attention over the P patch latents (not over bytes)."""

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
        B, P, _ = x.shape
        q = self.w_q(x).view(B, P, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, P, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, P, self.n_kv_heads, self.d_head).transpose(1, 2)

        pos = torch.arange(P, device=x.device)
        q = rope(q, pos, self.theta)
        k = rope(k, pos, self.theta)
        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.ones(P, P, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, -1e9)
        probs = scores.softmax(-1)
        y = (probs @ v).transpose(1, 2).reshape(B, P, self.n_heads * self.d_head)
        return self.w_o(y), {"attn/probs": probs, "attn/scores": scores}


class PatchBlock(ArchModule):
    AUX_KEYS = ("attn/probs", "mlp/out")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = RMSNorm(d)
        self.attn = PatchAttention(d, cfg["n_heads"], cfg.get("n_kv_heads", cfg["n_heads"]), cfg["d_head"])
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, cfg["d_ff"])

    def forward(self, x: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, aux = self.attn(self.norm1(x))
        x = x + a
        m = self.mlp(self.norm2(x))
        x = x + m
        aux["mlp/out"] = m
        return x, aux


@register("blt")
class ByteLatentTransformer(ArchModule):
    """Entropy-patched byte-level LM over ``[B, N]`` byte ids."""

    AUX_KEYS = (
        "patch/boundaries",
        "patch/entropy",
        "patch/sizes",
        "patch/count",
        "latent/patch",
        "attn/probs",
        "lm/logits",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        dl = cfg.get("d_local", 32)
        k = int(cfg.get("local_kernel", 4))
        n_bytes = int(cfg.get("n_bytes", 256))

        self.vocab = cfg["vocab"]
        self.n_bytes = n_bytes
        self.patch_size_max = int(cfg.get("patch_size_max", 8))
        self.kappa = float(cfg.get("patch_threshold_kappa", 1.0))
        th = cfg.get("patch_threshold", None)
        self.fixed_threshold = None if th is None else float(th)

        self.d_local = dl
        self.kernel = k

        # ---- local (byte-level) stack -------------------------------- #
        self.byte_embed = nn.Embedding(cfg["vocab"], dl)
        self.local_conv = nn.Conv1d(dl, dl, k, groups=dl, bias=True)
        self.local_norm = RMSNorm(dl)
        self.entropy_head = nn.Linear(dl, n_bytes)

        # ---- global (patch-level) stack ------------------------------ #
        self.patch_enc = nn.Linear(dl, d)
        self.blocks = nn.ModuleList(PatchBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = RMSNorm(d)
        self.start_latent = nn.Parameter(torch.zeros(d))

        # ---- local decoder ------------------------------------------- #
        self.dec = nn.Sequential(
            nn.Linear(d + dl, cfg["d_ff"]),
            nn.GELU(),
            nn.Linear(cfg["d_ff"], n_bytes),
        )

        nn.init.normal_(self.byte_embed.weight, std=0.02)
        nn.init.normal_(self.start_latent, std=0.02)

    # ------------------------------------------------------------------ #
    # local stack
    # ------------------------------------------------------------------ #
    def _causal_conv(self, e: torch.Tensor) -> torch.Tensor:
        """Depthwise conv with left padding only: output i sees inputs <= i."""
        B, N, _ = e.shape
        h = F.conv1d(
            e.transpose(1, 2), self.local_conv.weight, self.local_conv.bias,
            padding=self.kernel - 1, groups=self.d_local,
        )
        return h[:, :, :N].transpose(1, 2)

    def _entropy(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h_i = H[p(x_i | x_{<i})], from a right-shifted causal conv."""
        B, N = tokens.shape
        bos = torch.full((B, 1), self.cfg.get("bos_id", 256), dtype=tokens.dtype, device=tokens.device)
        prev = torch.cat([bos, tokens[:, :-1]], dim=1)          # x_{<i}
        h_ctx = self.local_norm(self._causal_conv(self.byte_embed(prev)))
        logp = self.entropy_head(h_ctx).log_softmax(-1)         # p(x_i | x_{<i})
        entropy = -(logp.exp() * logp).sum(-1)                  # nats, [B, N]
        return h_ctx, logp, entropy

    # ------------------------------------------------------------------ #
    # entropy patching (stop-gradient, vectorised, no byte loop)
    # ------------------------------------------------------------------ #
    def _segment(self, entropy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = entropy.shape
        device = entropy.device

        if self.fixed_threshold is not None:
            theta = torch.tensor(self.fixed_threshold, device=device)
        else:  # global-per-batch threshold, mean + kappa * std of the entropy
            ent_std = entropy.std() if entropy.numel() > 1 else torch.zeros_like(entropy.mean())
            theta = (entropy.mean() + self.kappa * ent_std).detach()

        with torch.no_grad():
            cand = entropy.detach() > theta
            cand[:, 0] = True

            # running max of (position if candidate else 0) == start of the
            # current entropy-only segment, in one cummax
            idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
            start = torch.cummax(torch.where(cand, idx, torch.zeros_like(idx)), dim=1).values

            # split entropy segments that exceed the hard cap
            split = ((idx - start) % max(1, self.patch_size_max) == 0) & (idx > 0)
            boundaries = cand | split
            boundaries[:, 0] = True
        return theta, boundaries

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, tokens: torch.Tensor, **kw) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tokens.dim() != 2:
            raise ValueError(f"blt expects [B, N] byte ids, got {tuple(tokens.shape)}")
        B, N = tokens.shape
        device = tokens.device

        h_ctx, ent_logp, entropy = self._entropy(tokens)
        theta, boundaries = self._segment(entropy)

        pid = boundaries.long().cumsum(dim=1) - 1                       # [B, N]
        P = int(pid.max().item()) + 1

        # ---- local encoder: mean-pool byte states inside each patch ---- #
        h_enc = self.local_norm(self._causal_conv(self.byte_embed(tokens)))
        ones = torch.ones(B, N, device=device)
        count = torch.zeros(B, P, device=device).scatter_add_(1, pid, ones)
        pooled = torch.zeros(B, P, self.d_local, device=device).scatter_add_(
            1, pid.unsqueeze(-1).expand(-1, -1, self.d_local), h_enc
        )
        z = self.patch_enc(pooled / count.clamp_min(1.0).unsqueeze(-1))  # [B, P, d]

        # ---- global transformer: causal over patches, never over bytes - #
        u = z
        aux: dict[str, torch.Tensor] = {"latent/patch": z}
        for i, blk in enumerate(self.blocks):
            u, a = blk(u)
            for key, val in a.items():
                aux[f"layer{i}/{key}"] = val
        u = self.norm_f(u)
        last = len(self.blocks) - 1
        aux["attn/probs"] = aux[f"layer{last}/attn/probs"]

        # ---- local decoder: previous patch state + intra-prefix bytes --- #
        prev = (pid - 1).clamp(min=0)
        u_prev = torch.gather(u, 1, prev.unsqueeze(-1).expand(-1, -1, u.shape[-1]))
        u_prev = torch.where((pid == 0).unsqueeze(-1), self.start_latent.expand_as(u_prev), u_prev)
        logits = self.dec(torch.cat([u_prev, h_ctx], dim=-1))            # [B, N, 256]

        byte_tokens = tokens.clamp(0, self.n_bytes - 1)
        nll = -logits.log_softmax(-1).gather(-1, byte_tokens.unsqueeze(-1)).squeeze(-1)
        entropy_loss = -ent_logp.gather(-1, byte_tokens.unsqueeze(-1)).squeeze(-1).mean()
        sizes = torch.zeros(B, P, device=device, dtype=torch.long).scatter_add_(
            1, pid, torch.ones_like(pid)
        )

        aux.update(
            {
                "lm/logits": logits,
                "loss/entropy": entropy_loss,
                "byte/nll": nll.mean(),
                "byte/accuracy": (logits.detach().argmax(-1) == byte_tokens).float().mean(),
                "byte/entropy": entropy,
                "patch/entropy": entropy,
                "patch/logprobs": ent_logp,
                "patch/boundaries": boundaries,
                "patch/id": pid,
                "patch/sizes": sizes,
                "patch/count": sizes.gt(0).sum(dim=1),
                "patch/mean_length": (sizes.float().sum(1) / sizes.gt(0).sum(1).clamp_min(1)).mean(),
                "patch/threshold": theta.detach(),
                "patch/rate": sizes.gt(0).sum(1).float().mean() / N,
                "patch/theta_gap": (entropy.detach() - theta),
            }
        )
        return logits, aux

    # ------------------------------------------------------------------ #
    # greedy generation (fixed threshold: no whole-sequence statistics)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, n_new: int = 16) -> torch.Tensor:
        """Byte-by-byte sampling; patches are re-derived from the causal prefix."""
        if self.fixed_threshold is None:
            raise ValueError("generate() needs cfg['patch_threshold'] (a global constant theta)")
        x = prompt.clone()
        for _ in range(n_new):
            logits, _ = self.forward(x)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
        return x


def tiny_config(**over) -> ArchConfig:
    p = dict(
        d_model=64, n_heads=4, n_kv_heads=2, d_head=16, n_layers=2, d_ff=128,
        vocab=260, seq=32, n_bytes=256, d_local=32, local_kernel=4,
        patch_size_max=8, patch_threshold=None, patch_threshold_kappa=1.0,
        bos_id=256, mask_id=257, eos_id=258, pad_id=259,
    )
    p.update(over)
    return ArchConfig("blt-tiny", "blt", p)


@register_spec("blt")
def blt_spec() -> MathSpec:
    return MathSpec(
        name="blt",
        family="blt",
        summary="Byte-level LM with entropy-based dynamic patching: O(P^2) global attention over patches, O(N) local byte work.",
        equations=[
            ("local_context",
             r"g_i = \mathrm{CausalConv}\big(E[x_{<i}]\big)_i,\qquad p_\phi(x_i = v \mid x_{<i}) = \mathrm{softmax}_v\big(W_e\, g_i\big)"),
            ("patch_entropy",
             r"H_i = -\sum_{v=1}^{256} p_\phi(x_i = v \mid x_{<i})\,\log p_\phi(x_i = v \mid x_{<i})"),
            ("threshold",
             r"\theta = \mu_H + \kappa\,\sigma_H \quad (\text{adaptive})\qquad \text{or}\qquad \theta = \theta_0 \quad (\text{global constant})"),
            ("segmentation",
             r"b_i = \mathbf{1}[i=0] \;\lor\; \Big(\mathbf{1}[i>0] \land \big(H_i > \theta \;\lor\; (i - s_i) \equiv 0 \pmod{L_{\max}}\big)\Big),\quad s_i = \max\big(\{0\} \cup \{j \le i : H_j > \theta\}\big)"),
            ("patch_id",
             r"\pi_i = \sum_{j \le i} b_j - 1,\qquad \mathcal{P}_p = \{i : \pi_i = p\},\qquad P = \sum_i b_i"),
            ("patch_encoder",
             r"z_p = W_{\mathrm{enc}}\left(\frac{1}{|\mathcal{P}_p|}\sum_{i \in \mathcal{P}_p} h_i\right),\qquad h_i = \mathrm{CausalConv}(E[x_i])_i"),
            ("global_transformer",
             r"u_{1:P} = \mathrm{Transformer}_{\text{causal}}\big(z_{1:P}\big),\qquad \alpha_{pq} \propto \exp\!\big(q_p \cdot k_q / \sqrt{d_h}\big)\ \text{for } q \le p"),
            ("local_decoder",
             r"p_\theta(x_i \mid x_{<i}) = \mathrm{softmax}\big(W_d\,[\,u_{\pi_i - 1}\,;\, g_i\,]\big),\qquad u_{-1} = u_{\mathrm{start}}"),
            ("objective",
             r"\mathcal{L}(\theta) = -\sum_{i=1}^{N} \log p_\theta(x_i \mid x_{<i})"),
            ("compute_scaling",
             r"C \;=\; \underbrace{O(P^2 d + P d^2)}_{\text{global, over patches}} \;+\; \underbrace{O(N\, d_{\mathrm{local}}^2 + N\, d_{\mathrm{local}}\, d)}_{\text{local, over bytes}},\qquad P \approx N / \bar{L}"),
        ],
        assumptions=[
            "Patch boundaries are a stop-gradient discrete decision, like a tokenizer: no gradient flows through b_i.",
            "The local context model is causal (left-padded depthwise conv), so H_i is computable online from x_{<i}.",
            "The adaptive threshold uses batch statistics; generation must use a fixed global theta.",
            "The decoder reads u_{pi_i - 1}, the previous patch's state, so the factorization stays autoregressive.",
            "Byte vocabulary is 256; the 4 special ids (BOS/MASK/EOS/PAD) are never predicted.",
        ],
        complexity={
            "params": "vocab*d_local + local_kernel*d_local + d_local*n_bytes + d_local*d_model + n_layers*(2*d_model*d_model + 2*d_model*n_kv_heads*d_head + 3*d_model*d_ff + 2*d_model) + (d_model+d_local)*d_ff + d_ff*n_bytes + d_model",
            "patches": "ceil(seq/patch_size_max)",
            "local_flops_per_byte": "2*local_kernel*d_local*d_local + 2*d_local*n_bytes + 2*d_local*d_model + 2*(d_model+d_local)*d_ff + 2*d_ff*n_bytes",
            "global_flops_per_patch": "n_layers*(2*d_model*d_model + 4*d_model*n_kv_heads*d_head + 4*d_model*ceil(seq/patch_size_max) + 6*d_model*d_ff)",
            "total_flops": "seq*(2*local_kernel*d_local*d_local + 2*d_local*n_bytes + 2*d_local*d_model + 2*(d_model+d_local)*d_ff + 2*d_ff*n_bytes) + ceil(seq/patch_size_max)*n_layers*(2*d_model*d_model + 4*d_model*n_kv_heads*d_head + 4*d_model*ceil(seq/patch_size_max) + 6*d_model*d_ff)",
            "attention_flops_per_patch": "2*n_layers*n_heads*d_head*ceil(seq/patch_size_max)",
        },
        invariants=[
            "patch boundaries are monotone: patch ids are non-decreasing in i",
            "every patch has size in [1, patch_size_max]",
            "patch_count * mean_patch_length == seq",
            "the global attention matrix is P x P, not N x N",
            "logits at byte i depend only on bytes < i",
            "byte entropy is non-negative and bounded by log(256)",
        ],
        references=[
            "Pagnoni et al. 2024 (arXiv:2412.09871) Byte Latent Transformer",
            "Kallini et al. 2026 (arXiv:2605.08044) Fast Byte Latent Transformer",
        ],
    )
