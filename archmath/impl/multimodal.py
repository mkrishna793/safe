"""Native multimodal early fusion, plus a late-fusion baseline.

The model consumes a *single* token stream

    [ image_1 ... image_P | audio_1 ... audio_A | text_1 ... text_T ]

where every position lives in the same residual width ``d_model``.  Each
modality has its own embedder, its own learned position table, and a shared
modality-type embedding.  A single Transformer then runs over the concatenated
sequence.  The attention mask is modality aware:

    image  -> image   (full within the image prefix)
    audio  -> audio   (full within the audio prefix)
    text   -> image, audio, and causal text

so image and audio cannot see each other before text arrives.  This is the
architectural definition of *early fusion*: the modalities meet inside the
residual stream, not at an output head.

The training objective is modality balanced.  With ``N_m`` tokens of modality
``m`` and ``M`` modalities, token ``t`` of modality ``m`` carries weight

    w_t = 1 / (M * N_m),          sum_t w_t = 1.

The same file exports ``LateFusionMultimodal`` and ``projector_overhead`` so
the early-fusion claim can be measured against the usual encoder + projector +
frozen-language-backbone baseline.  The late projector is a rank-``r`` linear
bottleneck: its image has dimension at most ``r``, independently of how many
patch tokens the encoder produced.

``forward`` accepts integer ``tokens`` of shape ``[B, T]``.  Image patches and
audio frames may be supplied as ``images`` ``[B, P, patch_dim]`` and ``audio``
``[B, A, audio_dim]``.  When they are omitted, deterministic pseudo-patches are
derived from the text token ids so that the block remains runnable from a single
tensor (the CLI and tracer do exactly this).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec, rms_norm

MOD_IMAGE, MOD_AUDIO, MOD_TEXT = 0, 1, 2
MODALITY_NAMES = ("image", "audio", "text")


# --------------------------------------------------------------------------- #
# Small shared pieces (kept local: this file is a drop-in architecture)
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    """SwiGLU(x) = (x W_gate ⊙ SiLU(x W_up)) W_down."""

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ModalityAwareAttention(nn.Module):
    """GQA attention with an arbitrary boolean mask ``[S, S]``."""

    def __init__(self, d: int, n_heads: int, n_kv_heads: int, d_head: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.group = n_heads // n_kv_heads
        self.w_q = nn.Linear(d, n_heads * d_head, bias=False)
        self.w_k = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.w_v = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.w_o = nn.Linear(n_heads * d_head, d, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape
        q = self.w_q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # mask is [S, S]; every row contains its diagonal, so softmax is finite.
        scores = scores.masked_fill(~mask.view(1, 1, S, S), -1e9)
        probs = scores.softmax(-1)
        y = (probs @ v).transpose(1, 2).reshape(B, S, self.n_heads * self.d_head)
        return self.w_o(y), probs


class ModalityMoE(nn.Module):
    """SwiGLU experts with a modality-conditioned router.

    The learned ``mod_bias`` is the architectural nudge that lets experts
    specialize by modality: the router sees the token state *and* which
    modality produced it.  This is the tiny analogue of the modality-specific
    experts observed in native multimodal MoE scaling studies.
    """

    def __init__(self, d: int, d_ff: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.router = nn.Linear(d, n_experts, bias=False)
        self.mod_bias = nn.Parameter(torch.zeros(3, n_experts))
        self.experts = nn.ModuleList(SwiGLU(d, d_ff) for _ in range(n_experts))

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.router(x) + self.mod_bias[modality_ids]
        probs = logits.softmax(-1)
        top = probs.topk(self.top_k, dim=-1)

        # Dense evaluation of all experts keeps the tiny model simple and
        # differentiable; the gather below is the actual sparse route.
        outs = torch.stack([expert(x) for expert in self.experts], dim=-2)  # [B,S,E,d]
        weights = torch.zeros_like(probs).scatter(-1, top.indices, top.values)
        y = (outs * weights.unsqueeze(-1)).sum(dim=-2)

        flat = top.indices.reshape(-1)
        load = torch.bincount(flat, minlength=self.n_experts).float() / flat.numel()
        ent = -(probs.clamp_min(1e-9).log() * probs).sum(-1).mean()
        return y, {
            "route/logits": logits,
            "route/expert_ids": top.indices,
            "route/load": load,
            "route/entropy": ent,
        }


class EarlyFusionBlock(ArchModule):
    AUX_KEYS = ("attn/probs", "route/expert_ids", "route/load", "mlp/out")

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = RMSNorm(d)
        self.attn = ModalityAwareAttention(
            d,
            cfg["n_heads"],
            cfg.get("n_kv_heads", cfg["n_heads"]),
            cfg["d_head"],
        )
        self.norm2 = RMSNorm(d)
        self.moe = ModalityMoE(
            d,
            cfg["d_ff"],
            cfg.get("n_experts", 4),
            cfg.get("top_k", 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, probs = self.attn(self.norm1(x), modality_mask)
        x = x + a
        m, route = self.moe(self.norm2(x), modality_ids)
        x = x + m
        aux: dict[str, torch.Tensor] = {"attn/probs": probs, "mlp/out": m}
        aux.update(route)
        return x, aux


# --------------------------------------------------------------------------- #
# Mask algebra
# --------------------------------------------------------------------------- #

def build_modality_mask(modality_ids: torch.Tensor, intra_causal: bool = False) -> torch.Tensor:
    """Boolean ``[S, S]`` mask for the early-fusion stream.

    ``allow[i, j]`` is true iff query ``i`` may read key ``j``:

      * image/image and audio/audio: full (a prefix is encoded bidirectionally)
      * text/text: causal, or full when ``intra_causal`` is false
      * text/non-text: all prefix tokens (they always precede text)
      * image/audio and audio/image: never

    ``intra_causal=True`` makes the prefix blocks lower triangular, which is the
    strict autoregressive variant.  The default is the prefix-encoder variant.
    """
    if modality_ids.dim() == 2:
        modality_ids = modality_ids[0]
    m = modality_ids
    S = m.numel()
    device = m.device
    same = m[:, None] == m[None, :]
    is_text_q = (m == MOD_TEXT)[:, None]
    is_text_k = (m == MOD_TEXT)[None, :]
    causal = torch.ones(S, S, device=device, dtype=torch.bool).tril()

    prefix_same = (~is_text_q) & same
    text_text = is_text_q & is_text_k
    text_prefix = is_text_q & (~is_text_k)

    if intra_causal:
        allowed = (prefix_same & causal) | (text_text & causal) | text_prefix
    else:
        allowed = prefix_same | (text_text & causal) | text_prefix
    return allowed


# --------------------------------------------------------------------------- #
# Native early-fusion model
# --------------------------------------------------------------------------- #

@register("multimodal")
class NativeMultimodal(ArchModule):
    AUX_KEYS = (
        "latent/modality_ids",
        "attn/modality_mask",
        "attn/probs",
        "loss/per_modality",
        "loss/balanced",
        "route/expert_ids",
        "route/load",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.d = d
        self.vocab = cfg["vocab"]
        self.n_patches = cfg.get("vision_patches", 8)
        self.n_frames = cfg.get("audio_frames", 8)
        self.patch_dim = cfg.get("patch_dim", 32)
        self.audio_dim = cfg.get("audio_dim", 32)

        self.text_embed = nn.Embedding(self.vocab, d)
        self.image_embed = nn.Linear(self.patch_dim, d, bias=False)
        self.audio_embed = nn.Linear(self.audio_dim, d, bias=False)
        self.modality_embed = nn.Embedding(3, d)
        self.image_pos = nn.Embedding(self.n_patches, d)
        self.audio_pos = nn.Embedding(self.n_frames, d)
        max_seq = max(int(cfg.get("seq", 32)), 2048)
        self.text_pos = nn.Embedding(max_seq, d)

        self.blocks = nn.ModuleList(EarlyFusionBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = RMSNorm(d)

        self.lm_head = nn.Linear(d, self.vocab, bias=False)
        self.lm_head.weight = self.text_embed.weight  # tied text vocabulary
        self.image_head = nn.Linear(d, self.patch_dim, bias=False)
        self.audio_head = nn.Linear(d, self.audio_dim, bias=False)

    # ------------------------------------------------------------------ #
    def _pseudo_images(self, tokens: torch.Tensor) -> torch.Tensor:
        B, _ = tokens.shape
        t = torch.arange(self.n_patches, device=tokens.device, dtype=torch.float32)
        dim = torch.arange(self.patch_dim, device=tokens.device, dtype=torch.float32)
        pat = torch.sin(t.unsqueeze(-1) * 0.1 + dim.unsqueeze(0) * 0.05)
        return pat.unsqueeze(0).expand(B, self.n_patches, self.patch_dim)

    def _pseudo_audio(self, tokens: torch.Tensor) -> torch.Tensor:
        B, _ = tokens.shape
        t = torch.arange(self.n_frames, device=tokens.device, dtype=torch.float32)
        dim = torch.arange(self.audio_dim, device=tokens.device, dtype=torch.float32)
        pat = torch.cos(t.unsqueeze(-1) * 0.1 + dim.unsqueeze(0) * 0.05)
        return pat.unsqueeze(0).expand(B, self.n_frames, self.audio_dim)

    def _prepare_images(self, tokens: torch.Tensor, images: torch.Tensor | None) -> torch.Tensor:
        if images is None:
            return self._pseudo_images(tokens)
        if images.dim() == 2:
            images = images.unsqueeze(-1).expand(-1, -1, self.patch_dim)
        if images.shape[1] != self.n_patches or images.shape[-1] != self.patch_dim:
            raise ValueError(
                f"images must be [B,{self.n_patches},{self.patch_dim}], got {tuple(images.shape)}"
            )
        return images

    def _prepare_audio(self, tokens: torch.Tensor, audio: torch.Tensor | None) -> torch.Tensor:
        if audio is None:
            return self._pseudo_audio(tokens)
        if audio.dim() == 2:
            audio = audio.unsqueeze(-1).expand(-1, -1, self.audio_dim)
        if audio.shape[1] != self.n_frames or audio.shape[-1] != self.audio_dim:
            raise ValueError(
                f"audio must be [B,{self.n_frames},{self.audio_dim}], got {tuple(audio.shape)}"
            )
        return audio

    # ------------------------------------------------------------------ #
    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        **kw,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``tokens``: ``[B, T]`` long.  Returns text logits ``[B, T, vocab]``."""
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [B, T], got {tuple(tokens.shape)}")
        B, T = tokens.shape
        P, A = self.n_patches, self.n_frames
        dev = tokens.device

        images = self._prepare_images(tokens, images).to(dtype=self.image_embed.weight.dtype)
        audio = self._prepare_audio(tokens, audio).to(dtype=self.audio_embed.weight.dtype)

        img = self.image_embed(images)
        img = img + self.modality_embed(torch.full((B, P), MOD_IMAGE, device=dev, dtype=torch.long))
        img = img + self.image_pos(torch.arange(P, device=dev))

        aud = self.audio_embed(audio)
        aud = aud + self.modality_embed(torch.full((B, A), MOD_AUDIO, device=dev, dtype=torch.long))
        aud = aud + self.audio_pos(torch.arange(A, device=dev))

        txt = self.text_embed(tokens)
        txt = txt + self.modality_embed(torch.full((B, T), MOD_TEXT, device=dev, dtype=torch.long))
        txt = txt + self.text_pos(torch.arange(T, device=dev))

        x = torch.cat([img, aud, txt], dim=1)  # [B, P+A+T, d]
        modality_ids = torch.cat(
            [
                torch.full((B, P), MOD_IMAGE, device=dev, dtype=torch.long),
                torch.full((B, A), MOD_AUDIO, device=dev, dtype=torch.long),
                torch.full((B, T), MOD_TEXT, device=dev, dtype=torch.long),
            ],
            dim=1,
        )
        modality_mask = build_modality_mask(modality_ids)

        aux: dict[str, torch.Tensor] = {
            "embed/out": x,
            "latent/modality_ids": modality_ids,
            "attn/modality_mask": modality_mask,
        }
        for i, blk in enumerate(self.blocks):
            x, a = blk(x, modality_ids, modality_mask)
            for key, value in a.items():
                aux[f"layer{i}/{key}"] = value

        h = self.norm_f(x)
        text_h = h[:, P + A :, :]
        logits = self.lm_head(text_h)

        # ---- modality-balanced loss ---------------------------------- #
        # Each modality contributes its own mean; equivalently every token is
        # weighted by 1/(M * N_m), so sum_t w_t = 1.
        image_mse = (self.image_head(h[:, :P]) - images.detach()).pow(2).mean(-1)   # [B,P]
        audio_mse = (self.audio_head(h[:, P : P + A]) - audio.detach()).pow(2).mean(-1)  # [B,A]
        text_ce = F.cross_entropy(
            logits.reshape(-1, self.vocab), tokens.reshape(-1), reduction="none"
        ).view(B, T)

        counts = torch.tensor([float(P), float(A), float(T)], device=dev)
        weights = 1.0 / (3.0 * counts)
        balanced = (
            image_mse.sum() * weights[0]
            + audio_mse.sum() * weights[1]
            + text_ce.sum() * weights[2]
        )

        aux["lm/logits"] = logits
        aux["latent/fused"] = h
        aux["loss/per_modality"] = torch.stack(
            [image_mse.mean(), audio_mse.mean(), text_ce.mean()]
        )
        aux["loss/per_token"] = torch.cat([image_mse, audio_mse, text_ce], dim=1)
        aux["loss/weights"] = weights
        aux["loss/balanced"] = balanced
        aux["modality/token_counts"] = counts
        aux["patch/image_span"] = torch.tensor([0, P], device=dev)
        aux["patch/audio_span"] = torch.tensor([P, P + A], device=dev)
        return logits, aux


# --------------------------------------------------------------------------- #
# Late-fusion baseline
# --------------------------------------------------------------------------- #

class LateFusionProjector(nn.Module):
    """Rank-``r`` bottleneck: R^d -> R^r -> R^d.

    The intermediate representation has at most ``r`` dimensions.  Whatever the
    image encoder knew, only an ``r``-dimensional shadow of it reaches the
    language backbone.
    """

    def __init__(self, d_in: int, d_out: int, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, d_out, bias=False)
        self.norm = RMSNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.up(self.down(x)))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _PlainBlock(nn.Module):
    """Pre-norm attention + SwiGLU without a router (late-fusion baseline)."""

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = RMSNorm(d)
        self.attn = ModalityAwareAttention(
            d, cfg["n_heads"], cfg.get("n_kv_heads", cfg["n_heads"]), cfg["d_head"]
        )
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, cfg["d_ff"])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, probs = self.attn(self.norm1(x), mask)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, probs


class LateFusionMultimodal(ArchModule):
    """Encoder -> rank-r projector -> text decoder.

    Each modality is encoded by its own small stack.  Image and audio are
    *pooled to a single vector*, projected through a rank-``r`` linear map, and
    prepended to the text sequence.  The text decoder never sees patch tokens:
    this is the capacity bottleneck the early-fusion model removes.
    """

    AUX_KEYS = (
        "latent/modality_ids",
        "attn/modality_mask",
        "loss/per_modality",
        "loss/balanced",
        "projector/rank",
    )

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.d = d
        self.vocab = cfg["vocab"]
        self.n_patches = cfg.get("vision_patches", 8)
        self.n_frames = cfg.get("audio_frames", 8)
        self.patch_dim = cfg.get("patch_dim", 32)
        self.audio_dim = cfg.get("audio_dim", 32)
        rank = cfg.get("projector_rank", max(1, d // 4))

        self.text_embed = nn.Embedding(self.vocab, d)
        self.image_embed = nn.Linear(self.patch_dim, d, bias=False)
        self.audio_embed = nn.Linear(self.audio_dim, d, bias=False)
        self.modality_embed = nn.Embedding(3, d)
        self.text_pos = nn.Embedding(max(int(cfg.get("seq", 32)), 2048), d)

        enc_layers = cfg.get("late_encoder_layers", 1)
        self.image_encoder = nn.ModuleList(_PlainBlock(cfg) for _ in range(enc_layers))
        self.audio_encoder = nn.ModuleList(_PlainBlock(cfg) for _ in range(enc_layers))

        self.image_projector = LateFusionProjector(d, d, rank)
        self.audio_projector = LateFusionProjector(d, d, rank)

        self.decoder = nn.ModuleList(_PlainBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, self.vocab, bias=False)
        self.lm_head.weight = self.text_embed.weight
        self.image_head = nn.Linear(d, self.patch_dim, bias=False)
        self.audio_head = nn.Linear(d, self.audio_dim, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        **kw,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [B, T], got {tuple(tokens.shape)}")
        B, T = tokens.shape
        P, A = self.n_patches, self.n_frames
        dev = tokens.device

        if images is None:
            idx = torch.arange(P, device=dev) % T
            images = (tokens[:, idx].float() / max(self.vocab, 1)).unsqueeze(-1).expand(B, P, self.patch_dim)
        if audio is None:
            idx = (torch.arange(A, device=dev) * 3 + 1) % T
            audio = (tokens[:, idx].float() / max(self.vocab, 1)).unsqueeze(-1).expand(B, A, self.audio_dim)
        images = images.to(dtype=self.image_embed.weight.dtype)
        audio = audio.to(dtype=self.audio_embed.weight.dtype)

        full = torch.ones(P, P, device=dev, dtype=torch.bool)
        img = self.image_embed(images)
        for blk in self.image_encoder:
            img, _ = blk(img, full)
        img_summary = self.image_projector(img.mean(dim=1))

        full_a = torch.ones(A, A, device=dev, dtype=torch.bool)
        aud = self.audio_embed(audio)
        for blk in self.audio_encoder:
            aud, _ = blk(aud, full_a)
        aud_summary = self.audio_projector(aud.mean(dim=1))

        prefix = torch.stack([img_summary, aud_summary], dim=1)
        prefix = prefix + self.modality_embed(
            torch.tensor([MOD_IMAGE, MOD_AUDIO], device=dev, dtype=torch.long)
        )[None]

        txt = self.text_embed(tokens) + self.modality_embed(
            torch.full((B, T), MOD_TEXT, device=dev, dtype=torch.long)
        ) + self.text_pos(torch.arange(T, device=dev))

        x = torch.cat([prefix, txt], dim=1)
        modality_ids = torch.cat(
            [
                torch.tensor([MOD_IMAGE, MOD_AUDIO], device=dev, dtype=torch.long).expand(B, 2),
                torch.full((B, T), MOD_TEXT, device=dev, dtype=torch.long),
            ],
            dim=1,
        )
        # Both prefix tokens are non-text, so they may see each other in full;
        # text sees both of them and is causal among itself.
        mask_ids = torch.cat(
            [
                torch.zeros(2, device=dev, dtype=torch.long),
                torch.full((T,), MOD_TEXT, device=dev, dtype=torch.long),
            ]
        )
        mask = build_modality_mask(mask_ids)

        probs = None
        for blk in self.decoder:
            x, probs = blk(x, mask)
        h = self.norm_f(x)
        logits = self.lm_head(h[:, 2:])

        text_ce = F.cross_entropy(
            logits.reshape(-1, self.vocab), tokens.reshape(-1), reduction="none"
        ).view(B, T)
        image_mse = (self.image_head(img_summary) - images.mean(dim=1).detach()).pow(2).mean(-1)
        audio_mse = (self.audio_head(aud_summary) - audio.mean(dim=1).detach()).pow(2).mean(-1)

        # One summary token per non-text modality; text has T tokens.
        balanced = (
            image_mse.mean() / 3.0
            + audio_mse.mean() / 3.0
            + text_ce.mean() / 3.0
        )
        aux: dict[str, torch.Tensor] = {
            "latent/modality_ids": modality_ids,
            "attn/modality_mask": mask,
            "latent/projected_summary": torch.stack([img_summary, aud_summary], dim=1),
            "loss/per_modality": torch.stack(
                [image_mse.mean(), audio_mse.mean(), text_ce.mean()]
            ),
            "loss/balanced": balanced,
            "loss/weights": torch.tensor(
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / (3.0 * T)], device=dev
            ),
            "projector/rank": torch.tensor(float(self.image_projector.rank), device=dev),
            "projector/params": torch.tensor(
                float(self.image_projector.param_count() + self.audio_projector.param_count()),
                device=dev,
            ),
            "lm/logits": logits,
        }
        if probs is not None:
            aux["attn/probs"] = probs
        return logits, aux


def build_late_fusion(cfg: ArchConfig) -> LateFusionMultimodal:
    """Convenience constructor for the comparison baseline."""
    return LateFusionMultimodal(cfg)


def projector_overhead(
    cfg: ArchConfig,
    encoder_dim: int | None = None,
    rank: int | None = None,
) -> dict[str, float]:
    """Parameter and capacity accounting for late-fusion projectors.

    The projector is a rank-``r`` map ``R^E -> R^r -> R^d`` per modality.
    Its parameter count is ``E r + r d`` (plus a norm), *linear* in ``r``,
    while the number of image/audio tokens it can preserve is at most ``r``.
    The early-fusion embedder spends ``patch_dim * d`` per modality and keeps
    all ``P`` patch tokens.
    """
    d = int(cfg["d_model"])
    enc = int(encoder_dim if encoder_dim is not None else cfg.get("encoder_dim", 4 * d))
    r = int(rank if rank is not None else cfg.get("projector_rank", max(1, d // 4)))
    p = int(cfg.get("vision_patches", 8))
    a = int(cfg.get("audio_frames", 8))
    patch_dim = int(cfg.get("patch_dim", 32))
    audio_dim = int(cfg.get("audio_dim", 32))
    n_layers = int(cfg["n_layers"])
    d_ff = int(cfg["d_ff"])
    n_experts = int(cfg.get("n_experts", 4))

    per_projector = enc * r + r * d + 2 * r
    late_projector_params = 2 * per_projector
    early_embed_params = (patch_dim + audio_dim) * d
    backbone = (
        int(cfg["vocab"]) * d
        + n_layers * (4 * d * d + n_experts * (3 * d * d_ff) + d * n_experts)
        + 2 * d * (patch_dim + audio_dim)
    )
    return {
        "rank": float(r),
        "encoder_dim": float(enc),
        "per_modality_projector_params": float(per_projector),
        "late_projector_params": float(late_projector_params),
        "early_fusion_embed_params": float(early_embed_params),
        "extra_params_vs_early": float(late_projector_params - early_embed_params),
        "projector_fraction_of_backbone": float(late_projector_params / max(backbone, 1)),
        "rank_fraction_of_d_model": float(r / d),
        "late_summary_dimensions": float(2 * r),
        "early_token_dimensions": float((p + a) * d),
        "capacity_ratio_late_over_early": float((2 * r) / max((p + a) * d, 1)),
    }


# --------------------------------------------------------------------------- #
# Config + symbolic spec
# --------------------------------------------------------------------------- #

def tiny_config(**over) -> ArchConfig:
    p = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=4,
        d_head=16,
        n_layers=4,
        d_ff=128,
        vocab=256,
        seq=32,
        vision_patches=8,
        audio_frames=8,
        patch_dim=32,
        audio_dim=32,
        n_experts=4,
        top_k=1,
        projector_rank=16,
        encoder_dim=256,
        late_encoder_layers=1,
    )
    p.update(over)
    return ArchConfig("multimodal-tiny", "multimodal", p)


@register_spec("multimodal")
def multimodal_spec() -> MathSpec:
    return MathSpec(
        name="multimodal",
        family="native-multimodal",
        summary="Early-fusion transformer over a unified text/image/audio token stream with a modality-aware mask and a balanced loss.",
        equations=[
            (
                "stream",
                r"\mathcal{S} = [\,x^{I}_{1:P}\;\|\;x^{A}_{1:A}\;\|\;x^{T}_{1:T}\,],\qquad "
                r"h^{(0)}_i = W_{m(i)} x_i + e_{m(i)} + p_{m(i),\,i}",
            ),
            (
                "modality_mask",
                r"M_{ij}=0 \iff \big(m_i=m_j\in\{I,A\}\big)\;\lor\;\big(m_i=m_j=T\land j\le i\big)"
                r"\;\lor\;\big(m_i=T\land m_j\in\{I,A\}\big),\quad M_{ij}=-\infty\ \text{otherwise}",
            ),
            (
                "attention",
                r"\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V",
            ),
            (
                "balanced_loss",
                r"\mathcal{L}=\frac{1}{M}\sum_{m=1}^{M}\frac{1}{N_m}\sum_{t:\,m(t)=m}\ell_t"
                r"=\sum_{t} \frac{1}{M\,N_{m(t)}}\,\ell_t,\qquad \sum_t w_t=1",
            ),
            (
                "late_projector",
                r"P_m = U_m V_m^\top,\quad U_m\in\mathbb{R}^{d\times r},\;V_m\in\mathbb{R}^{E\times r}"
                r"\;\Rightarrow\;\operatorname{rank}(P_m)\le r,\quad \dim\operatorname{Im}(P_m)\le r",
            ),
            (
                "rank_capacity",
                r"I(x_m;P_m E_m(x_m))\le \frac{r}{2}\log\!\left(1+\frac{\sigma_x^2}{\sigma_\epsilon^2}\right)"
                r"\quad\text{vs.}\quad (P+A)\,d\ \text{early-fusion coordinates}",
            ),
            (
                "sequence_cost",
                r"|\mathcal{S}|=P+A+T,\qquad \text{attention cost}=O\big((P+A+T)^2 d_h\big)",
            ),
        ],
        assumptions=[
            "All modalities share one residual width d_model; only the input embedders and position tables are modality specific.",
            "Image and audio are prefix modalities: they are encoded bidirectionally and cannot attend to text or to each other; text attends the whole prefix and is causal among itself.",
            "The late-fusion projector is linear and rank-r; the early-fusion embedder is full rank up to min(patch_dim, d_model).",
            "The balanced loss normalises each modality by its token count so no modality dominates the gradient.",
            "The tiny MoE router is modality-conditioned; real native multimodal MoE models learn modality-specific experts.",
        ],
        complexity={
            "params": (
                "vocab*d_model + (patch_dim+audio_dim)*d_model + 6*seq*d_model + 3*d_model"
                " + n_layers*(4*d_model*d_model + d_model*n_experts + n_experts*3*d_model*d_ff)"
                " + (patch_dim+audio_dim)*d_model"
            ),
            "time_per_token": (
                "n_layers*(8*d_model*d_model + 2*n_experts*d_model*d_ff"
                " + 4*n_heads*d_head*(vision_patches+audio_frames+seq))"
                " + 2*d_model*vocab"
            ),
            "kv_bytes_per_token": "2*2*n_layers*n_kv_heads*d_head",
            "projector_params": "2*(encoder_dim*projector_rank + projector_rank*d_model + 2*projector_rank)",
            "late_capacity_ratio": "(2*projector_rank)/((vision_patches+audio_frames)*d_model)",
        },
        invariants=[
            "attention rows sum to 1 for every modality block",
            "no image or audio token attends to text; image and audio never attend to each other",
            "sequence length equals vision_patches + audio_frames + seq",
            "balanced-loss weights sum to 1",
            "each per-modality loss is finite",
            "late-fusion projector rank is at most projector_rank",
        ],
        references=[
            "An et al. 2026, Toward Native Multimodal Modeling: A Roadmap (arXiv:2605.25343)",
            "Shukor et al. 2025, Scaling Laws for Native Multimodal Models (arXiv:2504.07951)",
            "DeepSeek-AI 2026, DeepSeek-V4.1-Flash (arXiv:2609.19969)",
        ],
    )
