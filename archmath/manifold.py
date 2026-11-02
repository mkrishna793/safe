"""Real-Time Mathematical Manifold Decoder and Latent Dynamics Framework.

Integrates differential geometry, spectral graph theory, and information theory
into real-time runtime telemetry for arbitrary neural network architectures:

1. Token Input Curvature & Metric Tensor:
   Computes local Jacobian J = ∂z_L / ∂x_embed for incoming tokens,
   pullback Riemannian metric tensor G = J^T J, and volume distortion
   sqrt(det(G + eps * I)) via Sylvester's determinant theorem.

2. Graph Laplacian Spectral Geometry & Over-Smoothing:
   Evaluates Graph Laplacian L = D - (A + A^T)/2 and Dirichlet Energy:
     E(Z) = 1/2 Tr(Z^T L Z) = 1/4 sum_{i,j} (A_ij + A_ji) ||z_i - z_j||^2.
   Flags representation collapse when E(Z) / ||Z||_F^2 -> 0, and extracts
   algebraic connectivity (Fiedler value lambda_2).

3. Centered Kernel Alignment (CKA) & Geometric Velocity:
   Measures layer-to-layer representation rotation and drift velocity:
     CKA(K_l, K_{l+1}) = HSIC(K_l, K_{l+1}) / sqrt(HSIC(K_l, K_l) * HSIC(K_{l+1}, K_{l+1}))
     v_geom = 1.0 - CKA(K_l, K_{l+1}).

4. Spectral Intrinsic Dimensionality:
   Extracts singular value spectrum sigma_1 >= ... >= sigma_d,
   Participation Ratio PR = (sum sigma_i)^2 / sum sigma_i^2,
   and Von Neumann Spectral Entropy H_spec = -sum p_i ln p_i.

5. Architecture-Specific Latent State Decoding:
   Decodes latent invariants for 12 architecture families:
   Dense/Sparse Attn, Linear Attn/DeltaNet, Looped Transformers, SSM/Mamba,
   MoE, Titans, BLT, JEPA, KAN, Neuromorphic SNNs, RNN/LSTM, and Diffusion.

6. Context Manager API:
   ```python
   with RealTimeMathDecoder(model, mode="train") as decoder:
       out, aux = model(tokens)
       loss = criterion(out, targets)
       metrics = decoder.step(loss=loss, inputs=tokens)
   ```
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import unwrap


# --------------------------------------------------------------------------- #
# Mathematical Core Primitives
# --------------------------------------------------------------------------- #

def compute_dirichlet_energy(
    A: Optional[torch.Tensor],
    Z: torch.Tensor,
    eps: float = 1e-12,
    normalized: bool = False,
) -> float:
    """Compute Graph Dirichlet Energy E(Z) = 1/2 Tr(Z^T L Z) >= 0.

    When normalized=True, returns strictly scale-invariant Dirichlet energy:
      E_norm(Z) = E(Z) / (||Z||_F^2 + eps).
    Guarantees positive semi-definiteness lambda_i >= 0 via exact symmetrization
    L = D_sym - A_sym where A_sym = (A + A^T) / 2 and D_sym = diag(A_sym 1).
    When A is None or non-existent, falls back to the canonical causal line/chain
    token graph A_{i, i+1} = A_{i+1, i} = 0.5.
    """
    with torch.no_grad():
        if Z.dim() >= 3:
            Z_2d = Z[0].float()
        elif Z.dim() == 2:
            Z_2d = Z.float()
        elif Z.dim() == 1:
            Z_2d = Z.unsqueeze(0).float()
        else:
            return 0.0

        Z_2d = torch.nan_to_num(Z_2d, nan=0.0, posinf=0.0, neginf=0.0)
        S, D = Z_2d.shape
        if S <= 1:
            return 0.0

        if A is None:
            idx = torch.arange(S - 1, device=Z_2d.device)
            A_mat = torch.zeros((S, S), device=Z_2d.device, dtype=torch.float32)
            A_mat[idx, idx + 1] = 0.5
            A_mat[idx + 1, idx] = 0.5
        else:
            A_f = A.detach().float()
            A_f = torch.nan_to_num(A_f, nan=0.0, posinf=0.0, neginf=0.0)
            if A_f.dim() == 4:  # (B, H, S, S)
                A_mat = A_f[0].mean(dim=0)
            elif A_f.dim() == 3:  # (B, S, S)
                A_mat = A_f[0]
            elif A_f.dim() == 2:
                A_mat = A_f
            else:
                A_mat = torch.eye(S, device=Z_2d.device, dtype=torch.float32)

            A_mat = torch.clamp_min(A_mat, 0.0)
            min_s = min(A_mat.shape[0], A_mat.shape[1], S)
            if min_s <= 1:
                return 0.0
            A_mat = A_mat[:min_s, :min_s]
            Z_2d = Z_2d[:min_s]

        # Exact Symmetrization: A_sym = (A + A^T) / 2, D_sym = diag(A_sym 1)
        A_sym = 0.5 * (A_mat + A_mat.T)
        deg = A_sym.sum(dim=-1)
        L = torch.diag(deg) - A_sym

        energy = 0.5 * torch.trace(Z_2d.T @ L @ Z_2d)
        val = float(energy.item())
        if not math.isfinite(val) or val < 0.0:
            val = 0.0

        if normalized:
            z_f_sq = float((Z_2d.pow(2)).sum().item())
            val = val / (z_f_sq + eps)
            if not math.isfinite(val) or val < 0.0:
                val = 0.0

        return val


def compute_normalized_dirichlet_energy(
    A: Optional[torch.Tensor],
    Z: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Compute strictly scale-invariant normalized Dirichlet energy E_norm(Z) = E(Z) / (||Z||_F^2 + eps)."""
    return compute_dirichlet_energy(A, Z, eps=eps, normalized=True)


def compute_fiedler_value(
    A: Optional[torch.Tensor],
    S: int = 8,
    device: Any = "cpu",
) -> float:
    """Compute algebraic connectivity (second-smallest eigenvalue of graph Laplacian)."""
    with torch.no_grad():
        if A is None or A.numel() == 0:
            if S <= 1:
                return 0.0
            idx = torch.arange(S - 1, device=device)
            A_mat = torch.zeros((S, S), device=device, dtype=torch.float32)
            A_mat[idx, idx + 1] = 0.5
            A_mat[idx + 1, idx] = 0.5
        else:
            A_f = A.detach().float()
            A_f = torch.nan_to_num(A_f, nan=0.0, posinf=0.0, neginf=0.0)
            if A_f.dim() == 4:
                A_mat = A_f[0].mean(dim=0)
            elif A_f.dim() == 3:
                A_mat = A_f[0]
            elif A_f.dim() == 2:
                A_mat = A_f
            else:
                return 0.0
            A_mat = torch.clamp_min(A_mat, 0.0)

        min_s = min(A_mat.shape[0], A_mat.shape[1])
        if min_s <= 1:
            return 0.0
        A_mat = A_mat[:min_s, :min_s]
        A_sym = 0.5 * (A_mat + A_mat.T)
        L = torch.diag(A_sym.sum(dim=-1)) - A_sym
        try:
            eigs = torch.linalg.eigvalsh(L)
            eigs = torch.sort(eigs).values
            if len(eigs) >= 2:
                return float(max(0.0, eigs[1].item()))
        except Exception:
            pass
        return 0.0


def compute_participation_ratio(Z: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute Participation Ratio PR = (sum sigma_i)^2 / sum sigma_i^2 >= 1.0."""
    with torch.no_grad():
        if Z.numel() == 0:
            return 1.0
        flat = Z.detach().float().reshape(-1, Z.shape[-1])
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        if flat.shape[0] > 256:
            idx = torch.linspace(0, flat.shape[0] - 1, 256, device=flat.device).long()
            flat = flat[idx]

        flat_c = flat - flat.mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(flat_c)
            s = s[s > eps]
            if s.numel() == 0:
                return 1.0
            sum_s = s.sum()
            sum_sq = s.pow(2).sum() + eps
            pr = (sum_s.pow(2)) / sum_sq
            return float(max(1.0, pr.item()))
        except Exception:
            return 1.0


def compute_spectral_entropy(Z: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute Von Neumann spectral entropy H_spec = -sum p_i ln p_i >= 0.0."""
    with torch.no_grad():
        if Z.numel() == 0:
            return 0.0
        flat = Z.detach().float().reshape(-1, Z.shape[-1])
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        if flat.shape[0] > 256:
            idx = torch.linspace(0, flat.shape[0] - 1, 256, device=flat.device).long()
            flat = flat[idx]

        try:
            s = torch.linalg.svdvals(flat)
            s = s[s > eps]
            if s.numel() == 0:
                return 0.0
            p = s / (s.sum() + eps)
            p_clamped = torch.clamp(p, min=eps, max=1.0)
            ent = -(p * torch.log(p_clamped)).sum()
            val = float(ent.item())
            if not math.isfinite(val) or val < 0.0:
                return 0.0
            return val
        except Exception:
            return 0.0


def compute_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute Linear Centered Kernel Alignment CKA(X, Y) in [0.0, 1.0].

    Implements epsilon-regularized HSIC normalization:
      CKA_eps(X, Y) = HSIC(X, Y) / (sqrt(HSIC(X, X) + eps) * sqrt(HSIC(Y, Y) + eps))
    Guarantees strict avoidance of 0/0 division when X or Y has zero variance (constant representations).
    """
    with torch.no_grad():
        if X.numel() == 0 or Y.numel() == 0:
            return 1.0
        x_flat = X.detach().float().reshape(-1, X.shape[-1])
        y_flat = Y.detach().float().reshape(-1, Y.shape[-1])

        x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)
        y_flat = torch.nan_to_num(y_flat, nan=0.0, posinf=0.0, neginf=0.0)

        min_n = min(x_flat.shape[0], y_flat.shape[0])
        if min_n <= 1:
            return 1.0

        if min_n > 256:
            idx = torch.linspace(0, min_n - 1, 256, device=x_flat.device).long()
            x_flat = x_flat[idx]
            y_flat = y_flat[idx]
        else:
            x_flat = x_flat[:min_n]
            y_flat = y_flat[:min_n]

        # Center columns
        x_c = x_flat - x_flat.mean(dim=0, keepdim=True)
        y_c = y_flat - y_flat.mean(dim=0, keepdim=True)

        # HSIC in feature space: ||Y_c^T X_c||_F^2
        dot_xy = torch.matmul(y_c.T, x_c)
        hsic_xy = (dot_xy.pow(2)).sum()

        dot_xx = torch.matmul(x_c.T, x_c)
        hsic_xx = (dot_xx.pow(2)).sum()

        dot_yy = torch.matmul(y_c.T, y_c)
        hsic_yy = (dot_yy.pow(2)).sum()

        hsic_xx_val = float(hsic_xx.item())
        hsic_yy_val = float(hsic_yy.item())

        # Zero-Variance Degeneracy Handling:
        # If both representations have zero variance (constant representations):
        if hsic_xx_val <= eps and hsic_yy_val <= eps:
            diff = float((x_flat - y_flat).abs().max().item())
            return 1.0 if diff <= 1e-5 else 0.0

        # If one representation has zero variance, geometric alignment is zero
        if hsic_xx_val <= eps or hsic_yy_val <= eps:
            return 0.0

        # Epsilon-regularized denominator: sqrt(HSIC_xx + eps) * sqrt(HSIC_yy + eps)
        denom = torch.sqrt((hsic_xx + eps) * (hsic_yy + eps))
        cka = hsic_xy / denom
        val = float(cka.item())
        if not math.isfinite(val):
            return 0.0
        return float(min(1.0 + 1e-4, max(0.0, val)))


def compute_riemannian_metric_tensor(
    J: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Compute Riemannian metric tensor G = J^T J and volume distortion via Sylvester's identity.

    For rectangular or rank-deficient J (m x n), computes regularized log-determinant:
      log det(I + (1/eps) J J^T) = sum log1p(sigma_i^2 / eps)
    via Sylvester's determinant theorem: det(I_m + (1/eps) J J^T) == det(I_n + (1/eps) J^T J).
    Provides unconditional numerical stability against overflow and underflow.
    """
    if J.numel() == 0:
        return {
            "jacobian_norm": 0.0,
            "metric_tensor_norm": 0.0,
            "metric_tensor_trace": 0.0,
            "volume_distortion": 1.0,
            "log_volume_distortion": 0.0,
            "sylvester_log_det": 0.0,
            "metric_condition_number": 1.0,
        }

    J_f = J.detach().float()
    J_f = torch.nan_to_num(J_f, nan=0.0, posinf=0.0, neginf=0.0)
    j_norm = float(J_f.norm().item())

    try:
        s = torch.linalg.svdvals(J_f)
    except Exception:
        s = torch.zeros(1, device=J_f.device)

    # Singular values: sigma_1 >= sigma_2 >= ... >= sigma_k >= 0
    g_trace = float((s.pow(2)).sum().item())
    g_norm = float(torch.sqrt((s.pow(4)).sum()).item())

    # Sylvester's determinant identity:
    # log det(I_m + (1/eps) J J^T) = log det(I_n + (1/eps) J^T J) = sum log1p(sigma_i^2 / eps)
    # Guaranteed non-negative, unconditionally stable with log1p
    sylvester_log_det = float(torch.log1p(s.pow(2) / eps).sum().item())
    log_vol = 0.5 * sylvester_log_det
    vol_distortion = float(math.exp(max(-100.0, min(100.0, log_vol))))

    # Exact condition number accounting for rank-deficiency and rectangularity
    m, n = J_f.shape[0], J_f.shape[1]
    min_dim = min(m, n)
    s_max = float(s[0].item()) if s.numel() > 0 else 0.0
    if s.numel() < min_dim or (s.numel() > 0 and float(s[-1].item()) <= 1e-12):
        s_min = 0.0
    else:
        s_min = float(s[-1].item())

    cond_num = float((s_max ** 2 + eps) / (s_min ** 2 + eps))
    if not math.isfinite(cond_num) or cond_num < 1.0:
        cond_num = 1.0

    return {
        "jacobian_norm": j_norm,
        "metric_tensor_norm": g_norm,
        "metric_tensor_trace": g_trace,
        "volume_distortion": vol_distortion,
        "log_volume_distortion": log_vol,
        "sylvester_log_det": sylvester_log_det,
        "metric_condition_number": cond_num,
    }


# --------------------------------------------------------------------------- #
# Metrics Data Structure & Report
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class ManifoldMetrics:
    """Rigorous snapshot of representation manifold geometry and latent dynamics."""

    step: int
    loss: Optional[float] = None
    mode: str = "train"

    # Graph Laplacian Spectral Geometry
    dirichlet_energy: float = 0.0
    relative_dirichlet_energy: float = 0.0
    normalized_dirichlet_energy: float = 0.0
    fiedler_value: float = 0.0
    laplacian_spectral_gap: float = 0.0
    is_oversmoothed: bool = False

    # Spectral Intrinsic Dimensionality
    participation_ratio: float = 1.0
    spectral_entropy: float = 0.0
    von_neumann_entropy: float = 0.0
    effective_rank: float = 1.0
    svd_spectrum: List[float] = dataclasses.field(default_factory=list)

    # CKA Dynamics & Geometric Velocity
    cka_adjacent: List[float] = dataclasses.field(default_factory=list)
    layer_cka: Dict[str, float] = dataclasses.field(default_factory=dict)
    mean_cka: float = 1.0
    geometric_velocity: float = 0.0

    # Curvature & Riemannian Metric Tensor
    curvature_norm: float = 0.0
    jacobian_norm: float = 0.0
    metric_tensor_norm: float = 0.0
    metric_tensor_trace: float = 0.0
    volume_distortion: float = 0.0
    log_volume_distortion: float = 0.0
    sylvester_log_det: float = 0.0
    metric_condition_number: float = 1.0

    # Architecture & Layer Dictionaries
    layer_metrics: Dict[str, Dict[str, float]] = dataclasses.field(default_factory=dict)
    architecture_metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    arch_type: Optional[str] = None
    arch_metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    # Convenience accessors for 12 architecture families
    # 1. Dense / Sparse Attention
    attn_entropy: Optional[float] = None
    attn_sink_mass: Optional[float] = None
    attn_sparsity: Optional[float] = None

    # 2. Linear Attention & DeltaNet
    assoc_memory_norm: Optional[float] = None
    memory_retention_rate: Optional[float] = None
    delta_proj_error: Optional[float] = None

    # 3. Looped Transformers
    contraction_ratio: Optional[float] = None
    banach_distance: Optional[float] = None

    # 4. SSM / Mamba
    state_vector_norm: Optional[float] = None
    decay_eigenvalue_prod: Optional[float] = None

    # 5. MoE
    router_entropy: Optional[float] = None
    expert_gini: Optional[float] = None
    routing_imbalance_cv: Optional[float] = None

    # 6. Titans
    memory_surprise: Optional[float] = None
    momentum_update_norm: Optional[float] = None

    # 7. BLT
    byte_entropy: Optional[float] = None
    patch_trigger_freq: Optional[float] = None

    # 8. JEPA
    latent_energy: Optional[float] = None
    latent_variance: Optional[float] = None
    is_vicreg_collapsed: Optional[bool] = None

    # 9. KAN
    spline_curvature_integral: Optional[float] = None

    # 10. Neuromorphic
    firing_frequency: Optional[float] = None
    membrane_variance: Optional[float] = None

    # 11. RNN / LSTM
    cec_product: Optional[float] = None
    w_hh_spectral_radius: Optional[float] = None

    # 12. Diffusion
    trajectory_velocity: Optional[float] = None
    noise_snr: Optional[float] = None

    # 13. MLA (Multi-Head Latent Attention)
    mla_compression_ratio: Optional[float] = None
    mla_absorption_error: Optional[float] = None

    # 14. Deep MLP
    mlp_expansion_ratio: Optional[float] = None
    mlp_drift_norm: Optional[float] = None

    # 15. Hybrid
    hybrid_kv_bytes: Optional[float] = None
    hybrid_flop_ratio: Optional[float] = None

    # 16. Multimodal
    cross_modal_alignment: Optional[float] = None
    modality_loss_imbalance: Optional[float] = None

    # 17. Advanced Frontier Mathematical Metrics (EK-FAC, Sheaf, Circuits, Harmonic)
    ekfac_condition_number: Optional[float] = None
    sheaf_algebraic_connectivity: Optional[float] = None
    sheaf_dirichlet_energy: Optional[float] = None
    is_sheaf_trivial: Optional[bool] = None
    dla_linearity_error: Optional[float] = None
    circuit_dag_sparsity: Optional[float] = None
    fourier_concentration_ratio: Optional[float] = None
    rope_weyl_equidistribution: Optional[float] = None
    ramanujan_expansion_ratio: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        lines = [
            f"[ManifoldMetrics step={self.step} mode={self.mode} loss={self.loss:.4f if self.loss is not None else 'None'}]",
            f"  Curvature: ||J||_F={self.jacobian_norm:.3e}, vol_dist={self.volume_distortion:.3e}, cond(G)={self.metric_condition_number:.2f}",
            f"  Laplacian: E(Z)={self.dirichlet_energy:.4f}, E_rel={self.relative_dirichlet_energy:.4e}, Fiedler(lambda_2)={self.fiedler_value:.4f}, collapse={self.is_oversmoothed}",
            f"  Geometry:  mean_CKA={self.mean_cka:.4f}, v_geom={self.geometric_velocity:.4f}",
            f"  Spectrum:  PR={self.participation_ratio:.2f}, H_spec={self.spectral_entropy:.3f}, erank={self.effective_rank:.2f}",
        ]
        if self.arch_type:
            arch_items = ", ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in self.arch_metrics.items())
            lines.append(f"  Arch[{self.arch_type}]: {arch_items}")
        if self.anomalies:
            lines.append(f"  Anomalies: {len(self.anomalies)} detected")
        return "\n".join(lines)


class ManifoldReport:
    """Historical telemetry aggregator for manifold analysis."""

    def __init__(self, tag: str, steps: List[ManifoldMetrics]) -> None:
        self.tag = tag
        self.steps = steps

    @property
    def anomalies(self) -> List[Dict[str, Any]]:
        out = []
        for s in self.steps:
            out.extend(s.anomalies)
        return out

    def summary(self) -> str:
        if not self.steps:
            return f"ManifoldReport [{self.tag}]: No steps recorded"
        last = self.steps[-1]
        return (
            f"ManifoldReport [{self.tag}] - Step {last.step}:\n"
            f"  Dirichlet Energy E(Z) : {last.dirichlet_energy:.4f}\n"
            f"  Participation Ratio   : {last.participation_ratio:.2f}\n"
            f"  Mean Adjacent CKA     : {last.mean_cka:.4f}\n"
            f"  Spectral Entropy      : {last.spectral_entropy:.4f}\n"
            f"  Curvature / Vol Dist  : {last.jacobian_norm:.4f} / {last.volume_distortion:.4f}\n"
            f"  Anomalies Detected    : {len(self.anomalies)}"
        )


# --------------------------------------------------------------------------- #
# Architecture-Specific Latent State Decoding
# --------------------------------------------------------------------------- #

def decode_arch_latent_state(
    aux: Dict[str, Any],
    model: Optional[nn.Module] = None,
    final_z: Optional[torch.Tensor] = None,
) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
    """Decode latent invariants across 12 distinct architecture families."""
    arch_type: Optional[str] = None
    metrics: Dict[str, Any] = {}
    conveniences: Dict[str, Any] = {}

    if model is not None:
        if hasattr(model, "spec") and hasattr(model.spec, "name"):
            arch_type = model.spec.name
        elif hasattr(model, "name"):
            arch_type = model.name

    def find_key(*patterns: str) -> Any:
        for p in patterns:
            for k, v in aux.items():
                if p in k:
                    return v
        return None

    # 1. Dense & Sparse Attention
    attn_probs = find_key("/attn/probs", "attn/probs")
    if attn_probs is not None and isinstance(attn_probs, torch.Tensor):
        if arch_type is None:
            arch_type = "sparse_attn" if any("sparse" in k or "indexer" in k for k in aux) else "dense"
        A = attn_probs.detach().float()
        A_clean = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A_clean = torch.clamp_min(A_clean, 0.0)
        eps = 1e-12
        A_clamped = torch.clamp(A_clean, min=eps, max=1.0)
        ent = -(A_clean * torch.log(A_clamped)).sum(dim=-1).mean().item()
        val_ent = float(max(0.0, ent)) if math.isfinite(ent) else 0.0
        metrics["attn_entropy"] = val_ent
        conveniences["attn_entropy"] = val_ent

        if A.shape[-1] > 1:
            sink = A[..., :, 0].mean().item()
            metrics["attn_sink_mass"] = float(sink)
            conveniences["attn_sink_mass"] = float(sink)

        sparsity = find_key("/attn/sparsity", "layer0/attn/sparsity")
        if sparsity is not None:
            sp_val = float(sparsity.detach().item()) if isinstance(sparsity, torch.Tensor) else float(sparsity)
            metrics["attn_sparsity"] = sp_val
            conveniences["attn_sparsity"] = sp_val

    # 2. Linear Attention & DeltaNet
    linear_s = find_key("state/S")
    if linear_s is not None and isinstance(linear_s, torch.Tensor):
        arch_type = arch_type or "linear_attn"
        s_norm = float(linear_s.detach().float().norm().item())
        metrics["assoc_memory_norm"] = s_norm
        conveniences["assoc_memory_norm"] = s_norm

        alpha = find_key("gate/alpha")
        if alpha is not None and isinstance(alpha, torch.Tensor):
            retention = float(alpha.detach().float().mean().item())
            metrics["memory_retention_rate"] = retention
            conveniences["memory_retention_rate"] = retention

        delta_norm = find_key("state/delta_norm")
        if delta_norm is not None and isinstance(delta_norm, torch.Tensor):
            proj_err = float(delta_norm.detach().float().mean().item())
            metrics["delta_proj_error"] = proj_err
            conveniences["delta_proj_error"] = proj_err

    # 3. Looped Transformers
    loop_h = find_key("state/loop_h")
    if loop_h is not None and isinstance(loop_h, torch.Tensor):
        arch_type = arch_type or "looped"
        K = loop_h.shape[0]
        if K >= 3:
            h_f = loop_h.detach().float()
            deltas = [float((h_f[k + 1] - h_f[k]).norm().item()) for k in range(K - 1)]
            ratios = [deltas[k + 1] / (deltas[k] + 1e-12) for k in range(len(deltas) - 1)]
            q = float(sum(ratios) / len(ratios)) if ratios else 1.0
            metrics["contraction_ratio"] = q
            conveniences["contraction_ratio"] = q

            banach = float((q / (1.0 - q + 1e-12)) * deltas[-1]) if q < 1.0 else float("inf")
            metrics["banach_distance"] = banach
            conveniences["banach_distance"] = banach

    # 4. SSM / Mamba
    ssm_h = find_key("layer0/state/h", "state/h")
    if ssm_h is not None and isinstance(ssm_h, torch.Tensor):
        arch_type = arch_type or "ssm"
        h_norm = float(ssm_h.detach().float().norm().item())
        metrics["state_vector_norm"] = h_norm
        conveniences["state_vector_norm"] = h_norm

        a_bar = find_key("state/A_bar")
        if a_bar is not None and isinstance(a_bar, torch.Tensor):
            log_a = torch.log(a_bar.detach().float().abs() + 1e-12).mean().item()
            decay_prod = float(math.exp(max(-100.0, min(100.0, log_a))))
            metrics["decay_eigenvalue_prod"] = decay_prod
            conveniences["decay_eigenvalue_prod"] = decay_prod

    # 5. MoE
    moe_probs = find_key("/route/probs", "route/probs")
    if moe_probs is not None and isinstance(moe_probs, torch.Tensor):
        arch_type = arch_type or "moe"
        P = moe_probs.detach().float()
        P_clean = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        P_clean = torch.clamp_min(P_clean, 0.0)
        P_clamped = torch.clamp(P_clean, min=1e-12, max=1.0)
        ent = -(P_clean * torch.log(P_clamped)).sum(dim=-1).mean().item()
        val_ent = float(max(0.0, ent)) if math.isfinite(ent) else 0.0
        metrics["router_entropy"] = val_ent
        conveniences["router_entropy"] = val_ent

        load = find_key("/route/load", "route/load")
        if load is not None and isinstance(load, torch.Tensor):
            L = load.detach().float().reshape(-1)
            E = L.shape[0]
            if E > 1 and L.sum() > 0:
                diff = (L.unsqueeze(1) - L.unsqueeze(0)).abs().sum().item()
                gini = float(diff / (2.0 * E * L.sum().item() + 1e-12))
                metrics["expert_gini"] = gini
                conveniences["expert_gini"] = gini

                cv = float((L.std() / (L.mean() + 1e-12)).item())
                metrics["routing_imbalance_cv"] = cv
                conveniences["routing_imbalance_cv"] = cv

    # 6. Titans
    surprise = find_key("state/surprise")
    if surprise is not None and isinstance(surprise, torch.Tensor):
        arch_type = arch_type or "titans"
        s_val = float(surprise.detach().float().mean().item())
        metrics["memory_surprise"] = s_val
        conveniences["memory_surprise"] = s_val

        m_delta = find_key("state/memory_delta")
        if m_delta is not None and isinstance(m_delta, torch.Tensor):
            delta_val = float(m_delta.detach().float().norm().item())
            metrics["momentum_update_norm"] = delta_val
            conveniences["momentum_update_norm"] = delta_val

    # 7. BLT
    byte_ent = find_key("byte/entropy")
    if byte_ent is not None and isinstance(byte_ent, torch.Tensor):
        arch_type = arch_type or "blt"
        b_val = float(byte_ent.detach().float().mean().item())
        metrics["byte_entropy"] = b_val
        conveniences["byte_entropy"] = b_val

        patch_rate = find_key("patch/rate")
        if patch_rate is not None:
            r_val = float(patch_rate.detach().item()) if isinstance(patch_rate, torch.Tensor) else float(patch_rate)
            metrics["patch_trigger_freq"] = r_val
            conveniences["patch_trigger_freq"] = r_val

    # 8. JEPA
    world_loss = find_key("world/loss")
    if world_loss is not None:
        arch_type = arch_type or "jepa"
        loss_val = float(world_loss.detach().item()) if isinstance(world_loss, torch.Tensor) else float(world_loss)
        metrics["latent_energy"] = loss_val
        conveniences["latent_energy"] = loss_val

        pred = find_key("world/pred")
        if pred is not None and isinstance(pred, torch.Tensor):
            p_flat = pred.detach().float().reshape(-1, pred.shape[-1])
            var = float(p_flat.var(dim=0).mean().item())
            metrics["latent_variance"] = var
            conveniences["latent_variance"] = var
            is_collapsed = bool(var < 0.01)
            metrics["is_vicreg_collapsed"] = is_collapsed
            conveniences["is_vicreg_collapsed"] = is_collapsed

    # 9. KAN
    kan_coeff = find_key("kan/coeff", "/kan/coeff")
    if kan_coeff is not None and isinstance(kan_coeff, torch.Tensor):
        arch_type = arch_type or "kan"
        C = kan_coeff.detach().float()
        if C.shape[-1] >= 3:
            d2 = C[..., 2:] - 2.0 * C[..., 1:-1] + C[..., :-2]
            curv = float((d2 ** 2).sum().item())
            metrics["spline_curvature_integral"] = curv
            conveniences["spline_curvature_integral"] = curv

    # 10. Neuromorphic
    spike_rate = find_key("spike/rate", "/spike/rate")
    if spike_rate is not None and isinstance(spike_rate, torch.Tensor):
        arch_type = arch_type or "neuromorphic"
        rate_val = float(spike_rate.detach().float().mean().item())
        metrics["firing_frequency"] = rate_val
        conveniences["firing_frequency"] = rate_val

        membrane = find_key("state/membrane", "/state/membrane")
        if membrane is not None and isinstance(membrane, torch.Tensor):
            v_var = float(membrane.detach().float().var().item())
            metrics["membrane_variance"] = v_var
            conveniences["membrane_variance"] = v_var

    # 11. RNN / LSTM
    forget_gate = find_key("gate/forget", "/gate/forget")
    if forget_gate is not None and isinstance(forget_gate, torch.Tensor):
        arch_type = arch_type or "rnn"
        f_val = forget_gate.detach().float()
        log_f = torch.log(f_val.clamp(min=1e-12)).mean().item()
        cec = float(math.exp(max(-100.0, min(100.0, log_f))))
        metrics["cec_product"] = cec
        conveniences["cec_product"] = cec

        radius = find_key("w_hh_spectral_radius", "state/w_hh_spectral_radius")
        if radius is not None and isinstance(radius, torch.Tensor):
            r_val = float(radius.detach().float().mean().item())
            metrics["w_hh_spectral_radius"] = r_val
            conveniences["w_hh_spectral_radius"] = r_val

    # 12. Diffusion
    denoise_conf = find_key("denoise/step_confidence", "denoise/conf_trace")
    if denoise_conf is not None and isinstance(denoise_conf, torch.Tensor):
        arch_type = arch_type or "diffusion"
        c_f = denoise_conf.detach().float()
        if c_f.dim() >= 1 and c_f.shape[0] >= 2:
            vel = float((c_f[1:] - c_f[:-1]).abs().mean().item())
            metrics["trajectory_velocity"] = vel
            conveniences["trajectory_velocity"] = vel

        elbo = find_key("denoise/elbo")
        if elbo is not None:
            elbo_val = float(elbo.detach().item()) if isinstance(elbo, torch.Tensor) else float(elbo)
            snr = float(1.0 / (abs(elbo_val) + 1e-4))
            metrics["noise_snr"] = snr
            conveniences["noise_snr"] = snr

    # 13. MLA (Multi-Head Latent Attention)
    c_kv = find_key("latent/c_kv")
    comp_ratio = find_key("latent/compression_ratio")
    abs_err = find_key("latent/absorption_error")
    if c_kv is not None or comp_ratio is not None:
        arch_type = arch_type or "mla"
        if comp_ratio is not None:
            c_val = float(comp_ratio.detach().item()) if isinstance(comp_ratio, torch.Tensor) else float(comp_ratio)
            metrics["mla_compression_ratio"] = c_val
            conveniences["mla_compression_ratio"] = c_val
        if abs_err is not None:
            e_val = float(abs_err.detach().item()) if isinstance(abs_err, torch.Tensor) else float(abs_err)
            metrics["mla_absorption_error"] = e_val
            conveniences["mla_absorption_error"] = e_val

    # 14. Deep MLP
    mlp_out = find_key("mlp/block_out")
    if mlp_out is not None and isinstance(mlp_out, torch.Tensor):
        arch_type = arch_type or "mlp"
        drift = float(mlp_out.detach().float().norm().item())
        metrics["mlp_drift_norm"] = drift
        conveniences["mlp_drift_norm"] = drift
        if model is not None and hasattr(model, "cfg"):
            d_ff = model.cfg.get("d_ff", 0)
            d_model = model.cfg.get("d_model", 1)
            ratio = float(d_ff / max(1, d_model))
            metrics["mlp_expansion_ratio"] = ratio
            conveniences["mlp_expansion_ratio"] = ratio

    # 15. Hybrid
    kv_bytes = find_key("hybrid/kv_bytes_per_token")
    if kv_bytes is not None:
        arch_type = arch_type or "hybrid"
        b_val = float(kv_bytes.detach().item()) if isinstance(kv_bytes, torch.Tensor) else float(kv_bytes)
        metrics["hybrid_kv_bytes"] = b_val
        conveniences["hybrid_kv_bytes"] = b_val
        decode_flops = find_key("hybrid/decode_flops_per_token")
        if decode_flops is not None:
            f_val = float(decode_flops.detach().item()) if isinstance(decode_flops, torch.Tensor) else float(decode_flops)
            metrics["hybrid_flop_ratio"] = f_val
            conveniences["hybrid_flop_ratio"] = f_val

    # 16. Multimodal
    mod_losses = find_key("loss/per_modality")
    if mod_losses is not None and isinstance(mod_losses, torch.Tensor):
        arch_type = arch_type or "multimodal"
        losses = mod_losses.detach().float().reshape(-1)
        if losses.numel() > 1:
            imb = float((losses.std() / (losses.mean() + 1e-12)).item())
            metrics["modality_loss_imbalance"] = imb
            conveniences["modality_loss_imbalance"] = imb
        fused = find_key("latent/fused")
        if fused is not None and isinstance(fused, torch.Tensor):
            f_norm = float(fused.detach().float().norm().item())
            metrics["cross_modal_alignment"] = f_norm
            conveniences["cross_modal_alignment"] = f_norm

    return arch_type, metrics, conveniences


# --------------------------------------------------------------------------- #
# RealTimeMathDecoder Context Manager & Telemetry Engine
# --------------------------------------------------------------------------- #

class RealTimeMathDecoder:
    """Attaches non-intrusively to any PyTorch architecture to decode hidden space dynamics."""

    def __init__(
        self,
        model: nn.Module,
        mode: str = "train",
        tag: str = "arch",
        *,
        max_jacobian_dim: int = 32,
        subsample_rows: int = 256,
        every: int = 1,
        eps: float = 1e-6,
        anomaly_z: float = 4.0,
    ) -> None:
        self.model = model
        self.mode = mode.lower() if isinstance(mode, str) else "train"
        self.tag = tag
        self.max_jacobian_dim = max(1, max_jacobian_dim)
        self.subsample_rows = subsample_rows
        self.every = max(1, every)
        self.eps = eps
        self.anomaly_z = anomaly_z

        self._handles: List[Any] = []
        self._layer_outputs: Dict[str, torch.Tensor] = {}
        self._accumulated_aux: Dict[str, Any] = {}
        self._records: List[ManifoldMetrics] = []
        self._step = -1
        self._inputs: Optional[torch.Tensor] = None
        self._embed_tensor: Optional[torch.Tensor] = None
        self._final_hidden: Optional[torch.Tensor] = None
        self._t0 = time.time()
        self._prev_training_mode: bool = model.training

    # -- Context Manager Lifecycle ---------------------------------------- #

    def __enter__(self) -> "RealTimeMathDecoder":
        self._prev_training_mode = self.model.training
        if self.mode == "train":
            self.model.train()
        elif self.mode in ("eval", "inference"):
            self.model.eval()

        self._install_hooks()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._remove_hooks()
        if self._prev_training_mode:
            self.model.train()
        else:
            self.model.eval()

    def _install_hooks(self) -> None:
        self._remove_hooks()

        # Root hook captures full model output tuple (logits, aux)
        def root_hook(module: nn.Module, inputs: Any, output: Any) -> None:
            y, aux = unwrap(output)
            if aux:
                self._accumulated_aux.update(aux)
            if isinstance(y, torch.Tensor):
                self._final_hidden = y

        self._handles.append(self.model.register_forward_hook(root_hook))

        # Submodule hooks capture layer-wise activations and embeddings
        for name, module in self.model.named_modules():
            if name == "":
                continue

            def make_submodule_hook(mod_name: str):
                def sub_hook(m: nn.Module, inp: Any, out: Any) -> None:
                    y, aux = unwrap(out)
                    if aux:
                        self._accumulated_aux.update(aux)

                    if any(term in mod_name.lower() for term in ("embed", "tok_embed", "wte")):
                        if isinstance(y, torch.Tensor):
                            self._embed_tensor = y

                    is_layer = any(
                        term in mod_name
                        for term in ("blocks.", "layers.", "layer", "transformer.")
                    ) and ("." not in mod_name or mod_name.count(".") <= 2)
                    if is_layer and isinstance(y, torch.Tensor) and y.dim() >= 2:
                        self._layer_outputs[mod_name] = y

                return sub_hook

            self._handles.append(module.register_forward_hook(make_submodule_hook(name)))

    def _remove_hooks(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    # -- Step Telemetry APIs ---------------------------------------------- #

    def begin_step(self, step: int, inputs: Optional[torch.Tensor] = None) -> None:
        """Mark the beginning of a training step (legacy / explicit API)."""
        self._step = step
        self._layer_outputs.clear()
        self._accumulated_aux.clear()
        self._embed_tensor = None
        self._final_hidden = None
        self._inputs = None
        if inputs is not None and isinstance(inputs, torch.Tensor):
            if inputs.is_floating_point() and inputs.requires_grad:
                inputs.retain_grad()
            self._inputs = inputs

    def end_step(
        self,
        loss: Optional[Union[torch.Tensor, float]] = None,
        aux: Optional[Dict[str, Any]] = None,
    ) -> ManifoldMetrics:
        """Mark the end of a training step (legacy / explicit API)."""
        return self.step(loss=loss, inputs=self._inputs, aux=aux)

    def step(
        self,
        loss: Optional[Union[torch.Tensor, float]] = None,
        inputs: Optional[torch.Tensor] = None,
        output: Any = None,
        aux: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ManifoldMetrics:
        """Primary step evaluation API for the real-time mathematical decoder.

        Usage:
            metrics = decoder.step(loss=loss, inputs=tokens)
        """
        self._step += 1
        loss_val = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else (float(loss) if loss is not None else None)

        if inputs is not None and isinstance(inputs, torch.Tensor):
            self._inputs = inputs

        if aux is not None:
            self._accumulated_aux.update(aux)

        if output is not None:
            y, out_aux = unwrap(output)
            if out_aux:
                self._accumulated_aux.update(out_aux)
            if isinstance(y, torch.Tensor):
                self._final_hidden = y

        # 1. Attention & Laplacian geometry
        attn_tensor: Optional[torch.Tensor] = None
        for k in ("layer0/attn/probs", "attn/probs", "attn_probs"):
            if k in self._accumulated_aux and isinstance(self._accumulated_aux[k], torch.Tensor):
                attn_tensor = self._accumulated_aux[k]
                break

        # 2. Selected structural layer representations
        selected_layers: List[Tuple[str, torch.Tensor]] = []
        for k in sorted(self._layer_outputs.keys()):
            val = self._layer_outputs[k]
            if val.dim() >= 2 and val.shape[-1] >= 4:
                selected_layers.append((k, val))

        if not selected_layers and self._layer_outputs:
            first_k = next(iter(self._layer_outputs))
            selected_layers.append((first_k, self._layer_outputs[first_k]))

        # Per-layer metrics
        layer_metrics: Dict[str, Dict[str, float]] = {}
        dirichlet_energies: List[float] = []
        prs: List[float] = []
        spectral_entropies: List[float] = []

        # If no explicit attention tensor, synthesize causal line graph or state affinity
        for name, layer_t in selected_layers:
            e_dir = compute_dirichlet_energy(attn_tensor, layer_t)
            pr = compute_participation_ratio(layer_t)
            s_ent = compute_spectral_entropy(layer_t)

            dirichlet_energies.append(e_dir)
            prs.append(pr)
            spectral_entropies.append(s_ent)

            layer_metrics[name] = {
                "dirichlet_energy": e_dir,
                "participation_ratio": pr,
                "spectral_entropy": s_ent,
                "norm": float(layer_t.float().norm().item()),
                "std": float(layer_t.float().std().item()),
            }

        mean_e_dir = float(sum(dirichlet_energies) / len(dirichlet_energies)) if dirichlet_energies else 0.0
        mean_pr = float(sum(prs) / len(prs)) if prs else 1.0
        mean_s_ent = float(sum(spectral_entropies) / len(spectral_entropies)) if spectral_entropies else 0.0
        fiedler = compute_fiedler_value(attn_tensor, S=8)

        # 3. Adjacent Layer Linear CKA
        cka_list: List[float] = []
        layer_cka_dict: Dict[str, float] = {}
        if len(selected_layers) >= 2:
            for i in range(len(selected_layers) - 1):
                trans_name = f"{selected_layers[i][0]}->{selected_layers[i + 1][0]}"
                cka_val = compute_cka(selected_layers[i][1], selected_layers[i + 1][1])
                cka_list.append(cka_val)
                layer_cka_dict[trans_name] = cka_val
        elif len(selected_layers) == 1:
            cka_list.append(1.0)
            layer_cka_dict["layer0->layer0"] = 1.0
        else:
            cka_list.append(1.0)

        mean_cka = float(sum(cka_list) / len(cka_list)) if cka_list else 1.0
        geom_vel = max(0.0, 1.0 - mean_cka)

        # 4. Token Input Curvature & Metric Tensor
        x_embed = self._accumulated_aux.get("embed/out")
        if x_embed is None:
            x_embed = self._embed_tensor
        if x_embed is None and self._inputs is not None and self._inputs.is_floating_point():
            x_embed = self._inputs

        z_L = self._accumulated_aux.get("hidden/final")
        if z_L is None:
            z_L = self._final_hidden
        if z_L is None and selected_layers:
            z_L = selected_layers[-1][1]

        curv_stats = self._compute_token_curvature(x_embed, z_L, self._inputs)

        # 5. Architecture-Specific Latent State Invariants
        arch_type, arch_metrics, conveniences = decode_arch_latent_state(
            self._accumulated_aux,
            model=self.model,
            final_z=z_L,
        )

        # 6. SVD Spectrum & Intrinsic Dimensionality
        rep_for_svd = z_L if z_L is not None else x_embed
        svd_spectrum: List[float] = []
        erank = 1.0
        if rep_for_svd is not None and rep_for_svd.numel() > 0:
            flat_s = rep_for_svd.detach().float().reshape(-1, rep_for_svd.shape[-1])
            if flat_s.shape[0] > 256:
                idx = torch.linspace(0, flat_s.shape[0] - 1, 256, device=flat_s.device).long()
                flat_s = flat_s[idx]
            try:
                s_vals = torch.linalg.svdvals(flat_s)
                s_vals = s_vals[s_vals > 1e-12]
                svd_spectrum = [float(v.item()) for v in s_vals[:10]]
                p2 = s_vals.pow(2) / (s_vals.pow(2).sum() + 1e-12)
                p2_clamped = torch.clamp(p2, min=1e-12, max=1.0)
                erank = float(torch.exp(-(p2 * torch.log(p2_clamped)).sum()).item())
            except Exception:
                pass

        # 7. Over-smoothing & Collapse Detection
        z_norm_sq = float((z_L.float().norm().item() ** 2)) if z_L is not None else 1.0
        rel_dirichlet = mean_e_dir / (z_norm_sq + 1e-12)
        is_oversmoothed = bool(rel_dirichlet < 1e-4 and mean_e_dir >= 0.0)

        # 8. Real-time Smoke Alarm / Anomaly Detection
        anomalies: List[Dict[str, Any]] = []
        for name, m in layer_metrics.items():
            if not math.isfinite(m["norm"]) or not math.isfinite(m["dirichlet_energy"]):
                anomalies.append({"step": self._step, "kind": "non_finite", "where": name})
            if m["std"] < 1e-7:
                anomalies.append({"step": self._step, "kind": "dead_layer", "where": name})
            if m["participation_ratio"] < 1.0:
                anomalies.append({"step": self._step, "kind": "pr_violation", "where": name, "value": m["participation_ratio"]})
            if m["dirichlet_energy"] < 0.0:
                anomalies.append({"step": self._step, "kind": "dirichlet_violation", "where": name, "value": m["dirichlet_energy"]})

        # 9. Advanced Frontier Mathematics: Sheaf Laplacian, Ramanujan, RoPE Torus
        sheaf_conn: Optional[float] = None
        sheaf_energy: Optional[float] = None
        is_sheaf_triv: Optional[bool] = None
        ramanujan_ratio: Optional[float] = None
        weyl_equidist: Optional[float] = None

        if attn_tensor is not None and z_L is not None:
            try:
                from .advanced_math import CellularSheafLaplacian, HarmonicTorusSpectrum
                S_nodes = min(8, z_L.shape[1] if z_L.dim() >= 2 else z_L.shape[0])
                D_dim = min(8, z_L.shape[-1])
                if S_nodes >= 2 and D_dim >= 2:
                    sheaf = CellularSheafLaplacian(num_nodes=S_nodes, dim=D_dim)
                    sheaf.build_from_attention(attn_tensor, threshold=0.05, rotation_scale=0.1)
                    sheaf_energy = sheaf.compute_sheaf_dirichlet_energy(z_L, normalized=True)
                    s_report = sheaf.spectral_analysis()
                    sheaf_conn = s_report.algebraic_connectivity
                    is_sheaf_triv = s_report.is_harmonic_trivial
                    ramanujan_ratio = HarmonicTorusSpectrum.ramanujan_expansion_ratio(attn_tensor)
            except Exception:
                pass

        if z_L is not None and z_L.dim() >= 2:
            try:
                from .advanced_math import HarmonicTorusSpectrum
                seq_l = z_L.shape[1] if z_L.dim() >= 3 else z_L.shape[0]
                d_m = z_L.shape[-1]
                if seq_l >= 2 and d_m >= 4:
                    mean_w, _ = HarmonicTorusSpectrum.rope_torus_winding(dim=d_m, seq_len=min(seq_l, 128))
                    weyl_equidist = mean_w
            except Exception:
                pass

        metric_snap = ManifoldMetrics(
            step=self._step,
            loss=loss_val,
            mode=self.mode,
            # Graph Laplacian
            dirichlet_energy=mean_e_dir,
            relative_dirichlet_energy=rel_dirichlet,
            normalized_dirichlet_energy=rel_dirichlet,
            fiedler_value=fiedler,
            laplacian_spectral_gap=fiedler,
            is_oversmoothed=is_oversmoothed,
            # Spectrum
            participation_ratio=mean_pr,
            spectral_entropy=mean_s_ent,
            von_neumann_entropy=mean_s_ent,
            effective_rank=erank,
            svd_spectrum=svd_spectrum,
            # CKA
            cka_adjacent=cka_list,
            layer_cka=layer_cka_dict,
            mean_cka=mean_cka,
            geometric_velocity=geom_vel,
            # Curvature
            curvature_norm=curv_stats["jacobian_norm"],
            jacobian_norm=curv_stats["jacobian_norm"],
            metric_tensor_norm=curv_stats["metric_tensor_norm"],
            metric_tensor_trace=curv_stats["metric_tensor_trace"],
            volume_distortion=curv_stats["volume_distortion"],
            log_volume_distortion=curv_stats.get("log_volume_distortion", 0.0),
            sylvester_log_det=curv_stats.get("sylvester_log_det", 0.0),
            metric_condition_number=curv_stats["metric_condition_number"],
            # Advanced Frontier Math
            sheaf_algebraic_connectivity=sheaf_conn,
            sheaf_dirichlet_energy=sheaf_energy,
            is_sheaf_trivial=is_sheaf_triv,
            ramanujan_expansion_ratio=ramanujan_ratio,
            rope_weyl_equidistribution=weyl_equidist,
            # Layers & Arch
            layer_metrics=layer_metrics,
            architecture_metrics=arch_metrics,
            arch_type=arch_type,
            arch_metrics=arch_metrics,
            anomalies=anomalies,
            **conveniences,
        )

        self._records.append(metric_snap)
        return metric_snap

    def report(self) -> ManifoldReport:
        return ManifoldReport(tag=self.tag, steps=self._records)

    # -- Curvature Helpers ------------------------------------------------ #

    def _compute_token_curvature(
        self,
        x_embed: Optional[torch.Tensor],
        z_L: Optional[torch.Tensor],
        inputs: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        default_stats = {
            "jacobian_norm": 0.0,
            "metric_tensor_norm": 0.0,
            "metric_tensor_trace": 0.0,
            "volume_distortion": 1.0,
            "log_volume_distortion": 0.0,
            "sylvester_log_det": 0.0,
            "metric_condition_number": 1.0,
        }

        # Check gradient attribution on inputs first if available
        if inputs is not None and isinstance(inputs, torch.Tensor) and inputs.grad is not None:
            g = inputs.grad.detach().float()
            g_norm = float(g.norm().item())
            default_stats["jacobian_norm"] = g_norm
            default_stats["volume_distortion"] = float(math.sqrt(1.0 + g_norm))

        if x_embed is None or z_L is None:
            return default_stats

        # Compute Jacobian via autograd if graph is alive
        try:
            if x_embed.requires_grad and z_L.grad_fn is not None:
                d_out = z_L.shape[-1]
                k = min(d_out, self.max_jacobian_dim)
                if k < d_out:
                    # Randomized subspace projection (Johnson-Lindenstrauss / Hutchinson)
                    # Scales projection so E[||J_proj||_F^2] = ||J||_F^2
                    proj = torch.randn(k, d_out, device=z_L.device, dtype=z_L.dtype) / math.sqrt(k)
                    J_rows = []
                    for i in range(k):
                        v = torch.zeros_like(z_L)
                        v[0, 0] = proj[i]
                        grads = torch.autograd.grad(z_L, x_embed, grad_outputs=v, retain_graph=True, allow_unused=True)[0]
                        if grads is not None:
                            J_rows.append(grads[0, 0].detach())
                        else:
                            J_rows.append(torch.zeros(x_embed.shape[-1], device=x_embed.device))
                    J = torch.stack(J_rows, dim=0)
                    return compute_riemannian_metric_tensor(J, eps=self.eps)
                else:
                    J_rows = []
                    for i in range(k):
                        v = torch.zeros_like(z_L)
                        v[0, 0, i] = 1.0
                        grads = torch.autograd.grad(z_L, x_embed, grad_outputs=v, retain_graph=True, allow_unused=True)[0]
                        if grads is not None:
                            J_rows.append(grads[0, 0].detach())
                        else:
                            J_rows.append(torch.zeros(x_embed.shape[-1], device=x_embed.device))
                    J = torch.stack(J_rows, dim=0)
                    return compute_riemannian_metric_tensor(J, eps=self.eps)
        except Exception:
            pass

        return default_stats


__all__ = [
    "RealTimeMathDecoder",
    "ManifoldMetrics",
    "ManifoldReport",
    "compute_dirichlet_energy",
    "compute_normalized_dirichlet_energy",
    "compute_fiedler_value",
    "compute_participation_ratio",
    "compute_spectral_entropy",
    "compute_cka",
    "compute_riemannian_metric_tensor",
    "decode_arch_latent_state",
]
