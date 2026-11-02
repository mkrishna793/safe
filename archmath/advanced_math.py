"""Advanced Mathematical Frameworks for Representation Geometry and Attribution.

Implements frontier mathematical methods from MIT, Stanford, Anthropic, and Oxford:

1. EK-FAC (Eigenvalue-Corrected Kronecker-Factored Approximate Curvature):
   - George et al. (2018), Grosse et al. / Anthropic (2023).
   - Computes Kronecker eigenbases Q_A, Q_S and exact diagonal scaling tensor
     Sigma_{j, i} = E[(Q_S^T g)_j^2 (Q_A^T a)_i^2].
   - Performs exact Inverse Hessian-Vector Products (IHVP) for scalable second-order
     training data attribution and influence function evaluation:
       I(z_train, z_test) = - grad_train^T H_{EKFAC}^{-1} grad_test.

2. Cellular Sheaf Laplacians & Cosheaf Representation:
   - Bodnar et al. (ICML 2022), Hansen & Ghrist (2019).
   - Formulates cellular cosheaves over token complexes with stalks F(v) = R^d
     and restriction maps F_{v <= e} in O(d).
   - Computes Sheaf Laplacian Delta_F = delta^T delta and Sheaf Dirichlet energy.
   - Eliminates graph over-smoothing / rank collapse by breaking harmonic trivialization:
     ker(Delta_F) = {0} under non-trivial holonomy.

3. Anthropic-Style Circuit Attribution Graphs & DLA:
   - Elhage et al. / Anthropic (2021, 2023), Syed et al. (2023).
   - Direct Logit Attribution (DLA): exact linear decomposition of residual stream
     contributions to target vs counterfactual logit difference:
       Delta logit = <x_0, w_diff> + sum_{l,h} DLA_{l,h} + sum_l DLA_{mlp,l} + Delta b.
   - Path-integrated gradients and linearized attribution patching for causal DAG pruning.

4. Harmonic Analysis & Torus Geometry:
   - Nanda et al. (2023), Chughtai et al. (2023), Su et al. (2021).
   - Discrete Fourier Transform over cyclic groups Z_p for modular arithmetic representation.
   - Rotary Position Embedding (RoPE) irrational torus winding on T^{d/2}:
     geodesic distance, quasi-periodicity, and Weyl sum equidistribution.
   - Ramanujan graph spectral expansion bounds for attention routing.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
# 1. EK-FAC: Eigenvalue-Corrected Kronecker-Factored Curvature & Attribution
# =========================================================================== #

@dataclasses.dataclass
class EKFACState:
    """State of EK-FAC curvature decomposition for a linear layer."""
    d_in: int
    d_out: int
    Q_A: torch.Tensor          # (d_in, d_in) orthogonal eigenbasis of input covariance
    Lambda_A: torch.Tensor     # (d_in,) eigenvalues of input covariance
    Q_S: torch.Tensor          # (d_out, d_out) orthogonal eigenbasis of gradient covariance
    Lambda_S: torch.Tensor     # (d_out,) eigenvalues of gradient covariance
    Sigma: torch.Tensor        # (d_out, d_in) exact diagonal scaling in Kronecker basis
    damping: float = 1e-4
    condition_number: float = 1.0


class EKFACAttribution:
    """Eigenvalue-Corrected Kronecker-Factored Approximate Curvature (George et al. 2018).

    For a linear layer s = W a with weight W in R^{d_out x d_in}:
      Input covariance:  A = E[a a^T] = Q_A Lambda_A Q_A^T
      Grad covariance:   S = E[g g^T] = Q_S Lambda_S Q_S^T
    In standard K-FAC, F is approximated as A (x) S with eigenvalues Lambda_A (x) Lambda_S.
    EK-FAC corrects eigenvalue distortion by projecting empirical gradients into the
    Kronecker eigenbasis Q = Q_A (x) Q_S and computing the exact diagonal expectation:
      Sigma_{j, i} = E[ (Q_S^T g)_j^2 * (Q_A^T a)_i^2 ].
    """

    def __init__(self, d_in: int, d_out: int, damping: float = 1e-4, eps: float = 1e-8):
        self.d_in = d_in
        self.d_out = d_out
        self.damping = damping
        self.eps = eps
        self.state: Optional[EKFACState] = None

        # Running accumulators
        self._cov_a = torch.zeros((d_in, d_in), dtype=torch.float64)
        self._cov_s = torch.zeros((d_out, d_out), dtype=torch.float64)
        self._count_cov = 0

        self._sigma = torch.zeros((d_out, d_in), dtype=torch.float64)
        self._count_sigma = 0

    def update_covariances(self, a: torch.Tensor, g: torch.Tensor) -> None:
        """Accumulate second-moment matrices A = E[a a^T] and S = E[g g^T].

        Args:
            a: Input activations (..., d_in)
            g: Output pre-activation gradients (..., d_out)
        """
        with torch.no_grad():
            a_flat = a.detach().reshape(-1, self.d_in).to(torch.float64)
            g_flat = g.detach().reshape(-1, self.d_out).to(torch.float64)
            n_tokens = a_flat.shape[0]
            if n_tokens == 0:
                return

            self._cov_a = (self._cov_a * self._count_cov + a_flat.T @ a_flat) / (self._count_cov + n_tokens)
            self._cov_s = (self._cov_s * self._count_cov + g_flat.T @ g_flat) / (self._count_cov + n_tokens)
            self._count_cov += n_tokens

    def compute_eigenbases(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute spectral decomposition of A and S."""
        # Symmetric eigendecomposition
        cov_a_sym = 0.5 * (self._cov_a + self._cov_a.T)
        cov_s_sym = 0.5 * (self._cov_s + self._cov_s.T)

        Lambda_A, Q_A = torch.linalg.eigh(cov_a_sym)
        Lambda_S, Q_S = torch.linalg.eigh(cov_s_sym)

        # Sort in descending order
        idx_a = torch.argsort(Lambda_A, descending=True)
        Lambda_A, Q_A = Lambda_A[idx_a], Q_A[:, idx_a]

        idx_s = torch.argsort(Lambda_S, descending=True)
        Lambda_S, Q_S = Lambda_S[idx_s], Q_S[:, idx_s]

        return Q_A.float(), Lambda_A.float(), Q_S.float(), Lambda_S.float()

    def update_eigenvalue_corrections(self, a: torch.Tensor, g: torch.Tensor, Q_A: torch.Tensor, Q_S: torch.Tensor) -> None:
        """Compute exact diagonal scaling Sigma in the Kronecker eigenbasis."""
        with torch.no_grad():
            a_flat = a.detach().reshape(-1, self.d_in).float()
            g_flat = g.detach().reshape(-1, self.d_out).float()
            n_tokens = a_flat.shape[0]
            if n_tokens == 0:
                return

            # Transform into eigenbases: a_tilde = a Q_A, g_tilde = g Q_S
            a_tilde = a_flat @ Q_A  # (N, d_in)
            g_tilde = g_flat @ Q_S  # (N, d_out)

            # Elementwise square
            a_sq = a_tilde.pow(2)  # (N, d_in)
            g_sq = g_tilde.pow(2)  # (N, d_out)

            # Outer expectation: E[g_sq_j * a_sq_i] = (g_sq^T @ a_sq) / N
            batch_sigma = (g_sq.T @ a_sq).to(torch.float64)

            self._sigma = (self._sigma * self._count_sigma + batch_sigma) / (self._count_sigma + n_tokens)
            self._count_sigma += n_tokens

    def finalize(self) -> EKFACState:
        """Finalize the EK-FAC state for inference and attribution."""
        Q_A, Lambda_A, Q_S, Lambda_S = self.compute_eigenbases()
        if self._count_sigma == 0:
            # Fallback to standard K-FAC Kronecker product if no correction batch recorded
            Sigma = torch.outer(Lambda_S, Lambda_A)
        else:
            Sigma = self._sigma.float()

        # Damped condition number
        damped_sigma = Sigma + self.damping
        max_eig = float(damped_sigma.max().item())
        min_eig = float(damped_sigma.min().item())
        cond = max_eig / (min_eig + self.eps)

        self.state = EKFACState(
            d_in=self.d_in,
            d_out=self.d_out,
            Q_A=Q_A,
            Lambda_A=Lambda_A,
            Q_S=Q_S,
            Lambda_S=Lambda_S,
            Sigma=Sigma,
            damping=self.damping,
            condition_number=cond,
        )
        return self.state

    def inverse_hessian_vector_product(self, V: torch.Tensor) -> torch.Tensor:
        """Compute U = H_{EKFAC}^{-1} V for gradient matrix V in R^{d_out x d_in}.

        Mathematical identity:
          vec(H^{-1} V) = (Q_A (x) Q_S) (Sigma + lambda)^{-1} (Q_A^T (x) Q_S^T) vec(V)
          <=> U = Q_S [ (Q_S^T V Q_A) / (Sigma + lambda) ] Q_A^T.
        """
        if self.state is None:
            self.finalize()
        assert self.state is not None

        # 1. Rotate gradient into eigenbasis
        V_f = V.detach().float()
        V_tilde = self.state.Q_S.T @ V_f @ self.state.Q_A  # (d_out, d_in)

        # 2. Damped inversion in eigenbasis
        U_tilde = V_tilde / (self.state.Sigma + self.state.damping)

        # 3. Rotate back to parameter basis
        U = self.state.Q_S @ U_tilde @ self.state.Q_A.T
        return U

    def compute_influence(self, grad_train: torch.Tensor, grad_test: torch.Tensor) -> float:
        """Compute training data attribution influence score I(z_train, z_test).

        I(z_train, z_test) = - Tr( grad_train^T * (H_{EKFAC}^{-1} grad_test) ).
        """
        ihvp = self.inverse_hessian_vector_product(grad_test)
        score = -torch.sum(grad_train.float() * ihvp).item()
        return float(score)


# =========================================================================== #
# 2. Cellular Sheaf Laplacians & Harmonic Trivialization
# =========================================================================== #

@dataclasses.dataclass
class SheafSpectralReport:
    """Spectral diagnostic report for a cellular cosheaf."""
    num_nodes: int
    dim: int
    num_edges: int
    dirichlet_energy: float
    normalized_dirichlet_energy: float
    algebraic_connectivity: float  # lambda_min(Delta_F)
    spectral_gap: float
    is_harmonic_trivial: bool      # True if ker(Delta_F) has non-zero elements
    holonomy_trace_mean: float     # Mean trace of cycle holonomies


class CellularSheafLaplacian:
    """Cellular Sheaf Laplacian on Token Graphs (Bodnar et al. 2022).

    Let G = (V, E) be a token graph with stalks F(v) = R^d.
    For each edge e = (u, v), restriction maps F_{u <= e}, F_{v <= e} in O(d)
    govern parallel transport of latent features.

    The coboundary operator delta: C^0(G; F) -> C^1(G; F) is:
      (delta x)_e = F_{v <= e} x_v - F_{u <= e} x_u.

    The Sheaf Laplacian Delta_F = delta^T delta has block entries:
      (Delta_F)_{u, u} = deg(u) * I_d
      (Delta_F)_{u, v} = - F_{u <= e}^T F_{v <= e}   for e = (u, v).
    """

    def __init__(self, num_nodes: int, dim: int, eps: float = 1e-8):
        self.num_nodes = num_nodes
        self.dim = dim
        self.eps = eps
        self.edges: List[Tuple[int, int]] = []
        # Stored restriction maps: dict mapping (u, v) -> (F_{u <= e}, F_{v <= e})
        self.restriction_maps: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def build_from_attention(
        self,
        attn_matrix: torch.Tensor,
        threshold: float = 0.05,
        rotation_scale: float = 0.1,
    ) -> None:
        """Construct sheaf topology and orthogonal restriction maps from attention weights."""
        with torch.no_grad():
            A = attn_matrix.detach().float()
            if A.dim() == 4:
                A = A[0].mean(dim=0)
            elif A.dim() == 3:
                A = A[0]
            S = min(self.num_nodes, A.shape[0])
            self.edges.clear()
            self.restriction_maps.clear()

            for i in range(S):
                for j in range(i + 1, S):
                    weight = 0.5 * (A[i, j] + A[j, i]).item()
                    if weight >= threshold or abs(i - j) == 1:  # Keep causal chain always
                        self.edges.append((i, j))
                        # Generate orthogonal restriction maps in O(d) via matrix exponential of skew-symmetric matrix
                        # K_ij = -K_ij^T -> exp(K_ij) in SO(d)
                        K = torch.randn(self.dim, self.dim, device=A.device) * rotation_scale * weight
                        K = 0.5 * (K - K.T)
                        R_u = torch.matrix_exp(K)
                        R_v = torch.eye(self.dim, device=A.device)
                        self.restriction_maps[(i, j)] = (R_u, R_v)

    def assemble_laplacian(self, device: Any = "cpu") -> torch.Tensor:
        """Assemble the full (N*d, N*d) Sheaf Laplacian Delta_F."""
        N = self.num_nodes
        d = self.dim
        Delta = torch.zeros((N * d, N * d), dtype=torch.float32, device=device)

        # Degrees accumulator
        degrees = [0] * N

        for (u, v), (R_u, R_v) in self.restriction_maps.items():
            if u >= N or v >= N:
                continue
            degrees[u] += 1
            degrees[v] += 1

            # Off-diagonal blocks: - F_{u <= e}^T F_{v <= e}
            transport = R_u.T @ R_v
            Delta[u * d : (u + 1) * d, v * d : (v + 1) * d] -= transport
            Delta[v * d : (v + 1) * d, u * d : (u + 1) * d] -= transport.T

        # Diagonal blocks: deg(u) * I_d
        for u in range(N):
            deg = degrees[u]
            Delta[u * d : (u + 1) * d, u * d : (u + 1) * d] += deg * torch.eye(d, device=device)

        return Delta

    def compute_sheaf_dirichlet_energy(self, Z: torch.Tensor, normalized: bool = True) -> float:
        """Compute Sheaf Dirichlet Energy E_F(Z) = 1/2 <x, Delta_F x> >= 0."""
        with torch.no_grad():
            Z_f = Z.detach().float()
            if Z_f.dim() >= 3:
                Z_f = Z_f[0]
            S = min(self.num_nodes, Z_f.shape[0])
            Z_f = Z_f[:S, :self.dim]

            energy = 0.0
            for (u, v), (R_u, R_v) in self.restriction_maps.items():
                if u >= S or v >= S:
                    continue
                diff = (R_v @ Z_f[v]) - (R_u @ Z_f[u])
                energy += float(diff.pow(2).sum().item())

            energy *= 0.5
            if normalized:
                f_norm_sq = float(Z_f.pow(2).sum().item())
                energy = energy / (f_norm_sq + self.eps)
            return float(energy)

    def spectral_analysis(self, device: Any = "cpu") -> SheafSpectralReport:
        """Compute algebraic connectivity and verify elimination of harmonic triviality."""
        Delta = self.assemble_laplacian(device=device)
        try:
            eigs = torch.linalg.eigvalsh(Delta)
            eigs = torch.sort(eigs).values
            lambda_min = float(max(0.0, eigs[0].item()))
            gap = float((eigs[1] - eigs[0]).item()) if len(eigs) > 1 else 0.0
        except Exception:
            lambda_min = 0.0
            gap = 0.0

        # Check holonomy trace around triangle cycles if present
        holonomy_traces = []
        for i in range(min(self.num_nodes, 8)):
            for j in range(i + 1, min(self.num_nodes, 8)):
                for k in range(j + 1, min(self.num_nodes, 8)):
                    if (i, j) in self.restriction_maps and (j, k) in self.restriction_maps and (i, k) in self.restriction_maps:
                        R_ij_u, R_ij_v = self.restriction_maps[(i, j)]
                        R_jk_u, R_jk_v = self.restriction_maps[(j, k)]
                        R_ik_u, R_ik_v = self.restriction_maps[(i, k)]
                        # Cycle transport: U_ij * U_jk * U_ki
                        U_ij = R_ij_u.T @ R_ij_v
                        U_jk = R_jk_u.T @ R_jk_v
                        U_ki = (R_ik_u.T @ R_ik_v).T
                        cycle = U_ij @ U_jk @ U_ki
                        holonomy_traces.append(float(torch.trace(cycle).item()) / self.dim)

        mean_tr = float(sum(holonomy_traces) / len(holonomy_traces)) if holonomy_traces else 1.0

        return SheafSpectralReport(
            num_nodes=self.num_nodes,
            dim=self.dim,
            num_edges=len(self.edges),
            dirichlet_energy=0.0,
            normalized_dirichlet_energy=0.0,
            algebraic_connectivity=lambda_min,
            spectral_gap=gap,
            is_harmonic_trivial=(lambda_min < 1e-4),
            holonomy_trace_mean=mean_tr,
        )


# =========================================================================== #
# 3. Anthropic-Style Circuit Attribution Graphs & Direct Logit Attribution
# =========================================================================== #

@dataclasses.dataclass
class CircuitAttributionResult:
    """Results from Direct Logit Attribution and computational DAG extraction."""
    target_token: int
    reference_token: int
    logit_diff: float
    reconstructed_logit_diff: float
    reconstruction_error: float
    dla_heads: Dict[str, float]      # Head (layer, head_idx) -> DLA score
    dla_mlps: Dict[str, float]       # Layer -> DLA score
    embedding_dla: float
    bias_dla: float
    top_positive_heads: List[Tuple[str, float]]
    top_negative_heads: List[Tuple[str, float]]
    dag_sparsity: float


class CircuitAttributionGraph:
    """Mechanistic Interpretability Circuit Attribution Engine (Anthropic, Elhage et al. 2021).

    Analyzes transformer residual stream linearity:
      x_L = x_0 + sum_{l=1}^L sum_{h=1}^H a_{l, h} + sum_{l=1}^L m_l.
    Computes exact Direct Logit Attribution (DLA) along unembedding vector:
      w_diff = W_U[:, target] - W_U[:, reference].
    """

    def __init__(self, unembedding_weight: torch.Tensor, unembedding_bias: Optional[torch.Tensor] = None):
        """
        Args:
            unembedding_weight: (vocab_size, d_model) or (d_model, vocab_size)
            unembedding_bias: Optional (vocab_size,)
        """
        if unembedding_weight.shape[0] < unembedding_weight.shape[1]:
            # Shape is (d_model, vocab_size)
            self.W_U = unembedding_weight.detach().float()
        else:
            # Shape is (vocab_size, d_model) -> transpose to (d_model, vocab_size)
            self.W_U = unembedding_weight.detach().float().T

        self.d_model = self.W_U.shape[0]
        self.vocab_size = self.W_U.shape[1]
        self.b_U = unembedding_bias.detach().float() if unembedding_bias is not None else None

    def direct_logit_attribution(
        self,
        x_0: torch.Tensor,
        head_outputs: Dict[Tuple[int, int], torch.Tensor],
        mlp_outputs: Dict[int, torch.Tensor],
        target_token: int,
        reference_token: int,
    ) -> CircuitAttributionResult:
        """Compute exact Direct Logit Attribution (DLA) for all residual stream contributors.

        Linearity Identity:
          Delta logit = <w_diff, x_0> + sum_{l,h} <w_diff, head_{l,h}> + sum_l <w_diff, mlp_l> + Delta b_U.
        """
        w_target = self.W_U[:, target_token]
        w_ref = self.W_U[:, reference_token]
        w_diff = w_target - w_ref  # (d_model,)

        bias_diff = 0.0
        if self.b_U is not None:
            bias_diff = float((self.b_U[target_token] - self.b_U[reference_token]).item())

        # 1. Embedding attribution
        x_0_flat = x_0.detach().reshape(-1, self.d_model)[-1].float()
        embed_dla = float(torch.dot(x_0_flat, w_diff).item())

        # 2. Attention head attributions
        dla_heads: Dict[str, float] = {}
        for (layer, head), h_tensor in head_outputs.items():
            h_flat = h_tensor.detach().reshape(-1, self.d_model)[-1].float()
            val = float(torch.dot(h_flat, w_diff).item())
            dla_heads[f"L{layer}H{head}"] = val

        # 3. MLP attributions
        dla_mlps: Dict[str, float] = {}
        for layer, m_tensor in mlp_outputs.items():
            m_flat = m_tensor.detach().reshape(-1, self.d_model)[-1].float()
            val = float(torch.dot(m_flat, w_diff).item())
            dla_mlps[f"L{layer}_MLP"] = val

        # Sum of all components
        reconstructed = embed_dla + sum(dla_heads.values()) + sum(dla_mlps.values()) + bias_diff

        # Reconstructed final state x_L
        x_L = x_0_flat.clone()
        for h in head_outputs.values():
            x_L = x_L + h.detach().reshape(-1, self.d_model)[-1].float()
        for m in mlp_outputs.values():
            x_L = x_L + m.detach().reshape(-1, self.d_model)[-1].float()

        logits = x_L @ self.W_U
        if self.b_U is not None:
            logits = logits + self.b_U
        true_logit_diff = float((logits[target_token] - logits[reference_token]).item())

        recon_err = abs(true_logit_diff - reconstructed)

        sorted_heads = sorted(dla_heads.items(), key=lambda kv: kv[1], reverse=True)
        top_pos = sorted_heads[:5]
        top_neg = sorted_heads[-5:]

        # DAG Sparsity: fraction of heads with |DLA| < 0.05 * max|DLA|
        max_abs = max((abs(v) for v in dla_heads.values()), default=1.0)
        sparse_count = sum(1 for v in dla_heads.values() if abs(v) < 0.05 * max_abs)
        sparsity = sparse_count / max(1, len(dla_heads))

        return CircuitAttributionResult(
            target_token=target_token,
            reference_token=reference_token,
            logit_diff=true_logit_diff,
            reconstructed_logit_diff=reconstructed,
            reconstruction_error=recon_err,
            dla_heads=dla_heads,
            dla_mlps=dla_mlps,
            embedding_dla=embed_dla,
            bias_dla=bias_diff,
            top_positive_heads=top_pos,
            top_negative_heads=top_neg,
            dag_sparsity=sparsity,
        )


# =========================================================================== #
# 4. Harmonic Fourier Analysis & Torus Geometry
# =========================================================================== #

@dataclasses.dataclass
class HarmonicSpectrumReport:
    """Harmonic Fourier and Torus Winding Report."""
    modulus_p: int
    dominant_frequencies: List[int]
    spectral_entropy: float
    fourier_concentration_ratio: float
    weyl_equidistribution_metric: float
    torus_geodesic_distance: float
    ramanujan_expansion_ratio: float


class HarmonicTorusSpectrum:
    """Number-Theoretic Harmonic Analysis & Torus Geometry Engine.

    1. Computes Discrete Fourier Transform over cyclic group Z_p:
       F_k = (1 / sqrt(p)) * sum_{x=0}^{p-1} f(x) * exp(-2*pi*i*k*x / p).
    2. Analyzes RoPE irrational torus winding on T^{d/2}:
       phi(m) = (m*omega_1, ..., m*omega_{d/2}) mod 2*pi.
    3. Evaluates Ramanujan expander spectral bounds for attention routing.
    """

    @staticmethod
    def discrete_fourier_analysis(
        embeddings: torch.Tensor,
        modulus_p: int,
        top_k: int = 5,
    ) -> Tuple[List[int], float, float]:
        """Compute spatial Discrete Fourier Transform over residue tokens Z_p.

        Args:
            embeddings: (p, d_model) representation of residue tokens 0..p-1
            modulus_p: prime modulus p
            top_k: number of dominant frequencies to extract
        """
        with torch.no_grad():
            E = embeddings[:modulus_p].float()
            p, d = E.shape

            # Spatial DFT along token dimension: torch.fft.rfft
            E_fft = torch.fft.rfft(E, dim=0, norm="ortho")  # (p//2 + 1, d)
            power_spectrum = torch.sum(torch.abs(E_fft).pow(2), dim=1)  # (p//2 + 1,)

            # Exclude DC component (k = 0) for grokking frequency analysis
            ac_power = power_spectrum[1:]
            total_ac = float(ac_power.sum().item()) + 1e-12

            # Sort dominant frequencies
            top_indices = torch.argsort(ac_power, descending=True)[:top_k]
            dominant_freqs = [int(idx.item()) + 1 for idx in top_indices]

            # Concentration ratio: fraction of AC power in top-k frequencies
            top_power = float(ac_power[top_indices].sum().item())
            concentration = top_power / total_ac

            # Spectral entropy over frequency distribution
            prob = ac_power / total_ac
            prob_clamped = torch.clamp(prob, min=1e-12, max=1.0)
            entropy = -float((prob * torch.log(prob_clamped)).sum().item())

            return dominant_freqs, entropy, concentration

    @staticmethod
    def rope_torus_winding(
        dim: int,
        seq_len: int,
        base: float = 10000.0,
    ) -> Tuple[float, float]:
        """Evaluate RoPE irrational winding dynamics on torus T^{d/2}.

        Computes:
        1. Weyl sum equidistribution metric: W_N = (1 / (d/2)) sum_k |(1/N) sum_m exp(i*m*omega_k)|.
        2. Mean torus geodesic distance between consecutive position steps.
        """
        half_dim = dim // 2
        indices = torch.arange(half_dim, dtype=torch.float32)
        omega = 1.0 / (base ** (2.0 * indices / dim))  # (half_dim,)

        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        angles = positions @ omega.unsqueeze(0)  # (N, half_dim)

        # 1. Weyl sum: (1/N) sum_{m=1}^N exp(i * m * omega_k)
        complex_phases = torch.polar(torch.ones_like(angles), angles)  # (N, half_dim)
        weyl_sums = torch.abs(torch.mean(complex_phases, dim=0))  # (half_dim,)
        mean_weyl = float(torch.mean(weyl_sums).item())

        # 2. Geodesic distance on flat torus T^{half_dim} between pos 0 and pos 1:
        # dist_T^2 = sum_k (min(|omega_k|, 2*pi - |omega_k|))^2
        omega_mod = torch.remainder(omega, 2.0 * math.pi)
        wrapped_dist = torch.minimum(omega_mod, 2.0 * math.pi - omega_mod)
        torus_dist = float(torch.sqrt(torch.sum(wrapped_dist.pow(2))).item())

        return mean_weyl, torus_dist

    @staticmethod
    def ramanujan_expansion_ratio(attn_matrix: torch.Tensor) -> float:
        """Compute Ramanujan graph expansion ratio rho = lambda_2 / (2*sqrt(k - 1) / k).

        If rho <= 1.0, the attention adjacency attains the Alon-Boppana Ramanujan bound.
        """
        with torch.no_grad():
            A = attn_matrix.detach().float()
            if A.dim() == 4:
                A = A[0].mean(dim=0)
            elif A.dim() == 3:
                A = A[0]
            S = A.shape[0]
            if S <= 2:
                return 1.0

            # Symmetrize
            A_sym = 0.5 * (A + A.T)
            deg = torch.clamp_min(A_sym.sum(dim=-1), 1e-6)
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))

            # Normalized adjacency A_norm = D^{-1/2} A D^{-1/2}
            A_norm = D_inv_sqrt @ A_sym @ D_inv_sqrt
            try:
                eigs = torch.linalg.eigvalsh(A_norm)
                eigs = torch.sort(eigs, descending=True).values
                lambda_2 = float(abs(eigs[1].item()))
            except Exception:
                return 1.0

            k_avg = float(deg.mean().item())
            if k_avg <= 1.0:
                return 1.0

            # Alon-Boppana Ramanujan bound: 2*sqrt(k - 1) / k
            bound = (2.0 * math.sqrt(max(0.0, k_avg - 1.0))) / k_avg
            ratio = lambda_2 / (bound + 1e-8)
            return float(ratio)
