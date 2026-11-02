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

from collections import deque
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
    algebraic_connectivity: float          # lambda_min(Delta_F)
    spectral_gap: float
    harmonic_space_dim: int                # dim ker(Delta_F)
    is_harmonic_trivial: bool              # True iff harmonic_space_dim == 0 (lambda_min > eps)
    joint_fixed_subspace_dim: int          # dim intersection ker(H_gamma - I) over cycle basis
    holonomy_trace_mean: float             # Mean trace of cycle holonomies / dim
    transport_fit_error: float = 0.0       # Procrustes alignment residual error (train contexts)
    val_transport_error: float = 0.0       # Procrustes alignment residual error (held-out contexts)
    held_out_cycle_error: float = 0.0      # Cycle consistency residual error on fundamental cycles
    num_fundamental_cycles: int = 0        # Count of fundamental cycles in spanning forest basis


class CellularSheafLaplacian:
    """Cellular Sheaf Laplacian on Token Graphs (Bodnar et al. 2022).

    Let G = (V, E) be a token graph with stalks F(v) = R^d.
    For each edge e = (u, v), restriction maps F_{u <= e}, F_{v <= e} in O(d)
    govern parallel transport of latent features.

    Mathematical Invariant:
      The Sheaf Laplacian Delta_F = delta^T delta has Dirichlet energy:
        E_F(Z) = 1/2 <x, Delta_F x> = 1/2 sum_{e=(u,v)} ||F_{v <= e} x_v - F_{u <= e} x_u||_2^2 >= 0.

      Kernel Elimination Theorem (Refined):
        The claim that any nonidentity holonomy (H != I) eliminates the kernel is FALSE.
        For example, a non-trivial 3D rotation with a fixed axis leaves ker(H - I) 1-dimensional,
        retaining a non-zero zero-energy global section along the rotation axis.
        The correct necessary & sufficient condition for a cycle gamma to eliminate the cycle kernel is:
          ker(H_gamma - I) = {0}  <=>  1 not in spec(H_gamma).
        For the entire graph, the harmonic space satisfies:
          ker(Delta_F) = H^0(G; F) = intersection_{gamma in pi_1(G)} ker(H_gamma - I).
        Hence over-smoothing is eliminated iff the joint fixed-point subspace is trivial: {0}.

      Spanning-Tree Fundamental Cycle Basis:
        Rather than heuristic triangle search, a complete basis of pi_1(G) is constructed via
        a spanning forest T. Each chord e in E \\ T induces a fundamental cycle.
        Conjugating fundamental cycle holonomies to component roots yields an exact isomorphism:
          dim intersection_{chords} ker(H_{gamma_e}^{(r)} - I) == dim ker(Delta_F).
    """

    def __init__(self, num_nodes: int, dim: int, eps: float = 1e-8):
        self.num_nodes = num_nodes
        self.dim = dim
        self.eps = eps
        self.edges: List[Tuple[int, int]] = []
        # Stored restriction maps: dict mapping (u, v) -> (F_{u <= e}, F_{v <= e})
        self.restriction_maps: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.transport_error: float = 0.0
        self.val_transport_error: float = 0.0
        self.held_out_cycle_error: float = 0.0
        self.fundamental_cycles: List[Dict[str, Any]] = []

    def get_edge_transport(self, u: int, v: int) -> torch.Tensor:
        """Return the orthogonal transport matrix P_{u -> v} in O(d)."""
        if (u, v) in self.restriction_maps:
            R_u, R_v = self.restriction_maps[(u, v)]
            return R_v.T @ R_u
        elif (v, u) in self.restriction_maps:
            R_v, R_u = self.restriction_maps[(v, u)]
            return (R_u.T @ R_v).T
        else:
            raise KeyError(f"No edge between node {u} and node {v}")

    def learn_procrustes_transports(
        self,
        Z: torch.Tensor,
        edges: Optional[List[Tuple[int, int]]] = None,
        attn_matrix: Optional[torch.Tensor] = None,
        threshold: float = 0.05,
        val_ratio: float = 0.0,
    ) -> float:
        """Learn and validate orthogonal transport maps P_{u -> v} in O(d) via Orthogonal Procrustes.

        Given empirical token representations Z in R^{B x S x d} (or R^{S x d}):
        For directed edge e = (u, v), restriction maps are set to:
          R_u = P_{u -> v},  R_v = I_d.
        The coboundary residual on representations is:
          (delta Z)_e = R_v Z_v - R_u Z_u = Z_v - P_{u -> v} Z_u.

        Objective:
          P_{u -> v}^* = argmin_{P in O(d)} sum_{b=1}^B ||Z_v^{(b)} - P Z_u^{(b)}||_F^2.
        Let cross-covariance matrix M = sum_b (Z_v^{(b)})^T (Z_u^{(b)}) = Z_v^T Z_u in R^{d x d}.
        By the Orthogonal Procrustes Theorem (Schönemann 1966), with SVD M = U Sigma V^T:
          P_{u -> v}^* = U V^T in O(d).

        Orientation Invariant:
          Evaluating ||Z_v - P_{u -> v} Z_u||_F^2 yields residual error 0 when Z_v = R Z_u.
          Applying the reversed transpose P^T yields an error of 4.0 on 90-degree 3D rotations.

        Multi-Context & Held-Out Cycle Validation:
          - Uses all B context sequences to fit M across multi-sequence batches.
          - If val_ratio > 0 and B >= 2, splits contexts into train and held-out validation sets.
          - Evaluates cycle consistency residuals across fundamental cycles.
        """
        with torch.no_grad():
            Z_f = Z.detach().float()
            if Z_f.dim() == 2:
                Z_f = Z_f.unsqueeze(0)  # (1, S, d)
            B, S, D = Z_f.shape
            S = min(self.num_nodes, S)
            d = min(self.dim, D)
            Z_f = Z_f[:, :S, :d]

            if val_ratio > 0.0 and B >= 2:
                B_val = max(1, int(round(B * val_ratio)))
                B_train = max(1, B - B_val)
                Z_train = Z_f[:B_train]
                Z_val = Z_f[B_train:]
            else:
                B_train = B
                B_val = 0
                Z_train = Z_f
                Z_val = None

            self.edges.clear()
            self.restriction_maps.clear()

            if edges is not None:
                candidate_edges = edges
            elif attn_matrix is not None:
                A = attn_matrix.detach().float()
                if A.dim() == 4:
                    A = A[0].mean(dim=0)
                elif A.dim() == 3:
                    A = A[0]
                candidate_edges = []
                for i in range(S):
                    for j in range(i + 1, S):
                        w = 0.5 * (A[i, j] + A[j, i]).item()
                        if w >= threshold or abs(i - j) == 1:
                            candidate_edges.append((i, j))
            else:
                candidate_edges = [(i, i + 1) for i in range(S - 1)]

            total_err = 0.0
            val_total_err = 0.0
            count = 0

            for (u, v) in candidate_edges:
                if u >= S or v >= S:
                    continue
                self.edges.append((u, v))
                z_u = Z_train[:, u, :]  # (B_train, d)
                z_v = Z_train[:, v, :]  # (B_train, d)

                # Cross-covariance matrix M = sum_b z_v^{(b)} (z_u^{(b)})^T = z_v.T @ z_u
                M = z_v.T @ z_u  # (d, d)
                try:
                    U, _, Vh = torch.linalg.svd(M)
                    # Correctly oriented Orthogonal Procrustes solution: P_{u -> v} = U @ Vh
                    P_uv = U @ Vh
                except Exception:
                    P_uv = torch.eye(d, device=Z_f.device)

                R_u = P_uv
                R_v = torch.eye(d, device=Z_f.device)
                self.restriction_maps[(u, v)] = (R_u, R_v)

                # Correctly oriented residual: Z_v - Z_u @ P_uv^T  (shape B_train, d)
                diff_train = z_v - z_u @ P_uv.T
                total_err += float(diff_train.pow(2).sum().item()) / B_train
                count += 1

                if Z_val is not None:
                    z_u_val = Z_val[:, u, :]
                    z_v_val = Z_val[:, v, :]
                    diff_val = z_v_val - z_u_val @ P_uv.T
                    val_total_err += float(diff_val.pow(2).sum().item()) / B_val

            self.transport_error = total_err / max(1, count)
            self.val_transport_error = (val_total_err / max(1, count)) if Z_val is not None else self.transport_error

            # Validate cycle consistency
            cycles, _, _, held_err = self.extract_fundamental_cycles()
            self.held_out_cycle_error = held_err
            return self.transport_error

    def build_from_attention(
        self,
        attn_matrix: torch.Tensor,
        threshold: float = 0.05,
        rotation_scale: float = 0.1,
    ) -> None:
        """Construct sheaf topology with orthogonal restriction maps parameterized by attention weights."""
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
                    if weight >= threshold or abs(i - j) == 1:
                        self.edges.append((i, j))
                        # Generate orthogonal restriction maps in SO(d) via Lie algebra exp(K)
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
        """Compute Sheaf Dirichlet Energy E_F(Z) = 1/2 <x, Delta_F x> = 1/2 ||delta x||^2."""
        with torch.no_grad():
            Z_f = Z.detach().float()
            if Z_f.dim() == 2:
                Z_f = Z_f.unsqueeze(0)  # (1, S, d)
            B, S, D = Z_f.shape
            S = min(self.num_nodes, S)
            d = min(self.dim, D)
            Z_f = Z_f[:, :S, :d]

            total_energy = 0.0
            for b in range(B):
                energy = 0.0
                for (u, v), (R_u, R_v) in self.restriction_maps.items():
                    if u >= S or v >= S:
                        continue
                    diff = (R_v @ Z_f[b, v]) - (R_u @ Z_f[b, u])
                    energy += float(diff.pow(2).sum().item())

                energy *= 0.5
                if normalized:
                    f_norm_sq = float(Z_f[b].pow(2).sum().item())
                    energy = energy / (f_norm_sq + self.eps)
                total_energy += energy

            return float(total_energy / max(1, B))

    def extract_fundamental_cycles(self) -> Tuple[List[Dict[str, Any]], int, float, float]:
        """Extract a cycle basis of fundamental cycles using spanning trees on connected components.

        For each connected component C of G = (V, E):
          1. Constructs a BFS spanning tree T_C rooted at r = min(C).
          2. Identifies all chord edges e = (u, v) in E(C) \\ T_C.
          3. For each chord, builds the unique cycle: r --(tree)--> u --(chord)--> v --(tree)--> r.
          4. Computes holonomy conjugated to root r:
             H_{gamma_e}^{(r)} = P_{r -> v}^T P_{u -> v} P_{r -> u} in O(d).
          5. Stacks all (H_{gamma_e}^{(r)} - I) across all chords in component C.
             The joint fixed-point subspace on C is ker(stacked_mat), whose dimension
             is equal to dim H^0(C; F) = dim ker(Delta_F|_C).

        Returns:
            cycle_reports: list of dicts describing each fundamental cycle
            total_joint_fixed_dim: sum of fixed subspace dimensions across components
            mean_holonomy_trace: mean trace / d across all cycles
            held_out_cycle_error: mean ||H_gamma - I||_F^2 / d across cycles
        """
        N = self.num_nodes
        d = self.dim
        adj: Dict[int, List[int]] = {i: [] for i in range(N)}
        for (u, v) in self.restriction_maps.keys():
            if u < N and v < N:
                adj[u].append(v)
                adj[v].append(u)

        visited = set()
        total_joint_fixed_dim = 0
        all_cycle_reports: List[Dict[str, Any]] = []
        holonomy_traces: List[float] = []
        cycle_sq_errors: List[float] = []

        for root in range(N):
            if root in visited:
                continue

            # BFS spanning tree on connected component
            queue = deque([root])
            visited.add(root)
            parent: Dict[int, Optional[int]] = {root: None}
            tree_edges = set()
            chords: List[Tuple[int, int]] = []
            comp_nodes: List[int] = []

            while queue:
                curr = queue.popleft()
                comp_nodes.append(curr)
                for nbr in adj[curr]:
                    edge_key = (min(curr, nbr), max(curr, nbr))
                    if nbr not in visited:
                        visited.add(nbr)
                        parent[nbr] = curr
                        tree_edges.add(edge_key)
                        queue.append(nbr)
                    elif parent[curr] != nbr and edge_key not in tree_edges:
                        if edge_key not in chords:
                            chords.append(edge_key)

            def get_tree_path(node: int) -> List[int]:
                path = []
                curr = node
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1]

            def get_root_to_node_transport(node: int) -> torch.Tensor:
                path = get_tree_path(node)
                P = torch.eye(d)
                for i in range(len(path) - 1):
                    step_P = self.get_edge_transport(path[i], path[i + 1])
                    P = step_P @ P
                return P

            stacked_H_minus_I = []
            for (u, v) in chords:
                P_r_u = get_root_to_node_transport(u)
                P_u_v = self.get_edge_transport(u, v)
                P_r_v = get_root_to_node_transport(v)

                # Cycle holonomy conjugated to root r:
                H_root = P_r_v.T @ P_u_v @ P_r_u
                H_minus_I = H_root - torch.eye(d, device=H_root.device)
                stacked_H_minus_I.append(H_minus_I)

                tr = float(torch.trace(H_root).item()) / d
                holonomy_traces.append(tr)

                cycle_err = float(H_minus_I.pow(2).sum().item()) / d
                cycle_sq_errors.append(cycle_err)

                sv_c = torch.linalg.svdvals(H_minus_I)
                c_fixed_dim = int((sv_c < 1e-4).sum().item())

                all_cycle_reports.append({
                    "component_root": root,
                    "chord": (u, v),
                    "holonomy_trace": tr,
                    "fixed_subspace_dim": c_fixed_dim,
                    "cycle_error": cycle_err,
                })

            if stacked_H_minus_I:
                stacked_mat = torch.cat(stacked_H_minus_I, dim=0)  # (K*d, d)
                sv_joint = torch.linalg.svdvals(stacked_mat)
                comp_fixed = int((sv_joint < 1e-4).sum().item())
            else:
                # Tree or isolated vertex has dim H^0(T; F) = d
                comp_fixed = d

            total_joint_fixed_dim += comp_fixed

        mean_tr = float(sum(holonomy_traces) / len(holonomy_traces)) if holonomy_traces else 1.0
        held_err = float(sum(cycle_sq_errors) / len(cycle_sq_errors)) if cycle_sq_errors else 0.0
        self.fundamental_cycles = all_cycle_reports
        return all_cycle_reports, total_joint_fixed_dim, mean_tr, held_err

    def spectral_analysis(self, device: Any = "cpu") -> SheafSpectralReport:
        """Compute algebraic connectivity and verify elimination of harmonic triviality.

        Uses the complete fundamental cycle basis on the graph's spanning forest.
        """
        Delta = self.assemble_laplacian(device=device)
        try:
            eigs = torch.linalg.eigvalsh(Delta)
            eigs = torch.sort(eigs).values
            lambda_min = float(max(0.0, eigs[0].item()))
            gap = float((eigs[1] - eigs[0]).item()) if len(eigs) > 1 else 0.0
            null_dim = int((eigs < 1e-4).sum().item())
        except Exception:
            lambda_min = 0.0
            gap = 0.0
            null_dim = self.dim

        cycles, joint_fixed_dim, mean_tr, held_err = self.extract_fundamental_cycles()

        return SheafSpectralReport(
            num_nodes=self.num_nodes,
            dim=self.dim,
            num_edges=len(self.edges),
            dirichlet_energy=0.0,
            normalized_dirichlet_energy=0.0,
            algebraic_connectivity=lambda_min,
            spectral_gap=gap,
            harmonic_space_dim=null_dim,
            is_harmonic_trivial=(null_dim == 0),
            joint_fixed_subspace_dim=joint_fixed_dim,
            holonomy_trace_mean=mean_tr,
            transport_fit_error=self.transport_error,
            val_transport_error=self.val_transport_error,
            held_out_cycle_error=held_err,
            num_fundamental_cycles=len(cycles),
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
class RamanujanDiagnosticReport:
    """Ramanujan Expander Diagnostic and Domain Validation Report."""
    is_ramanujan: bool                      # True iff connected, regular, and ratio <= 1.0
    expansion_ratio: float                  # lambda_nontrivial / (2*sqrt(k - 1) / k)
    is_connected: bool                      # True iff multiplicity of lambda = 1.0 is exactly 1
    is_regular: bool                        # True iff degree CV < 0.05
    is_bipartite: bool                      # True iff multiplicity of lambda = -1.0 is >= 1
    num_components: int                     # Multiplicity of lambda = 1.0
    avg_degree: float                       # k_avg
    degree_cv: float                        # sigma_k / k_avg
    lambda_nontrivial: float                # max_{non-trivial} |lambda_i|
    alon_boppana_bound: float               # 2*sqrt(k - 1) / k
    domain_valid: bool                      # True iff graph satisfies domain invariants (connected & regular)
    domain_warning: Optional[str] = None    # Warning if domain invariants are violated


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
    is_rational_dependent: bool = False
    joint_character_magnitude: float = 0.0
    is_ramanujan: bool = False
    is_connected: bool = True
    is_regular: bool = True


class HarmonicTorusSpectrum:
    """Number-Theoretic Harmonic Analysis & Torus Geometry Engine.

    1. Computes Discrete Fourier Transform over cyclic group Z_p:
       F_k = (1 / sqrt(p)) * sum_{x=0}^{p-1} f(x) * exp(-2*pi*i*k*x / p).
    2. Analyzes RoPE irrational torus winding on T^{d/2}:
       phi(m) = (m*omega_1, ..., m*omega_{d/2}) mod 2*pi.
       Evaluates multi-index rational resonance spectra across all coordinate pairs.
    3. Evaluates Ramanujan expander spectral bounds with strict domain validation:
       connectedness, degree regularity, and bidirectional spectral tails.
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
        return_joint: bool = False,
    ) -> Union[Tuple[float, float], Tuple[float, float, float, bool]]:
        """Evaluate RoPE irrational winding dynamics on torus T^{d/2}.

        Computes:
        1. Marginal Weyl sum metric: W_N = (1 / (d/2)) sum_k |(1/N) sum_m exp(i*m*omega_k)|.
        2. Mean torus geodesic distance between consecutive position steps.
        3. Multi-Coordinate Joint Character Resonance Analysis:
           Searches all coordinate pairs (i, j) with 0 <= i < j < d/2 for integer and rational
           relations: c_j * omega_j - c_i * omega_i == 0.
           Computes joint character magnitude: max_{c != 0} |(1/N) sum_m exp(i*m*c^T omega)|.
           Detects rational confinement across arbitrary coordinate pairs (e.g. coordinates 1 and 3
           at dim=16 having 10*omega_3 - omega_1 == 0 with joint character magnitude exactly 1.0).
        """
        half_dim = dim // 2
        indices = torch.arange(half_dim, dtype=torch.float32)
        omega = 1.0 / (base ** (2.0 * indices / dim))  # (half_dim,)

        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        angles = positions @ omega.unsqueeze(0)  # (N, half_dim)

        # 1. Marginal Weyl sum: (1/N) sum_{m=1}^N exp(i * m * omega_k)
        complex_phases = torch.polar(torch.ones_like(angles), angles)  # (N, half_dim)
        weyl_sums = torch.abs(torch.mean(complex_phases, dim=0))  # (half_dim,)
        mean_weyl = float(torch.mean(weyl_sums).item())

        # 2. Geodesic distance on flat torus T^{half_dim} between pos 0 and pos 1:
        omega_mod = torch.remainder(omega, 2.0 * math.pi)
        wrapped_dist = torch.minimum(omega_mod, 2.0 * math.pi - omega_mod)
        torus_dist = float(torch.sqrt(torch.sum(wrapped_dist.pow(2))).item())

        # 3. Multi-Coordinate Joint Character Resonance Analysis
        is_rat_dep = False
        max_joint_char_mag = 0.0

        if half_dim >= 2:
            for i in range(half_dim):
                for j in range(i + 1, half_dim):
                    w_i = float(omega[i].item())
                    w_j = float(omega[j].item())
                    ratio = w_i / (w_j + 1e-12)

                    cand_pairs = []
                    ratio_int = round(ratio)
                    if 1 <= ratio_int <= 10000 and abs(ratio - ratio_int) < 1e-3:
                        cand_pairs.append((1, ratio_int))

                    # Test rational ratios p/q for q in 1..20
                    for q in range(1, 21):
                        p = round(q * ratio)
                        if 1 <= p <= 10000 and abs(q * ratio - p) < 1e-3:
                            cand_pairs.append((q, p))

                    for c_i, c_j in set(cand_pairs):
                        c_vec = torch.zeros(half_dim, dtype=torch.float32)
                        c_vec[i] = -float(c_i)
                        c_vec[j] = float(c_j)
                        scalar_freq = float(torch.sum(c_vec * omega).item())
                        joint_angle = positions * scalar_freq  # (N, 1)
                        joint_phase = torch.polar(torch.ones_like(joint_angle), joint_angle)
                        char_mag = float(torch.abs(torch.mean(joint_phase)).item())

                        if char_mag > max_joint_char_mag:
                            max_joint_char_mag = char_mag

                        if abs(scalar_freq) < 1e-4 or char_mag > 0.9:
                            is_rat_dep = True

        joint_char_mag = max_joint_char_mag
        if return_joint:
            return mean_weyl, torus_dist, joint_char_mag, is_rat_dep
        return mean_weyl, torus_dist

    @staticmethod
    def ramanujan_expansion_ratio(
        attn_matrix: torch.Tensor,
        return_report: bool = False,
        regularity_threshold: float = 0.05,
    ) -> Union[float, RamanujanDiagnosticReport]:
        """Compute Ramanujan graph expansion ratio rho with rigorous domain validation.

        Domain Invariants:
        1. Connectedness:
           For the normalized adjacency A_norm = D^{-1/2} A D^{-1/2}, the multiplicity of
           eigenvalue lambda = 1.0 is the number of connected components.
           If multiplicity > 1, the graph is disconnected, lambda_2 = 1.0, spectral gap = 0,
           and the graph strictly fails the Ramanujan expander property.
        2. Regularity:
           The Alon-Boppana theorem (Lubotzky et al. 1988) strictly applies to k-regular graphs.
           We evaluate degree coefficient of variation CV_k = sigma_k / k_avg.
           If CV_k >= regularity_threshold, the graph is irregular; average degree does NOT
           automatically validate the Ramanujan bound.
        3. Spectral Tails (Positive and Negative):
           Trivial eigenvalues: exactly ONE +1.0 (principal Perron-Frobenius eigenvalue),
           and if bipartite, ONE -1.0.
           The non-trivial eigenvalues are the remaining eigenvalues:
             lambda_nontrivial = max_{lambda in Spec_rem} |lambda|.
           If disconnected, remaining eigenvalues contain the other +1.0's, correctly yielding
           lambda_nontrivial = 1.0, which strictly violates the Ramanujan bound 2*sqrt(k-1)/k < 1.0.
        """
        with torch.no_grad():
            A = attn_matrix.detach().float()
            if A.dim() == 4:
                A = A[0].mean(dim=0)
            elif A.dim() == 3:
                A = A[0]
            S = A.shape[0]
            if S <= 2:
                report = RamanujanDiagnosticReport(
                    is_ramanujan=True,
                    expansion_ratio=1.0,
                    is_connected=True,
                    is_regular=True,
                    is_bipartite=False,
                    num_components=1,
                    avg_degree=1.0,
                    degree_cv=0.0,
                    lambda_nontrivial=0.0,
                    alon_boppana_bound=1.0,
                    domain_valid=True,
                )
                return report if return_report else 1.0

            # Symmetrize
            A_sym = 0.5 * (A + A.T)
            deg = torch.clamp_min(A_sym.sum(dim=-1), 1e-6)
            k_avg = float(deg.mean().item())
            k_std = float(deg.std().item())
            degree_cv = k_std / (k_avg + 1e-8)
            is_regular = degree_cv < regularity_threshold

            # Normalized adjacency A_norm = D^{-1/2} A D^{-1/2}
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
            A_norm = D_inv_sqrt @ A_sym @ D_inv_sqrt
            try:
                eigs = torch.linalg.eigvalsh(A_norm)
                eigs_sorted = torch.sort(eigs).values
            except Exception:
                eigs_sorted = torch.zeros(S)

            # Connectedness check: multiplicity of 1.0
            mult_one = int((torch.abs(eigs_sorted - 1.0) < 1e-4).sum().item())
            num_components = max(1, mult_one)
            is_connected = (mult_one == 1)

            # Bipartiteness check: multiplicity of -1.0
            mult_neg_one = int((torch.abs(eigs_sorted - (-1.0)) < 1e-4).sum().item())
            is_bipartite = (mult_neg_one >= 1)

            # Non-trivial eigenvalues:
            # Drop exactly ONE +1.0 (the principal trivial eigenvalue)
            # If bipartite, drop ONE -1.0 (the trivial bipartite eigenvalue)
            # All other eigenvalues (including duplicate 1.0s on disconnected graphs) remain!
            nontrivial_eigs = eigs_sorted[:-1]
            if is_bipartite and len(nontrivial_eigs) > 0:
                nontrivial_eigs = nontrivial_eigs[1:]

            if len(nontrivial_eigs) > 0:
                lambda_nontrivial = float(torch.max(torch.abs(nontrivial_eigs)).item())
            else:
                lambda_nontrivial = 0.0

            if k_avg <= 1.0:
                bound = 1.0
                ratio = 1.0
            else:
                bound = (2.0 * math.sqrt(max(0.0, k_avg - 1.0))) / k_avg
                ratio = lambda_nontrivial / (bound + 1e-8)

            domain_valid = is_connected and is_regular
            is_ramanujan = domain_valid and (ratio <= 1.0)

            warning = None
            if not is_connected:
                warning = f"Domain invalid: Graph is disconnected (num_components={num_components}, lambda_2=1.0). Alon-Boppana expander property fails."
            elif not is_regular:
                warning = f"Domain invalid: Degree distribution is irregular (CV={degree_cv:.4f} >= {regularity_threshold}). Standard regular-graph Ramanujan bound is heuristic."

            report = RamanujanDiagnosticReport(
                is_ramanujan=is_ramanujan,
                expansion_ratio=float(ratio),
                is_connected=is_connected,
                is_regular=is_regular,
                is_bipartite=is_bipartite,
                num_components=num_components,
                avg_degree=k_avg,
                degree_cv=degree_cv,
                lambda_nontrivial=lambda_nontrivial,
                alon_boppana_bound=bound,
                domain_valid=domain_valid,
                domain_warning=warning,
            )

            if return_report:
                return report
            return float(ratio)
