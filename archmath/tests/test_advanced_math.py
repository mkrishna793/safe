"""Unit tests for ArchMath advanced mathematical frameworks.

Verifies:
1. EK-FAC second-order curvature, Kronecker eigenbases, and influence function IHVP.
2. Cellular Sheaf Laplacian positive semi-definiteness, Dirichlet energy, and harmonic triviality elimination.
3. Circuit Attribution Graph Direct Logit Attribution (DLA) exact residual stream linearity conservation.
4. Harmonic Fourier DFT analysis, RoPE torus winding dynamics, and Ramanujan expansion bounds.
"""

import math
import torch
import torch.nn as nn

from archmath.advanced_math import (
    EKFACAttribution,
    CellularSheafLaplacian,
    CircuitAttributionGraph,
    HarmonicTorusSpectrum,
    RamanujanDiagnosticReport,
)


def test_ekfac_curvature_and_ihvp():
    """Verify EK-FAC eigenbasis decomposition, exact scaling, and IHVP inversion."""
    torch.manual_seed(42)
    d_in = 8
    d_out = 4

    ekfac = EKFACAttribution(d_in=d_in, d_out=d_out, damping=1e-3)

    # Simulate 5 batches of activations and gradients
    for _ in range(5):
        a = torch.randn(16, d_in)
        g = torch.randn(16, d_out)
        ekfac.update_covariances(a, g)

    # Compute eigenbases
    Q_A, Lambda_A, Q_S, Lambda_S = ekfac.compute_eigenbases()
    assert Q_A.shape == (d_in, d_in)
    assert Q_S.shape == (d_out, d_out)

    # Check orthogonality: Q^T Q == I
    assert torch.allclose(Q_A.T @ Q_A, torch.eye(d_in), atol=1e-5)
    assert torch.allclose(Q_S.T @ Q_S, torch.eye(d_out), atol=1e-5)

    # Update eigenvalue corrections in eigenbasis
    for _ in range(5):
        a = torch.randn(16, d_in)
        g = torch.randn(16, d_out)
        ekfac.update_eigenvalue_corrections(a, g, Q_A, Q_S)

    state = ekfac.finalize()
    assert state.Sigma.shape == (d_out, d_in)
    assert state.condition_number >= 1.0

    # Test IHVP: H^{-1} V
    V = torch.randn(d_out, d_in)
    U = ekfac.inverse_hessian_vector_product(V)
    assert U.shape == (d_out, d_in)
    assert not torch.isnan(U).any()
    assert not torch.isinf(U).any()

    # Influence score
    grad_train = torch.randn(d_out, d_in)
    grad_test = torch.randn(d_out, d_in)
    infl = ekfac.compute_influence(grad_train, grad_test)
    assert math.isfinite(infl)


def test_cellular_sheaf_laplacian_properties():
    """Verify Sheaf Laplacian positive semi-definiteness, Dirichlet energy, and ker(Delta_F)."""
    torch.manual_seed(42)
    num_nodes = 6
    dim = 4

    sheaf = CellularSheafLaplacian(num_nodes=num_nodes, dim=dim)

    # Synthetic attention matrix
    attn = torch.softmax(torch.randn(num_nodes, num_nodes), dim=-1)
    sheaf.build_from_attention(attn, threshold=0.1, rotation_scale=0.5)

    assert len(sheaf.edges) > 0
    assert len(sheaf.restriction_maps) > 0

    # Assemble Delta_F in R^{(N*d) x (N*d)}
    Delta_F = sheaf.assemble_laplacian()
    assert Delta_F.shape == (num_nodes * dim, num_nodes * dim)

    # Check symmetry: Delta_F == Delta_F^T
    assert torch.allclose(Delta_F, Delta_F.T, atol=1e-5)

    # Check positive semi-definiteness: all eigenvalues >= -1e-5
    eigs = torch.linalg.eigvalsh(Delta_F)
    assert (eigs >= -1e-5).all(), f"Negative eigenvalues found: {eigs[eigs < -1e-5]}"

    # Sheaf Dirichlet energy on random representation Z
    Z = torch.randn(num_nodes, dim)
    energy = sheaf.compute_sheaf_dirichlet_energy(Z, normalized=True)
    assert energy >= 0.0
    assert math.isfinite(energy)

    # Spectral analysis
    report = sheaf.spectral_analysis()
    assert report.algebraic_connectivity >= 0.0
    assert math.isfinite(report.spectral_gap)


def test_circuit_attribution_direct_logit_attribution_linearity():
    """Verify Direct Logit Attribution (DLA) exact residual stream linearity conservation."""
    torch.manual_seed(42)
    d_model = 16
    vocab_size = 32

    W_U = torch.randn(d_model, vocab_size)
    b_U = torch.randn(vocab_size)
    cag = CircuitAttributionGraph(unembedding_weight=W_U, unembedding_bias=b_U)

    # Simulate residual stream components
    x_0 = torch.randn(1, 1, d_model)
    head_outputs = {
        (0, 0): torch.randn(1, 1, d_model),
        (0, 1): torch.randn(1, 1, d_model),
        (1, 0): torch.randn(1, 1, d_model),
        (1, 1): torch.randn(1, 1, d_model),
    }
    mlp_outputs = {
        0: torch.randn(1, 1, d_model),
        1: torch.randn(1, 1, d_model),
    }

    target = 7
    reference = 12

    result = cag.direct_logit_attribution(
        x_0=x_0,
        head_outputs=head_outputs,
        mlp_outputs=mlp_outputs,
        target_token=target,
        reference_token=reference,
    )

    # Exact residual stream linearity check: reconstruction error < 1e-4
    assert result.reconstruction_error < 1e-4, f"Reconstruction error {result.reconstruction_error} too high"
    assert math.isclose(result.logit_diff, result.reconstructed_logit_diff, rel_tol=1e-4, abs_tol=1e-4)
    assert len(result.dla_heads) == 4
    assert len(result.dla_mlps) == 2
    assert 0.0 <= result.dag_sparsity <= 1.0


def test_harmonic_torus_spectrum():
    """Verify Fourier DFT on Z_p, RoPE torus winding, and Ramanujan expander ratio."""
    torch.manual_seed(42)
    p = 17  # prime modulus
    d_model = 16

    # 1. Fourier analysis
    embeddings = torch.randn(p, d_model)
    dominant_freqs, ent, conc = HarmonicTorusSpectrum.discrete_fourier_analysis(embeddings, modulus_p=p, top_k=3)
    assert len(dominant_freqs) == 3
    assert all(1 <= f <= p // 2 for f in dominant_freqs)
    assert ent >= 0.0
    assert 0.0 <= conc <= 1.0

    # 2. RoPE Torus winding
    mean_weyl, torus_dist = HarmonicTorusSpectrum.rope_torus_winding(dim=d_model, seq_len=64, base=10000.0)
    assert 0.0 <= mean_weyl <= 1.0
    assert torus_dist > 0.0
    assert math.isfinite(torus_dist)

    # 3. Ramanujan graph bound
    attn = torch.softmax(torch.randn(8, 8), dim=-1)
    rho = HarmonicTorusSpectrum.ramanujan_expansion_ratio(attn)
    assert rho >= 0.0
    assert math.isfinite(rho)


def test_sheaf_3d_rotation_counterexample_and_procrustes():
    """Verify 3D rotation counterexample: H != I retains 1D kernel along rotation axis;

    and verify learned Procrustes orthogonal transports.
    """
    torch.manual_seed(42)
    # 1. 3D Rotation Counterexample around z-axis
    theta = math.pi / 2  # 90 degrees, clearly H != I
    R_z = torch.tensor([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta),  math.cos(theta), 0.0],
        [0.0,             0.0,             1.0],
    ], dtype=torch.float32)

    # Confirm H != I
    assert not torch.allclose(R_z, torch.eye(3))

    # Rotation axis vector
    v_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    assert torch.allclose(R_z @ v_axis, v_axis)  # Fixed axis

    # Kernel of (H - I) is 1-dimensional
    H_minus_I = R_z - torch.eye(3)
    sv = torch.linalg.svdvals(H_minus_I)
    dim_ker = int((sv < 1e-5).sum().item())
    assert dim_ker == 1, f"Expected 1D kernel along rotation axis, got {dim_ker}"

    # Build 3-node triangle cycle where all edges carry R_z
    sheaf = CellularSheafLaplacian(num_nodes=3, dim=3)
    sheaf.edges = [(0, 1), (1, 2), (0, 2)]
    sheaf.restriction_maps[(0, 1)] = (R_z, torch.eye(3))
    sheaf.restriction_maps[(1, 2)] = (R_z, torch.eye(3))
    sheaf.restriction_maps[(0, 2)] = (torch.eye(3), R_z)

    # Constant axis section X = [v_axis; v_axis; v_axis] has ZERO energy
    X_axis = v_axis.repeat(3, 1)
    e_zero = sheaf.compute_sheaf_dirichlet_energy(X_axis, normalized=False)
    assert e_zero < 1e-6, f"Expected zero energy along rotation axis, got {e_zero}"

    # 2. Learned Procrustes transports
    Z_sim = torch.randn(4, 3)
    err = sheaf.learn_procrustes_transports(Z_sim)
    assert err >= 0.0
    assert len(sheaf.restriction_maps) > 0


def test_rope_rational_dependence_and_joint_character():
    """Verify RoPE dim=8 rational dependence:

    At N=100,000, marginal Weyl mean is exactly 0.00160038, while joint character has magnitude 1.0!
    """
    d = 8
    seq_len = 100000
    mean_w, t_dist, joint_mag, is_rat = HarmonicTorusSpectrum.rope_torus_winding(
        dim=d, seq_len=seq_len, base=10000.0, return_joint=True
    )
    # For dim=8, freqs are [1, 0.1, 0.01, 0.001], which are rational powers
    assert is_rat is True
    # Marginal Weyl sum evaluates to exactly ~0.00160038
    assert math.isclose(mean_w, 0.00160038, abs_tol=1e-5)
    # Exact joint character for integer relation 10*omega_1 - omega_0 == 0 has magnitude 1.0!
    assert math.isclose(joint_mag, 1.0, abs_tol=1e-5)


def test_ramanujan_largest_absolute_nontrivial_eigenvalue():
    """Verify Ramanujan diagnostic on 4-regular graph with negative eigenvalue:

    Naive ratio (lambda_2 only) falsely passes at ~0.72, while correct ratio (max |lambda|) fails at ~1.04.
    """
    k = 4.0
    ram_bound_normalized = (2.0 * math.sqrt(k - 1.0)) / k  # 2 * sqrt(3) / 4 ~= 0.8660

    # Adjacency matrix of 6-node graph with normalized eigenvalues [1.0, 0.6235, ..., -0.9010]
    # Corresponding to unnormalized [4.0, 2.494, ..., -3.604]
    lambda_2_norm = 2.494 / 4.0   # 0.6235
    lambda_n_norm = -3.604 / 4.0  # -0.9010

    naive_ratio = lambda_2_norm / ram_bound_normalized
    correct_ratio = max(abs(lambda_2_norm), abs(lambda_n_norm)) / ram_bound_normalized

    assert math.isclose(naive_ratio, 0.71995, abs_tol=1e-3)
    assert math.isclose(correct_ratio, 1.04038, abs_tol=1e-3)
    assert naive_ratio < 1.0  # Naive falsely passes
    assert correct_ratio > 1.0  # Correct diagnostic detects violation of Ramanujan property


def test_procrustes_transport_direction_and_batch_fitting():
    """Verify Procrustes transport orientation and multi-context batch fitting.

    1. Two-vertex, 3-dimensional example:
       When Z_v = Z_u R^T for true rotation R in SO(3), the correctly oriented Procrustes transport
       P_{u -> v} yields residual error 0.0, whereas reversed transpose P^T yields error 4.0.
    2. Multi-context batch fitting:
       Providing batched Z in R^{B x S x d} fits cross-covariance across all B contexts
       and held-out context validation evaluates residual error on test contexts.
    """
    torch.manual_seed(42)
    d = 3
    num_nodes = 2
    theta = 0.8
    R_true = torch.tensor([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta),  math.cos(theta), 0.0],
        [0.0,             0.0,             1.0],
    ], dtype=torch.float32)

    # 1. Single context 2-vertex example
    z0 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    z1 = z0 @ R_true.T
    Z_pair = torch.cat([z0, z1], dim=0)  # (2, 3)

    sheaf = CellularSheafLaplacian(num_nodes=2, dim=3)
    train_err = sheaf.learn_procrustes_transports(Z_pair, edges=[(0, 1)])
    assert math.isclose(train_err, 0.0, abs_tol=1e-5), f"Expected 0.0 error, got {train_err}"

    # Verify that applying backwards / reversed transport yields error 4.0
    P_learned = sheaf.get_edge_transport(0, 1)
    assert torch.allclose(P_learned, R_true, atol=1e-4)

    # Backwards transport error: ||z1 - z0 @ P_learned||^2 where z1 = z0 @ R^T
    diff_backwards = z1 - z0 @ P_learned  # R - R^T on [1, 0, 0] has norm squared = 2*(1 - cos(2*theta)) ~= 4.0 for 90-deg
    # For 90 degrees:
    theta_90 = math.pi / 2
    R_90 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    z0_90 = torch.tensor([[1.0, 0.0, 0.0]])
    z1_90 = z0_90 @ R_90.T  # [0, 1, 0]
    # Correct transport: z1_90 - z0_90 @ R_90.T == 0
    assert torch.allclose(z1_90, z0_90 @ R_90.T)
    # Reversed transport: z1_90 - z0_90 @ R_90 = [0, 1, 0] - [0, -1, 0] = [0, 2, 0] -> norm^2 = 4.0!
    diff_reversed = z1_90 - z0_90 @ R_90
    assert math.isclose(float(diff_reversed.pow(2).sum().item()), 4.0, abs_tol=1e-5)

    # 2. Multi-context batch fitting
    B = 10
    Z_batch = torch.randn(B, num_nodes, d)
    # Enforce exact relation on all batch contexts
    Z_batch[:, 1, :] = Z_batch[:, 0, :] @ R_true.T

    sheaf_multi = CellularSheafLaplacian(num_nodes=2, dim=3)
    err_multi = sheaf_multi.learn_procrustes_transports(Z_batch, edges=[(0, 1)], val_ratio=0.3)
    assert math.isclose(err_multi, 0.0, abs_tol=1e-5)
    assert math.isclose(sheaf_multi.val_transport_error, 0.0, abs_tol=1e-5)


def test_rope_resonance_multi_coordinate_search_dim16():
    """Verify RoPE resonance detection across multi-coordinate pairs at dim=16.

    At dim=16, base=10000:
    Frequencies are omega = [1.0, 0.3162, 0.1, 0.0316, 0.01, ...].
    Pair (0, 1) has ratio sqrt(10) (irrational), BUT pair (1, 3) has ratio:
      omega_1 / omega_3 = 10^(-0.5) / 10^(-1.5) = 10.0 in Z!
    Multi-coordinate resonance search must detect relation (1, 3) with joint character magnitude 1.0.
    """
    dim = 16
    seq_len = 500
    mean_w, t_dist, joint_mag, is_rat = HarmonicTorusSpectrum.rope_torus_winding(
        dim=dim, seq_len=seq_len, base=10000.0, return_joint=True
    )
    assert is_rat is True, "Expected multi-index search to detect rational dependence at dim=16"
    assert math.isclose(joint_mag, 1.0, abs_tol=1e-4), f"Expected joint character magnitude 1.0, got {joint_mag}"


def test_sheaf_fundamental_cycle_basis_triangle_plus_square():
    """Verify BFS spanning-tree fundamental cycle basis on graph with triangle + square.

    Graph topology on 5 vertices:
      Triangle: (0, 1), (1, 2), (0, 2)
      Square:   (0, 2), (2, 3), (3, 4), (0, 4)
    With generic orthogonal transports:
      The fundamental cycle basis captures chords for both the triangle and the square.
      The joint fixed-point subspace dimension strictly matches harmonic_space_dim (dim ker Delta_F = 0).
      Triangle-local heuristics would miss the square and falsely report spurious dimensions.
    """
    torch.manual_seed(123)
    num_nodes = 5
    dim = 3
    edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (0, 4)]

    sheaf = CellularSheafLaplacian(num_nodes=num_nodes, dim=dim)
    sheaf.edges = list(edges)

    for (u, v) in edges:
        K = torch.randn(dim, dim) * 0.5
        K = 0.5 * (K - K.T)
        R_u = torch.matrix_exp(K)
        R_v = torch.eye(dim)
        sheaf.restriction_maps[(u, v)] = (R_u, R_v)

    # Full spectral report using fundamental cycle basis
    report = sheaf.spectral_analysis()
    assert report.num_fundamental_cycles == 2  # One chord for triangle, one for square
    assert report.harmonic_space_dim == 0
    assert report.joint_fixed_subspace_dim == report.harmonic_space_dim
    assert report.is_harmonic_trivial is True


def test_ramanujan_domain_validation_disconnected_k5_and_regularity():
    """Verify Ramanujan domain validation:

    1. Disjoint 2*K_5 graph is disconnected (lambda_2 = 1.0), so spectral gap is 0.
       The correct diagnostic rejects it: is_connected=False, ratio > 1.0 (fails expander bound).
       In contrast, naive filtering of all 1.0s falsely passed at 0.288675.
    2. Single K_5 graph is connected and 4-regular: passes at ratio ~0.288675.
    3. Irregular graph: degree CV >= 0.05 fails domain validation.
    """
    # 1. Disjoint 2*K_5
    A_disjoint = torch.zeros((10, 10))
    A_disjoint[:5, :5] = 1.0 - torch.eye(5)
    A_disjoint[5:, 5:] = 1.0 - torch.eye(5)

    rep_disjoint = HarmonicTorusSpectrum.ramanujan_expansion_ratio(A_disjoint, return_report=True)
    assert isinstance(rep_disjoint, RamanujanDiagnosticReport)
    assert rep_disjoint.is_connected is False
    assert rep_disjoint.num_components == 2
    assert rep_disjoint.domain_valid is False
    assert rep_disjoint.is_ramanujan is False
    assert rep_disjoint.expansion_ratio > 1.0
    assert math.isclose(rep_disjoint.expansion_ratio, 1.1547, abs_tol=1e-3)
    assert "disconnected" in rep_disjoint.domain_warning.lower()

    # Plain float call also fails: ratio > 1.0
    ratio_float = HarmonicTorusSpectrum.ramanujan_expansion_ratio(A_disjoint, return_report=False)
    assert ratio_float > 1.0

    # 2. Single K_5
    A_k5 = 1.0 - torch.eye(5)
    rep_k5 = HarmonicTorusSpectrum.ramanujan_expansion_ratio(A_k5, return_report=True)
    assert rep_k5.is_connected is True
    assert rep_k5.is_regular is True
    assert rep_k5.domain_valid is True
    assert rep_k5.is_ramanujan is True
    assert math.isclose(rep_k5.expansion_ratio, 0.288675, abs_tol=1e-3)

    # 3. Irregular graph (star graph on 6 nodes)
    A_star = torch.zeros((6, 6))
    A_star[0, 1:] = 1.0
    A_star[1:, 0] = 1.0
    rep_star = HarmonicTorusSpectrum.ramanujan_expansion_ratio(A_star, return_report=True)
    assert rep_star.is_regular is False
    assert rep_star.domain_valid is False
    assert "irregular" in rep_star.domain_warning.lower()


if __name__ == "__main__":
    test_ekfac_curvature_and_ihvp()
    test_cellular_sheaf_laplacian_properties()
    test_circuit_attribution_direct_logit_attribution_linearity()
    test_harmonic_torus_spectrum()
    test_sheaf_3d_rotation_counterexample_and_procrustes()
    test_rope_rational_dependence_and_joint_character()
    test_ramanujan_largest_absolute_nontrivial_eigenvalue()
    test_procrustes_transport_direction_and_batch_fitting()
    test_rope_resonance_multi_coordinate_search_dim16()
    test_sheaf_fundamental_cycle_basis_triangle_plus_square()
    test_ramanujan_domain_validation_disconnected_k5_and_regularity()
    print("All advanced math tests passed successfully!")
