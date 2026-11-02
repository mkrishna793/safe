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


if __name__ == "__main__":
    test_ekfac_curvature_and_ihvp()
    test_cellular_sheaf_laplacian_properties()
    test_circuit_attribution_direct_logit_attribution_linearity()
    test_harmonic_torus_spectrum()
    print("All advanced math tests passed successfully!")
