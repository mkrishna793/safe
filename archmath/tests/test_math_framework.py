"""Comprehensive Pareto Verification Battery for ArchMath RealTimeMathDecoder.

Run:  ./.venv/bin/python -m archmath.tests.test_math_framework
      ./.venv/bin/python -m archmath.tests.test_math_framework moe       # one arch

Verifies all 17 architectures under real-time hidden space decoding:
  1. RealTimeMathDecoder hooks non-intrusively across 2 training steps
  2. Forward + backward pass produces finite loss and parameter gradients
  3. Mathematical invariants strictly hold:
     - Dirichlet energy E(Z) >= 0.0
     - Participation ratio PR >= 1.0
     - Adjacent CKA in [0.0, 1.0 + 1e-4]
     - Von Neumann spectral entropy H_spec >= 0.0
  4. Anomaly detection system properly identifies non-finite, dead, or collapsed states.
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Any, List

import torch
import torch.nn.functional as F

import archmath.impl  # noqa: F401
import archmath.symbolic  # noqa: F401
from archmath.core import get_arch, list_archs

# Support import from either manifold or math_framework
try:
    from archmath.manifold import (
        RealTimeMathDecoder,
        ManifoldMetrics,
        compute_cka,
        compute_dirichlet_energy,
        compute_normalized_dirichlet_energy,
        compute_fiedler_value,
        compute_participation_ratio,
        compute_spectral_entropy,
        compute_riemannian_metric_tensor,
        decode_arch_latent_state,
    )
except ImportError:
    from archmath.math_framework import (
        RealTimeMathDecoder,
        ManifoldMetrics,
        compute_cka,
        compute_dirichlet_energy,
        compute_normalized_dirichlet_energy,
        compute_fiedler_value,
        compute_participation_ratio,
        compute_spectral_entropy,
        compute_riemannian_metric_tensor,
        decode_arch_latent_state,
    )


def _cfg(name: str):
    mod = getattr(archmath.impl, name, None)
    if mod is not None and hasattr(mod, "tiny_config"):
        return mod.tiny_config()
    raise AssertionError(f"archmath.impl.{name} does not export tiny_config()")


def _logits(out: torch.Tensor, vocab: int) -> torch.Tensor:
    if out.dim() == 3 and out.shape[-1] == vocab:
        return out
    if out.dim() == 3:
        return torch.nn.Linear(out.shape[-1], vocab)(out)
    raise AssertionError(f"unexpected output shape {tuple(out.shape)}")


def test_anomaly_detection() -> List[str]:
    """Verify anomaly detection flags trigger correctly under pathological conditions."""
    fails: List[str] = []

    # 1. Non-finite detection
    try:
        dummy_model = torch.nn.Linear(8, 8)
        with RealTimeMathDecoder(dummy_model, tag="anomaly_test") as dec:
            dec.begin_step(0)
            # simulate pathological non-finite tensor
            nan_tensor = torch.tensor([[float("nan")] * 8])
            dec._layer_outputs["pathological_layer"] = nan_tensor
            snap = dec.end_step(loss=0.5)
            kinds = [a["kind"] for a in snap.anomalies]
            if "non_finite" not in kinds:
                fails.append("Failed to flag non_finite anomaly")
    except Exception as e:
        fails.append(f"anomaly test non_finite raised: {e}")

    # 2. Dead layer detection
    try:
        dummy_model = torch.nn.Linear(8, 8)
        with RealTimeMathDecoder(dummy_model, tag="anomaly_test") as dec:
            dec.begin_step(0)
            dead_tensor = torch.zeros((4, 8))
            dec._layer_outputs["dead_layer"] = dead_tensor
            snap = dec.end_step(loss=0.5)
            kinds = [a["kind"] for a in snap.anomalies]
            if "dead_layer" not in kinds:
                fails.append("Failed to flag dead_layer anomaly")
    except Exception as e:
        fails.append(f"anomaly test dead_layer raised: {e}")

    return fails


def test_primitives() -> List[str]:
    """Verify mathematical invariant properties on synthetic data."""
    fails: List[str] = []

    torch.manual_seed(42)
    # 1. Dirichlet energy
    A = torch.softmax(torch.randn(10, 10), dim=-1)
    Z = torch.randn(10, 16)
    e = compute_dirichlet_energy(A, Z)
    if not math.isfinite(e) or e < 0.0:
        fails.append(f"compute_dirichlet_energy returned {e} (expected >= 0.0)")

    # 2. Participation ratio (must be >= 1.0)
    pr = compute_participation_ratio(Z)
    if not math.isfinite(pr) or pr < 1.0:
        fails.append(f"compute_participation_ratio returned {pr} (expected >= 1.0)")

    # 3. CKA (must be in [0.0, 1.0 + 1e-4])
    cka_same = compute_cka(Z, Z)
    if abs(cka_same - 1.0) > 1e-3:
        fails.append(f"compute_cka(Z, Z) returned {cka_same} (expected ~1.0)")
    cka_diff = compute_cka(Z, torch.randn_like(Z))
    if not (0.0 <= cka_diff <= 1.0 + 1e-4):
        fails.append(f"compute_cka(Z, W) returned {cka_diff} (expected in [0, 1])")

    # 4. Spectral entropy (must be >= 0.0)
    h_spec = compute_spectral_entropy(Z)
    if not math.isfinite(h_spec) or h_spec < 0.0:
        fails.append(f"compute_spectral_entropy returned {h_spec} (expected >= 0.0)")

    return fails


def test_adversarial_mathematical_attacks() -> List[str]:
    """Rigorously stress-test mathematical formulations against adversarial attacks."""
    fails: List[str] = []

    # 1. Singularity & Invertibility Attack (Rank-deficient & rectangular Jacobian)
    try:
        # Zero Jacobian
        J_zero = torch.zeros(16, 32)
        res_zero = compute_riemannian_metric_tensor(J_zero, eps=1e-6)
        if res_zero["volume_distortion"] != 1.0 or res_zero["sylvester_log_det"] != 0.0:
            fails.append(f"J_zero volume distortion failed: {res_zero}")

        # Rank-deficient (rank 1 in 16 x 32)
        J_r1 = torch.outer(torch.randn(16), torch.randn(32))
        res_r1 = compute_riemannian_metric_tensor(J_r1, eps=1e-6)
        if not math.isfinite(res_r1["volume_distortion"]) or res_r1["volume_distortion"] <= 0.0:
            fails.append(f"Rank-1 volume distortion non-positive or non-finite: {res_r1}")
        if res_r1["sylvester_log_det"] < 0.0:
            fails.append(f"Sylvester log det negative: {res_r1['sylvester_log_det']}")
        if res_r1["metric_condition_number"] < 1e3:
            fails.append(f"Rank-deficient Jacobian condition number failed to detect singularity: {res_r1['metric_condition_number']}")

        # Rectangular large Jacobian
        J_large = 100.0 * torch.randn(8, 64)
        res_large = compute_riemannian_metric_tensor(J_large, eps=1e-6)
        if not math.isfinite(res_large["volume_distortion"]) or res_large["volume_distortion"] <= 0.0:
            fails.append(f"Large rectangular Jacobian volume distortion overflowed or became non-finite: {res_large}")
    except Exception as e:
        fails.append(f"Singularity attack raised exception: {e}")

    # 2. Negative Eigenvalue & Asymmetry Attack (Asymmetric attention & Laplacian)
    try:
        # Directed causal chain without return edges
        A_asym = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        Z = torch.randn(3, 8)
        e_dir = compute_dirichlet_energy(A_asym, Z)
        if not math.isfinite(e_dir) or e_dir < 0.0:
            fails.append(f"Asymmetric attention produced negative or non-finite Dirichlet energy: {e_dir}")

        fiedler = compute_fiedler_value(A_asym, S=3)
        if not math.isfinite(fiedler) or fiedler < 0.0:
            fails.append(f"Asymmetric attention produced negative or non-finite Fiedler value: {fiedler}")

        # Highly asymmetric random attention
        A_rand = torch.rand(20, 20)
        A_rand = torch.triu(A_rand, diagonal=1) # Upper triangular causal
        Z_rand = torch.randn(20, 16)
        e_rand = compute_dirichlet_energy(A_rand, Z_rand)
        if not math.isfinite(e_rand) or e_rand < 0.0:
            fails.append(f"Upper triangular attention produced negative Dirichlet energy: {e_rand}")
    except Exception as e:
        fails.append(f"Asymmetry attack raised exception: {e}")

    # 3. Scale Invariance Attack (Normalized Dirichlet Energy)
    try:
        A = torch.softmax(torch.randn(12, 12), dim=-1)
        Z = torch.randn(12, 16)
        e_norm_base = compute_normalized_dirichlet_energy(A, Z)
        e_norm_scaled_up = compute_normalized_dirichlet_energy(A, 100.0 * Z)
        e_norm_scaled_down = compute_normalized_dirichlet_energy(A, 0.01 * Z)

        diff_up = abs(e_norm_scaled_up - e_norm_base)
        diff_down = abs(e_norm_scaled_down - e_norm_base)
        if diff_up > 1e-4:
            fails.append(f"Normalized Dirichlet energy not scale invariant under 100x scale: diff={diff_up}")
        if diff_down > 1e-4:
            fails.append(f"Normalized Dirichlet energy not scale invariant under 0.01x scale: diff={diff_down}")
    except Exception as e:
        fails.append(f"Scale invariance attack raised exception: {e}")

    # 4. Zero Variance & Degeneracy Attack (CKA on constant representations)
    try:
        # Constant representation against active
        X_const = torch.ones(20, 16) * 5.0
        Y_active = torch.randn(20, 16)
        cka_zero_var = compute_cka(X_const, Y_active)
        if cka_zero_var != 0.0:
            fails.append(f"CKA(constant, active) returned {cka_zero_var} (expected exactly 0.0)")

        # Two identical constant representations
        X_const2 = torch.ones(20, 16) * 5.0
        cka_identical = compute_cka(X_const, X_const2)
        if abs(cka_identical - 1.0) > 1e-4:
            fails.append(f"CKA(constant, identical_constant) returned {cka_identical} (expected 1.0)")

        # Constant representation with microscopic float noise (simulating rounding errors)
        X_noisy = X_const + 1e-8 * torch.randn_like(X_const)
        cka_noisy = compute_cka(X_noisy, Y_active)
        if not (0.0 <= cka_noisy <= 1.0 + 1e-4):
            fails.append(f"CKA with epsilon-level noise blew up: {cka_noisy}")
    except Exception as e:
        fails.append(f"Zero variance CKA attack raised exception: {e}")

    # 5. MoE Shannon Entropy Attack (Zero and negative router probabilities)
    try:
        # Sparse routing with unselected experts (zeros)
        P_sparse = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
        _, m_sparse, _ = decode_arch_latent_state({"route/probs": P_sparse})
        ent_sparse = m_sparse.get("router_entropy")
        if ent_sparse is None or not math.isfinite(ent_sparse) or ent_sparse < 0.0:
            fails.append(f"MoE router entropy with zeros produced non-finite or negative entropy: {ent_sparse}")

        # Pathological negative values / underflows
        P_patho = torch.tensor([[-1e-6, 0.0, 0.5, 0.5]])
        _, m_patho, _ = decode_arch_latent_state({"route/probs": P_patho})
        ent_patho = m_patho.get("router_entropy")
        if ent_patho is None or not math.isfinite(ent_patho) or ent_patho < 0.0:
            fails.append(f"MoE router entropy with negative input produced non-finite or negative entropy: {ent_patho}")
    except Exception as e:
        fails.append(f"MoE Shannon entropy attack raised exception: {e}")

    return fails


def check_arch(name: str) -> List[str]:
    """Run 2-step training battery through RealTimeMathDecoder and verify invariants."""
    fails: List[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    try:
        cfg = _cfg(name)
    except Exception as e:
        return [f"tiny_config: {e}"]

    vocab = int(cfg.get("vocab", 256))
    seq = int(cfg.get("seq", 16))

    try:
        model = get_arch(name)(cfg)
    except Exception as e:
        return [f"build: {e}"]

    model.train()
    torch.manual_seed(0)
    _gv = 256 if name == "blt" else vocab
    tokens = torch.randint(0, _gv, (2, seq))

    try:
        with RealTimeMathDecoder(model, tag=name) as decoder:
            for step in range(2):
                decoder.begin_step(step, inputs=tokens)
                out, aux = model(tokens)
                logits = _logits(out, vocab)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab), tokens[:, 1:].reshape(-1)
                )

                expect(torch.isfinite(loss).item(), f"step {step}: non-finite loss")

                model.zero_grad(set_to_none=True)
                loss.backward()

                # Verify gradients exist and are finite
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                expect(len(grads) > 0, f"step {step}: no parameter gradients computed")
                expect(
                    all(torch.isfinite(g).all() for g in grads),
                    f"step {step}: non-finite parameter gradients",
                )

                snap = decoder.end_step(loss=loss, aux=aux)

                # Invariant 1: Dirichlet Energy E(Z) >= 0.0
                expect(
                    math.isfinite(snap.dirichlet_energy),
                    f"step {step}: non-finite Dirichlet energy: {snap.dirichlet_energy}",
                )
                expect(
                    snap.dirichlet_energy >= 0.0,
                    f"step {step}: Dirichlet energy violation: {snap.dirichlet_energy} < 0.0",
                )

                # Invariant 2: Participation Ratio PR >= 1.0
                expect(
                    math.isfinite(snap.participation_ratio),
                    f"step {step}: non-finite Participation ratio: {snap.participation_ratio}",
                )
                expect(
                    snap.participation_ratio >= 1.0,
                    f"step {step}: Participation ratio violation: {snap.participation_ratio} < 1.0",
                )

                # Invariant 3: Adjacent CKA in [0.0, 1.0 + 1e-4]
                expect(
                    math.isfinite(snap.mean_cka),
                    f"step {step}: non-finite mean CKA: {snap.mean_cka}",
                )
                expect(
                    0.0 <= snap.mean_cka <= 1.0 + 1e-4,
                    f"step {step}: CKA bounds violation: {snap.mean_cka} not in [0.0, 1.0]",
                )
                for i, cka in enumerate(snap.cka_adjacent):
                    expect(
                        math.isfinite(cka) and 0.0 <= cka <= 1.0 + 1e-4,
                        f"step {step}: layer CKA[{i}] = {cka} out of bounds",
                    )

                # Invariant 4: Spectral Entropy >= 0.0
                expect(
                    math.isfinite(snap.spectral_entropy),
                    f"step {step}: non-finite spectral entropy: {snap.spectral_entropy}",
                )
                expect(
                    snap.spectral_entropy >= 0.0,
                    f"step {step}: spectral entropy violation: {snap.spectral_entropy} < 0.0",
                )

                # Invariant 5: No unhandled non_finite anomalies during healthy run
                fatal_anomalies = [a for a in snap.anomalies if a.get("kind") == "non_finite"]
                expect(
                    not fatal_anomalies,
                    f"step {step}: healthy run produced non_finite anomalies: {fatal_anomalies}",
                )

        rep = decoder.report()
        expect(len(rep.steps) == 2, f"expected 2 recorded steps, got {len(rep.steps)}")
    except Exception as e:
        fails.append(f"decoder error: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")

    return fails


def main(argv: List[str]) -> int:
    targets = argv[1:] or list_archs()
    print("=" * 70)
    print("ArchMath RealTimeMathDecoder Verification Battery")
    print(f"Target Architectures ({len(targets)}): {', '.join(targets)}")
    print("=" * 70)

    # 1. Primitives verification
    prim_fails = test_primitives()
    if prim_fails:
        print("  [FAIL] Math Primitives Unit Tests:")
        for f in prim_fails:
            print(f"         - {f}")
        return 1
    print("  [PASS] Mathematical Primitives Verified (E >= 0, PR >= 1, CKA in [0, 1], H >= 0)")

    # 2. Anomaly flags verification
    anomaly_fails = test_anomaly_detection()
    if anomaly_fails:
        print("  [FAIL] Anomaly Detection Unit Tests:")
        for f in anomaly_fails:
            print(f"         - {f}")
        return 1
    print("  [PASS] Anomaly Detection Smoke Alarms Verified (non_finite, dead_layer)")

    # 3. Adversarial attacks verification
    adv_fails = test_adversarial_mathematical_attacks()
    if adv_fails:
        print("  [FAIL] Adversarial Mathematical Red-Team Tests:")
        for f in adv_fails:
            print(f"         - {f}")
        return 1
    print("  [PASS] Adversarial Mathematical Red-Team Tests Passed (Sylvester, PSD Symmetrization, Scale Invariance, CKA Degeneracy, MoE Shannon Clamping)")

    print("-" * 70)
    total_fails = 0
    for name in targets:
        fails = check_arch(name)
        if fails:
            total_fails += 1
            print(f"  [FAIL] {name:15s}")
            for f in fails:
                print(f"         - {f}")
        else:
            print(f"  [PASS] {name:15s}")

    print("=" * 70)
    print(f"Summary: {len(targets) - total_fails}/{len(targets)} architectures passed verification")
    print("=" * 70)
    return 1 if total_fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
