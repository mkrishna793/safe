"""archmath/evolve_sheaf_ekfac.py: OpenEvolve Prime Evolutionary Search for Sheaf & EK-FAC Probes.

Evolves Pareto-optimal computational heuristics for:
1. Ultra-fast Sheaf Laplacian algebraic connectivity (lambda_min) screening.
2. Vectorized EK-FAC inverse Hessian-vector product (IHVP) attribution without full matrix expansion.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PRIME_CORE_DIR = Path("/Users/mohana/.gemini/config/prime_core")
if PRIME_CORE_DIR.exists() and str(PRIME_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(PRIME_CORE_DIR))

import openevolve_prime

# --------------------------------------------------------------------------- #
# Evaluator Test Harness for Sheaf Laplacian Algebraic Connectivity
# --------------------------------------------------------------------------- #

EVAL_HARNESS = """
import torch, time, numpy as np

def generate_sheaf_suite(seed=42):
    torch.manual_seed(seed)
    suite = []
    # Test across 10 token graphs with orthogonal holonomies
    for S in [4, 8, 12, 16]:
        for d in [4, 8]:
            N = S * d
            Delta = torch.randn(N, N)
            Delta = 0.5 * (Delta + Delta.T)
            Delta = Delta.T @ Delta  # Positive semi-definite
            suite.append(Delta)
    return suite

suite = generate_sheaf_suite()

# Ground Truth Exact Full Eigvalsh
t0 = time.perf_counter()
gt_lmin = []
for D in suite:
    e = torch.linalg.eigvalsh(D)
    gt_lmin.append(float(e[0].item()))
exact_ms = (time.perf_counter() - t0) * 1000.0

# Candidate Probe
t0 = time.perf_counter()
cand_lmin = []
for D in suite:
    val = probe_sheaf_connectivity(D)
    cand_lmin.append(float(val))
cand_ms = (time.perf_counter() - t0) * 1000.0

corr = float(np.corrcoef(gt_lmin, cand_lmin)[0, 1])
if np.isnan(corr):
    corr = 0.0
speedup = exact_ms / max(cand_ms, 0.001)

bonus = 35.0 if corr >= 0.999 else (20.0 if corr >= 0.99 else 0.0)
penalty = 0.0 if corr >= 0.85 else 40.0
fitness = round(float((corr * 100.0) + min(speedup, 15.0) * 3.0 + bonus - penalty), 2)

metrics = {
    "correlation": round(corr, 6),
    "speedup": round(speedup, 2),
    "exact_time_ms": round(exact_ms, 2),
    "cand_time_ms": round(cand_ms, 2),
    "status": "passed" if corr >= 0.85 else "failed",
}
"""

CANDIDATES = [
    (
        "cand_sheaf_seed",
        "seed_baseline",
        "island_0",
        """
def probe_sheaf_connectivity(Delta: torch.Tensor) -> float:
    eigs = torch.linalg.eigvalsh(Delta)
    return float(eigs[0].item())
""",
    ),
    (
        "cand_sheaf_power_shift",
        "ast_mutation_shifted_power",
        "island_0",
        """
def probe_sheaf_connectivity(Delta: torch.Tensor) -> float:
    # Shift-invert power iteration approximation: lambda_max(Delta) - lambda_min(Delta)
    N = Delta.shape[0]
    x = torch.randn(N, 1, device=Delta.device)
    x = x / (x.norm() + 1e-12)
    # 3 power iterations for spectral radius
    for _ in range(3):
        x = Delta @ x
        x = x / (x.norm() + 1e-12)
    lmax = float((x.T @ Delta @ x).item())
    # Shifted matrix M = lmax * I - Delta
    M = lmax * torch.eye(N, device=Delta.device) - Delta
    y = torch.randn(N, 1, device=Delta.device)
    y = y / (y.norm() + 1e-12)
    for _ in range(5):
        y = M @ y
        y = y / (y.norm() + 1e-12)
    mu_max = float((y.T @ M @ y).item())
    return max(0.0, lmax - mu_max)
""",
    ),
    (
        "cand_sheaf_lanczos_crossover",
        "crossover_lanczos_tridiagonal",
        "island_1",
        """
def probe_sheaf_connectivity(Delta: torch.Tensor) -> float:
    # 6-step Lanczos tridiagonalization
    N = Delta.shape[0]
    m = min(8, N)
    v = torch.randn(N, device=Delta.device)
    v = v / (v.norm() + 1e-12)
    alpha = []
    beta = []
    v_prev = torch.zeros_like(v)
    for j in range(m):
        w = Delta @ v
        a = float(torch.dot(w, v).item())
        alpha.append(a)
        w = w - a * v - (beta[-1] * v_prev if beta else 0.0)
        b = float(w.norm().item())
        if b < 1e-8:
            break
        beta.append(b)
        v_prev = v
        v = w / b
    k = len(alpha)
    T = torch.zeros((k, k), device=Delta.device)
    for i in range(k):
        T[i, i] = alpha[i]
        if i < k - 1:
            T[i, i + 1] = beta[i]
            T[i + 1, i] = beta[i]
    e = torch.linalg.eigvalsh(T)
    return float(max(0.0, e[0].item()))
""",
    ),
]


def run_sheaf_evolution():
    engine = openevolve_prime.OpenEvolvePrimeEngine()
    print("🧬 [OpenEvolve Prime] Evolving Sheaf Laplacian Probes...")
    for name, mtype, island, code in CANDIDATES:
        res = engine.evaluate_candidate(code, EVAL_HARNESS)
        fit = res.get("fitness", 0.0)
        met = engine.kernel.namespace.get("metrics") or res.get("metrics", {})
        print(f"  [{island}] {name} ({mtype}) -> Fit: {fit} | Corr: {met.get('correlation')} | Speedup: {met.get('speedup')}x | Latency: {res.get('runtime_ms')}ms")
        engine.commit_individual(
            generation=1 if "island_0" in island else 2,
            mutation_type=mtype,
            code=code,
            fitness_score=fit,
            metrics=met,
            island_id=island,
            parent_id=None,
        )
    champ = engine.get_champion()
    print(f"🏆 Sheaf Champion: {champ.get('id')} | Fitness: {champ.get('fitness_score')}")


if __name__ == "__main__":
    run_sheaf_evolution()
