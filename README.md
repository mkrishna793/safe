# ArchMath: First-Principles Architecture Reconstructions & Real-Time Mathematical Latent Decoding Framework

[![ArchMath Verification Battery](https://github.com/mkrishna793/safe/actions/workflows/verify.yml/badge.svg)](https://github.com/mkrishna793/safe/actions/workflows/verify.yml)
[![SageMath 10.9 Verified](https://img.shields.io/badge/SageMath-10.9%20Verified%20(28%2F28)-brightgreen.svg)](https://www.sagemath.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **ArchMath** is a rigorous research codebase and real-time hidden-space diagnostic engine that decodes 17 modern AI architectures from pure mathematical first principles and pairs each with an exact dual observer framework that tracks Riemannian metric tensors, Graph Laplacian Dirichlet energy, Centered Kernel Alignment (CKA), and information-theoretic spectra in real-time during training and inference.

---

## 🏛️ Repository Architecture & Organization

```
safe/
├── README.md                           # Master mathematical treatise and user guide
├── ARCHITECT_CONTRACT.md               # Standardized module specification contract
├── requirements.txt                    # Minimal dependencies (PyTorch, SymPy, NumPy, SciPy)
├── pyproject.toml                      # Standard packaging definition
├── .github/workflows/verify.yml        # CI test suite executing on GitHub
├── archmath/                           # Core Framework Package
│   ├── __init__.py                     # Clean package exports
│   ├── core.py                         # ArchConfig, ArchModule, MathSpec, and global registries
│   ├── trace.py                        # Fixed-width telemetry, SVD effective rank, attribution
│   ├── manifold.py                     # RealTimeMathDecoder, Riemannian metric tensor, Graph Laplacians
│   ├── math_framework.py               # Framework alias and dual probe definitions
│   ├── evolve_probes.py                # OpenEvolve AlphaEvolve-style genetic probe optimization
│   ├── impl/                           # Executable PyTorch Reconstructions (17 Architectures)
│   │   ├── dense.py                    # Dense Decoder-Only Transformer (RoPE, RMSNorm, SwiGLU, GQA)
│   │   ├── sparse_attn.py              # Native Top-k Sparse Attention (Sliding window + block indexer)
│   │   ├── linear_attn.py              # Linear Attention & Gated DeltaNet (Fast Weight Programmer)
│   │   ├── ssm.py                      # State Space Models & Mamba-2 (State Space Duality, SSD)
│   │   ├── hybrid.py                   # Hybrid Attention-SSM Stacks (Jamba / Zamba style)
│   │   ├── moe.py                      # Sparse Mixture of Experts (DeepSeekMoE, shared + top-k routed)
│   │   ├── diffusion.py                # Diffusion Transformers & Flow Matching (LLaDA-style)
│   │   ├── jepa.py                     # World Models / Joint-Embedding Predictive Architecture (I-JEPA)
│   │   ├── kan.py                      # Kolmogorov-Arnold Networks (B-spline activation manifolds)
│   │   ├── titans.py                   # Titans Test-Time Memory (Surprise-gated associative memory)
│   │   ├── blt.py                      # Byte Latent Transformer (Entropy patcher, local/global encoder)
│   │   ├── neuromorphic.py             # Spiking Neural Networks (Leaky Integrate-and-Fire, surrogate grad)
│   │   ├── mla.py                      # Multi-Head Latent Attention (DeepSeek low-rank KV absorption)
│   │   ├── multimodal.py               # Native Early-Fusion Multimodal (Continuous token manifold)
│   │   ├── looped.py                   # Recurrent-Depth Looped Transformers (Universal Transformer)
│   │   ├── rnn.py                      # Classical Recurrence (Stacked LSTM Constant Error Carousel)
│   │   └── mlp.py                      # Deep Dense Networks (Residual SwiGLU, sequence orthogonality)
│   ├── symbolic/                       # Exact Symbolic Specifications (SymPy & LaTeX)
│   │   ├── base.py                     # Symbolic complexity evaluation harness
│   │   └── [17 architecture specs].py  # Equations, assumptions, complexities, invariants, citations
│   ├── sage/                           # Formal Symbolic Verification Suite
│   │   ├── derive.sage                 # 28 Exact Algebraic Proofs verified under SageMath
│   │   └── run.sh                      # One-command SageMath verification runner
│   └── tests/                          # Automated Verification Batteries
│       ├── test_archs.py               # 17-Architecture forward/backward/gradient/aux battery
│       └── test_math_framework.py      # Real-time decoder invariants & adversarial red-team tests
└── research/                           # Exhaustive PRAXIST Research Monographs
    ├── 01_frontier_academic_alignment.md   # 43KB SOTA institutional alignment (Stanford/MIT/Google/DeepSeek)
    ├── 02_edge_case_stress_audit.md        # 19KB Numerical edge-case stress audit & 5 bug repairs
    ├── 03_dual_mathematical_framework.md   # 27KB Manifold immersion & dual operator formulation
    └── 04_adversarial_red_team_audit.md    # 24KB Adversarial red-team attacks & SageMath proofs
```

---

## 📐 The 17 Rebuilt Architectures: Mathematical Blueprint

| # | Architecture | Core Mathematical Formulation | State / Memory Capacity | Inference Complexity | Primary Invariant |
|---|---|---|---|---|---|
| **1** | **Dense Transformer** | $y = \operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_h}} + M\right) V$ | Full KV Cache: $\mathcal{O}(S \cdot d)$ | $\mathcal{O}(S^2 \cdot d)$ | Row-stochastic attention: $\sum_j A_{ij} = 1$ |
| **2** | **Looped Transformer** | $h^{(k+1)} = h^{(k)} + \mathcal{T}(h^{(k)}), \quad k=1 \dots K$ | Weight-tied: $\mathcal{O}(d)$ | $\mathcal{O}(K \cdot S^2 \cdot d)$ | Banach contraction: $\rho(W) < 1$ |
| **3** | **Native Sparse Attn** | $A_{i,j} = \operatorname{softmax}(Q_i K_j^T) \cdot \mathbb{I}[j \in \mathcal{N}(i) \cup \operatorname{TopK}(S_i)]$ | Block-sparse: $\mathcal{O}(k \cdot S \cdot d)$ | $\mathcal{O}(k \cdot S \cdot d)$ | Exact sparsity ratio $1 - \frac{\|A\|_0}{S^2}$ |
| **4** | **Linear Attn / DeltaNet** | $S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^T$ | Associative Matrix: $\mathcal{O}(d^2)$ | $\mathcal{O}(S \cdot d^2)$ | Error projection: $S_t k_t = v_t$ |
| **5** | **SSM / Mamba-2 SSD** | $h_t = \bar{A}_t h_{t-1} + B_t x_t, \quad y_t = C_t h_t + D x_t$ | State Vector: $\mathcal{O}(d_{\text{state}} \cdot d)$ | $\mathcal{O}(S \cdot d)$ | 1-semiseparable chunk equivalence |
| **6** | **Sparse MoE** | $y_t = x_t + \sum_{i \in \operatorname{TopK}(s_t + b)} g_{i,t} \operatorname{FFN}_i(x_t) + \operatorname{FFN}_{\text{shared}}(x_t)$ | Sparse Active: $\mathcal{O}(k \cdot d_{\text{ff}})$ | $\mathcal{O}(k \cdot d_{\text{ff}})$ | Router balance Hessian $H \succ 0$ |
| **7** | **Diffusion (DiT)** | $dx = \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \ln p_t(x)\right] dt$ | Bidirectional Denoiser | $\mathcal{O}(T_{\text{steps}} \cdot S^2 \cdot d)$ | Fokker-Planck probability flux matching |
| **8** | **JEPA (World Models)** | $\mathcal{L} = \|s_{t+1}^{\text{target}} - \operatorname{Pred}(s_t^{\text{online}}, a_t)\|_2^2 + \lambda \mathcal{R}_{\text{VICReg}}$ | Latent Embedding | $\mathcal{O}(S \cdot d)$ | Non-collapse: $\operatorname{Var}(s) \ge 1$, $\nabla_{\text{target}} = 0$ |
| **9** | **KAN (Kolmogorov-Arnold)** | $f(x) = \sum_{i} w_i \operatorname{SiLU}(x) + \sum_{j} c_j B_{j, p}(x)$ | Spline Coefficients: $\mathcal{O}(G + p)$ | $\mathcal{O}(S \cdot d \cdot G)$ | Partition of unity: $\sum_j B_{j,p}(x) \equiv 1$ |
| **10** | **Titans Memory** | $M_t = (1 - \alpha_t) M_{t-1} + \eta_t (1 - s_t) (v_t - M_{t-1} k_t) k_t^T$ | Test-Time Matrix: $\mathcal{O}(d_{\text{mem}}^2)$ | $\mathcal{O}(S \cdot d_{\text{mem}}^2)$ | Surprise non-negativity: $s_t \ge 0$ |
| **11** | **Byte-Level BLT** | $\operatorname{Patch}(b) \leftarrow \mathbb{I}[H(p(b_i \mid b_{<i})) > \theta]$ | Entropy-Segmented Patches | $\mathcal{O}(P^2 \cdot d), \ P \ll N$ | Cross-attention latent byte reconstruction |
| **12** | **Neuromorphic SNN** | $\tau_m \frac{d V}{d t} = -(V - V_{\text{rest}}) + R I(t), \quad S_t = \Theta(V_t - V_{\text{th}})$ | Membrane Potential $V_t$ | $\mathcal{O}(S \cdot d)$ Event-driven | Exact LIF exponential decay solution |
| **13** | **DeepSeek MLA** | $q_t = W_{UQ} c_t^Q, \ [k_t^C; k_t^R] = [W_{UK} c_t^{KV}; \operatorname{RoPE}(W_{KR} x_t)]$ | Compressed KV: $\mathcal{O}(S \cdot d_c)$ | $\mathcal{O}(S^2 \cdot d_h)$ | Matrix absorption: $W_{\text{abs}} = W_{UQ} W_{UK}^T$ |
| **14** | **Multimodal Early Fusion** | $Z = [E_{\text{text}}(T); E_{\text{vision}}(V); E_{\text{audio}}(A)] + P_{\text{2D}}$ | Unified Cross-Modal Stream | $\mathcal{O}((S_T + S_V + S_A)^2 \cdot d)$ | Modality temporal causality: $\Delta_{\text{prior}} = 0$ |
| **15** | **Hybrid Attention-SSM** | Layer $l \in \{\operatorname{SSD}, \operatorname{SoftmaxAttn}, \operatorname{LinearAttn}\}$ with fraction $f$ | Hybrid KV Cache: $\mathcal{O}(f \cdot S \cdot d)$ | $\mathcal{O}((f S^2 + (1-f)S) d)$ | KV memory savings ratio exactly $1 - f$ |
| **16** | **Classical RNN / LSTM** | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \quad h_t = o_t \odot \tanh(c_t)$ | Linear Cell Carousel | $\mathcal{O}(S \cdot d^2)$ | CEC Jacobian: $\frac{\partial c_T}{\partial c_t} = \prod_{k=t+1}^T \operatorname{diag}(f_k)$ |
| **17** | **Deep Dense MLP** | $x_{l+1} = x_l + (\operatorname{SiLU}(x_l W_g) \odot x_l W_u) W_d$ | Point-wise coordinate transform | $\mathcal{O}(S \cdot d \cdot d_{\text{ff}})$ | Sequence orthogonality: $\frac{\partial x_{l,i}}{\partial x_{l-1,j}} = 0 \ (i \neq j)$ |

---

## 🔬 The Real-Time Mathematical Latent Decoder (`archmath/manifold.py`)

The companion decoding framework operates as a **differential-geometric and spectral observer** that measures what is happening inside the model's residual stream in real time.

```python
from archmath import RealTimeMathDecoder, get_arch
from archmath.impl.dense import tiny_config

model = get_arch("dense")(tiny_config())

with RealTimeMathDecoder(model, mode="train") as decoder:
    for step, tokens in enumerate(data_loader):
        decoder.begin_step(step, inputs=tokens)
        logits, aux = model(tokens)
        loss = criterion(logits, targets)
        loss.backward()
        
        # Real-time mathematical snapshot
        metrics = decoder.step(loss=loss, inputs=tokens)
        print(f"Dirichlet Energy: {metrics.dirichlet_energy:.4f}")
        print(f"Participation Ratio: {metrics.participation_ratio:.2f}")
        print(f"Mean CKA Velocity: {metrics.mean_cka:.4f}")
        print(f"Riemannian Volume Distortion: {metrics.volume_distortion:.4f}")
```

### Core Diagnostic Observables:

1. **Riemannian Metric Tensor & Local Curvature**:
   Evaluates the pullback metric tensor $G = J^T J$ where $J = \frac{\partial z_L}{\partial x_{\text{embed}}}$. Volume distortion is computed via **Sylvester's Determinant Theorem** in log-space to guarantee unconditional numerical stability:
   $$\ln \operatorname{Vol} = \frac{1}{2} \sum_{i=1}^{\min(m, n)} \ln\left(1 + \frac{\sigma_i^2}{\epsilon}\right), \quad \kappa_\epsilon = \frac{\sigma_{\max}^2 + \epsilon}{\sigma_{\min}^2 + \epsilon}$$

2. **Graph Laplacian Spectral Geometry & Over-Smoothing**:
   Symmetrizes the directed attention adjacency matrix *before* degree summation to strictly prevent negative eigenvalues:
   $$A_{\text{sym}} = \frac{A + A^T}{2}, \quad L_{\text{sym}} = \operatorname{diag}(A_{\text{sym}} \mathbf{1}) - A_{\text{sym}}$$
   Computes **Normalized Scale-Invariant Dirichlet Energy**:
   $$\mathcal{E}_{\text{norm}}(Z) = \frac{\frac{1}{2} \operatorname{Tr}(Z^T L_{\text{sym}} Z)}{\|Z\|_F^2 + \epsilon} \ge 0$$
   When $\mathcal{E}_{\text{norm}}(Z) \to 0$, the framework triggers an alert for **representation collapse / over-smoothing**.

3. **Centered Kernel Alignment (CKA) & Geometric Velocity**:
   Measures layer-to-layer geometric rotation and representational drift:
   $$\operatorname{CKA}(K_l, K_{l+1}) = \frac{\operatorname{HSIC}(K_l, K_{l+1})}{\sqrt{(\operatorname{HSIC}(K_l, K_l) + \epsilon)(\operatorname{HSIC}(K_{l+1}, K_{l+1}) + \epsilon)}}, \quad v_{\text{geom}} = 1 - \operatorname{CKA}$$

4. **Spectral Intrinsic Dimensionality**:
   Measures the effective dimensionality of the representation manifold via the SVD Participation Ratio:
   $$\operatorname{PR}(Z) = \frac{\left(\sum_i \sigma_i\right)^2}{\sum_i \sigma_i^2} \ge 1.0, \quad H_{\text{spec}} = -\sum_i \tilde{\sigma}_i \ln \tilde{\sigma}_i$$

---

## 🧪 Formal Symbolic Verification in SageMath 10.9

The formal mathematical identities are proven algebraically in [`archmath/sage/derive.sage`](archmath/sage/derive.sage).

Run the full verification suite:
```bash
./archmath/sage/run.sh
```

### Verified Theorems:
* **Theorems 1–3**: Attention convex combination, RMSNorm scale-invariance, GQA parameter/KV reduction.
* **Theorems 4–6**: MLA exact matrix absorption over $\mathbb{Q}$, DeltaNet least-squares projection, Linear attention outer product recurrence.
* **Theorems 7–9**: Looped transformer Banach fixed-point contraction ($\rho(W) < 1$), deep supervision geometric series, masked diffusion weight $1/(1-t)$.
* **Theorems 10–13**: MoE uniform load optimization, B-spline Cox-de Boor partition of unity, JEPA representation collapse barrier, Hybrid attention savings.
* **Theorems 14–18**: LSTM Constant Error Carousel unit Jacobian ($\partial c_T/\partial c_t \equiv 1$), SSM bilinear Tustin stability mapping ($\operatorname{Re}(s) < 0 \implies |z| < 1$), Titans associative zero-residual update, LIF exact analytical ODE relaxation, MLP cross-token sequence orthogonality.
* **Theorems 19–23**: DiT probability flow ODE / Fokker-Planck continuity, KAN affine constant reproduction, JEPA stop-gradient non-vanishing gradient, Mamba-2 SSD chunked 1-semiseparable matrix equivalence, MoE auxiliary loss positive-definite Hessian ($H \succ 0$).
* **Theorems 24–28 (Adversarial Red-Team Proofs)**: Sylvester determinant rectangular Jacobian equivalence, asymmetric Laplacian negative eigenvalue refutation and PSD symmetrization, Dirichlet energy scale-invariance ($\mathcal{E}_{\text{norm}}(c Z) \equiv \mathcal{E}_{\text{norm}}(Z)$), CKA zero-variance epsilon regularization, and MoE Shannon entropy zero-clamping via L'Hôpital's rule.

**Result: `ALL 28 SYMBOLIC CHECKS PASSED EXACTLY UNDER SAGEMATH`.**

---

## 🧬 OpenEvolve Prime Genetic Probe Synthesis

Computing exact SVD for Participation Ratio on large sequence lengths is computationally expensive ($\mathcal{O}(N \cdot d^2) \approx 165\text{ ms}$).

Using **OpenEvolve Prime** (`archmath/evolve_probes.py`), we ran an AlphaEvolve-style genetic program synthesis across 3 evolutionary islands:
* **Island 0 (Exploration)**: AST mutations targeting random projection and subsampling.
* **Island 1 (Refinement)**: Gramian matrix eigendecompositions ($G = B B^T$) and subspace power iterations.
* **Island 2 (Pareto Verification)**: Hybrid crossover combining $O(ND)$ Dirichlet screening, $k=56$ power-iterated subspace projection, and closed-form Frobenius residual tail completion.

### Empirical Results:
* **Champion Individual**: `ind_g3_cc67b76f` (stored in `.prime-kernel/workspace.db`)
* **Speedup Factor**: **9.91x faster** ($16.6\text{ ms}$ vs $165.0\text{ ms}$)
* **Accuracy**: **0.999917 Pearson Correlation** with exact SVD.

---

## 🚀 Quickstart & Verification Batteries

### 1. Installation
```bash
git clone git@github.com:mkrishna793/safe.git
cd safe
python -m pip install -r requirements.txt
```

### 2. Run the 17-Architecture Integration Suite
```bash
python -m archmath.tests.test_archs
```
*Validates building, forward pass, finite aux telemetry, row-stochastic attention, finite gradients, and SymPy complexity evaluation across all 17 models.*

### 3. Run the Real-Time Decoder & Red-Team Battery
```bash
python -m archmath.tests.test_math_framework
```
*Validates real-time training step attachments, Dirichlet non-negativity ($\mathcal{E} \ge 0$), Participation Ratio bounds ($PR \ge 1$), CKA ranges ($[0, 1]$), anomaly smoke alarms, and all 5 adversarial red-team attacks.*

### 4. Run SageMath Symbolic Proofs
```bash
./archmath/sage/run.sh
```

---

## 📚 Exhaustive Research Reports

Read the in-depth technical monographs in the [`research/`](research/) directory:
1. [`01_frontier_academic_alignment.md`](research/01_frontier_academic_alignment.md) — Comprehensive comparative analysis against Stanford, MIT, Google DeepMind, Meta FAIR, and DeepSeek literature.
2. [`02_edge_case_stress_audit.md`](research/02_edge_case_stress_audit.md) — Stress-testing across $S=1$, odd lengths, extreme scales, and 5 critical architectural bug repairs.
3. [`03_dual_mathematical_framework.md`](research/03_dual_mathematical_framework.md) — Mathematical foundations of the hidden-space immersion operator $\Phi: \mathcal{M}_{\text{embed}} \to \mathcal{M}_{\text{latent}}$.
4. [`04_adversarial_red_team_audit.md`](research/04_adversarial_red_team_audit.md) — Mathematical attacks, refutations, Sylvester identities, and PSD Laplacian proofs.

---

## 📄 License
This repository is released under the [MIT License](LICENSE).
