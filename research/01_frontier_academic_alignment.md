# Frontier SOTA Academic Alignment & Institutional Analysis of Modern AI Architectures

**Author**: Isolated Research Subagent 1 (Frontier SOTA Academic Alignment Scout)  
**Protocol**: PRAXIST Lineage Protocol | `archmath` Formal Verification Engine  
**Institutional Scope**: Stanford (CRFM/HAI), MIT (CSAIL/IAIFI), Anthropic Research, Google DeepMind, Meta FAIR, DeepSeek-AI, CMU LTI, Idiap/EPFL  
**Target Repository**: `archmath/impl/` and `archmath/symbolic/`  
**Date**: November 2026 (Reflecting 2017–2026 Institutional Breakthroughs)

---

## 1. Executive Summary & Cross-Institutional Architectural Taxonomy

Modern frontier foundation models are fundamentally shaped by the **Memory-Compute-Manifold Trilemma**:
1. **Compute Scalability**: Minimizing floating-point operations (FLOPs) per token during pre-training and inference generation.
2. **Memory Wall & KV Cache Footprint**: Mitigating High Bandwidth Memory (HBM) bandwidth saturation during autoregressive decoding where token generation is bound by memory traffic rather than compute.
3. **Manifold Geometry & Expressivity**: Preserving representation rank across layers, avoiding over-smoothing (Dirichlet energy collapse), and sustaining associative recall across long token horizons.

The `archmath` codebase formalizes and mathematically verifies 17 distinct architectural mechanics across dense decoders, recurrent depths, sparse routers, linear associative kernels, continuous state-space models, neural test-time memory, and tokenizer-free byte representations.

### Master Architecture Cross-Comparison Matrix

| # | Architecture | Primary Proponent | Prefill Time | Decode State / KV Cache | Memory Wall Sensitivity | Core Mathematical Mechanism |
|---|---|---|---|---|---|---|
| 1 | **Dense Decoder** | Google Brain / DeepMind / Meta | $O(S^2 d)$ | $O(S \cdot h_{kv} \cdot d_h)$ | High (Memory-Bound) | Causal Multi-Head / GQA + RoPE + SwiGLU |
| 2 | **Looped / Recurrent-Depth** | Google DeepMind / NYU | $O(n_{\text{loops}} S^2 d)$ | $O(S \cdot d)$ | Medium | Iterated contraction mapping, loop LoRA, deep supervision |
| 3 | **Native Sparse Attention** | Stanford / DeepSeek / AllenAI | $O(S \cdot k \cdot d)$ | $O(S \cdot d)$ (Selective) | Low-Medium | Learned top-$k$ block indexing, straight-through mask routing |
| 4 | **Linear Attn / Gated DeltaNet**| Idiap / NVIDIA / Microsoft | $O(S \cdot d_k d_v)$ | $O(d_k d_v)$ (Constant) | Very Low (Compute-Bound)| Associative feature map + online least-squares SGD state update |
| 5 | **SSM & Mamba-2 (SSD)** | Stanford / CMU / Princeton | $O(S \cdot d \cdot d_s)$| $O(d_s \cdot d)$ (Constant) | Very Low | Continuous ODE ZOH discretization, 1-SS scalar identity scan |
| 6 | **Sparse MoE** | Google Brain / Mistral / DeepSeek| $O(S \cdot k \cdot d)$ | $O(S \cdot d)$ | Medium (All-to-All) | Token-choice top-$k$ routing, shared experts, aux-loss-free bias |
| 7 | **Diffusion LM (LLaDA/DiT)** | Meta FAIR / UC Berkeley / NYU | $O(K S^2 d)$ | $O(S \cdot d)$ (Non-causal) | Medium (High FLOP) | Absorbing discrete forward process, score matching, ODE flow |
| 8 | **JEPA & World Models** | Meta FAIR / NYU / DeepMind | $O(S \cdot d)$ | $O(d)$ | Low | Non-generative latent prediction, EMA target, VICReg |
| 9 | **KAN (Kolmogorov-Arnold)** | MIT / Harvard | $O(S \cdot d^2 (G+k))$| $O(S \cdot d)$ | Medium-High (Splines) | Univariate B-splines on edges, Cox-de Boor recursion |
| 10 | **Titans Memory** | Google Research | $O(S \cdot d^2)$ | $O(d^2)$ (Fast Weights) | Low | Test-time online SGD on surprise metric + momentum |
| 11 | **Byte-level BLT** | Meta FAIR / UW | $O(P^2 d + N d_l^2)$ | $O(P \cdot d)$ ($P \ll N$) | Low | Tokenizer-free dynamic entropy-based byte patching |
| 12 | **Neuromorphic / SNN** | TU Graz / UC Irvine | $O(T S d^2)$ | $O(d)$ | Minimal (Event-Driven) | LIF membrane potential dynamics + surrogate gradients |
| 13 | **Multi-Head Latent Attention** | DeepSeek-AI | $O(S^2 d)$ | $O(S \cdot d_c)$ ($d_c \ll h d_h$)| Minimal (93% reduction)| Low-rank KV joint compression, decoupled RoPE, matrix absorption |
| 14 | **Multimodal Early Fusion** | Meta FAIR / Google DeepMind | $O(S_{\text{tot}}^2 d)$ | $O(S_{\text{tot}} d)$ | High | Unified continuous token intertwining, modality-aware attention |
| 15 | **Classical Recurrence (LSTM)** | TUM / IDSIA / MILA | $O(S \cdot d^2)$ | $O(d)$ (Constant) | Very Low (Sequential) | Constant Error Carousel (CEC), additive cell state, forget gating |
| 16 | **Deep Dense MLP** | Nature 1986 / EPFL | $O(S \cdot d \cdot d_{ff})$| $O(1)$ | Minimal | Point-wise coordinate transformations, NTK regime, rank restore |
| 17 | **Hybrid Attention-SSM** | AI21 Labs / NVIDIA / Upstage | $O(f S^2 d + (1-f) S d)$| $O(f S d + (1-f) d_s d)$| Balanced Pareto Optimal| Layer-ratio recipe $f = N_{\text{softmax}}/L$ balancing KV vs recall |

## 2. Exhaustive Institutional Analysis of the 17 AI Architecture Mechanics

### 1. Dense Decoder (Chinchilla, LLaMA 3, RoPE, SwiGLU)
- **Institutional Genesis**:
  - Google Brain: Vaswani et al. (2017) introduced multi-head attention.
  - Google DeepMind: Hoffmann et al. (2022) established Chinchilla compute-optimal scaling laws ($N \propto C^{0.5}, D \propto C^{0.5}$), proving that older models (e.g. GPT-3 175B) were undertrained.
  - Meta FAIR: Dubey et al. (2024 - LLaMA 3 herd of models) scaled dense decoders to 405B parameters on 15.6T tokens with 128k context, solidifying Grouped Query Attention (GQA) with 8 KV heads.
  - Key Innovations: Su et al. (2021) Rotary Position Embedding (RoPE); Shazeer (2019) Fast Transformer Decoding (GQA); Shazeer (2020) SwiGLU; Zhang & Sennrich (2019) RMSNorm.
- **Mathematical Manifold Mechanics**:
  $$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot g$$
  $$\mathrm{RoPE}(x, t)_{2i:2i+2} = \begin{pmatrix} \cos(t\theta^{-2i/d}) & -\sin(t\theta^{-2i/d}) \\ \sin(t\theta^{-2i/d}) & \cos(t\theta^{-2i/d}) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$
  $$\mathrm{SwiGLU}(x) = \big(\mathrm{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}\big) W_{\text{down}}$$
- **Implementation in `archmath`**:
  - `archmath/impl/dense.py` (`DenseTransformer`): Full pre-norm decoder-only stack with GQA, RoPE, RMSNorm, and SwiGLU MLP. Aux keys: `attn/probs`, `lm/logits`.
  - `archmath/symbolic/dense.py`: Invariant checks: causal lower-triangular mask, attention row stochasticity (sum to 1), KV cache byte scaling per token: $2 \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}} \times 2$ bytes.
- **Empirical Frontiers & Benchmarks**:
  - LLaMA 3 70B/405B establishes SOTA open-weights performance across MMLU, GSM8K, and HumanEval.
  - SwiGLU yields $+0.5$ to $+1.2$ perplexity improvements over standard GELU at equivalent FLOP budget ($d_{ff} = \frac{8}{3}d$).
- **Theoretical Boundaries & Bottlenecks**:
  - Double-exponential rank collapse $O(e^{-c L})$ in pure self-attention (Dong et al., ICML 2021) without skip connections.
  - Autoregressive memory wall: At batch size $B=64$, $S=128\text{k}$, FP16 KV cache consumes $>200\text{ GB}$, bottlenecking decode throughput strictly to GPU memory bandwidth (Roofline regime).

---

### 2. Looped / Recurrent-Depth Transformer
- **Institutional Genesis**:
  - Google Brain / DeepMind: Dehghani et al. (2018 - Universal Transformers) introduced ACT (Adaptive Computation Time) and recurrent layer depth.
  - Google Research: Lan et al. (2019 - ALBERT) demonstrated that cross-layer parameter sharing slashes parameters by $89\%$.
  - NYU / Stanford: Giannou et al. (2023 - Looped Transformers as Programmable Computers); Saunshi et al. (2025 - "How to Loop Your Transformer"); Geiping et al. (2025 - Recurrent Depth); Bae et al. (2025).
- **Mathematical Manifold Mechanics**:
  $$h^{(k+1)} = h^{(k)} + f_\theta(h^{(k)}) + \Delta_{\text{LoRA}}^{(k)}(h^{(k)})$$
  Iteration acts as an iterated map in continuous space. Under contractive dynamics (spectral radius $\rho(J) < 1$), hidden states converge toward fixed points $h^* = T(h^*)$.
- **Implementation in `archmath`**:
  - `archmath/impl/looped.py` (`LoopedTransformer`): Repeated application of a single core Transformer block $n_{\text{loops}}$ times, loop-specific LoRA adaptation ($r=8$), auxiliary deep supervision heads $\mathcal{L}_{\text{aux}}^{(k)}$, and a learned adaptive halting gate.
  - `archmath/symbolic/looped.py`: Tracks contraction: $\|h^{(k+1)} - h^{(k)}\|$ is monotone non-increasing; effective depth $= n_{\text{loops}} \times n_{\text{layers}}$.
- **Empirical Frontiers & Benchmarks**:
  - Looped models achieve $3\times$ parameter efficiency on parity, algorithmic graph traversal, and mathematical induction tasks.
  - ALBERT-xxlarge (47M params) achieves performance matching BERT-large (334M params).
- **Theoretical Boundaries & Bottlenecks**:
  - Fixed-point representation collapse: Once $h^{(k)}$ contracts within $\epsilon$-ball of $h^*$, marginal information gain vanishes.
  - Latency penalty: Decode latency scales linearly with $n_{\text{loops}}$ without reducing FLOPs.

---

### 3. Native Sparse Attention (NSA, FlashAttention, BigBird)
- **Institutional Genesis**:
  - Stanford University: Dao et al. (2022 - FlashAttention, arXiv:2205.14135; 2023 - FlashAttention-2; Shah et al., 2024 - FlashAttention-3) established IO-aware tiling, eliminating HBM memory roundtrips.
  - Google Research: Zaheer et al. (2020 - BigBird, arXiv:2007.14062) proved that random + sliding window + global tokens preserve full Turing completeness.
  - Allen AI: Beltagy et al. (2020 - Longformer) enabled linear scaling via dilated sliding window attention.
  - DeepSeek-AI: Yuan et al. (2025 - Native Sparse Attention, arXiv:2502.11089; DeepSeek-V3.2, 2025) designed hardware-aligned differentiable top-$k$ block indexing.
- **Mathematical Manifold Mechanics**:
  $$I_{tj} = \sum_{a=1}^{H_I} w_{ta} \, \mathrm{ReLU}(q_{ta}^\top k_j^I), \quad \mathcal{S}_t = \mathrm{TopK}_{j \le t}(I_{tj})$$
  $$p_{tj} = \frac{M_{tj} \exp(q_t^\top k_j / \sqrt{d_h})}{\sum_{\ell \le t} M_{t\ell} \exp(q_t^\top k_\ell / \sqrt{d_h})}, \quad M_{tj} = \mathbf{1}[j \in \mathcal{S}_t]$$
  Differentiable surrogate backpropagation via straight-through estimator: $\tilde{M} = M + q - \mathrm{sg}(q)$.
- **Implementation in `archmath`**:
  - `archmath/impl/sparse_attn.py` (`SparseAttentionTransformer`): Combines learned multi-head indexer projections, top-$k$ selection, sliding window $w$, block sparse strides, and BigBird global token routing.
  - `archmath/symbolic/sparse_attn.py`: Formulates total variation bound $\|p - p_{\mathcal{S}}\|_1 = 2\varepsilon_k$ and power-law recall bound $\varepsilon_k \le \frac{C}{\alpha-1} k^{-(\alpha-1)}$.
- **Empirical Frontiers & Benchmarks**:
  - DeepSeek NSA demonstrates $85\text{--}90\%$ compute reduction at $S = 64\text{k}$ context while preserving $>98.5\%$ dense attention perplexity.
  - FlashAttention-3 achieves $75\%$ FP16 tensor core utilization on NVIDIA Hopper H100 GPUs.
- **Theoretical Boundaries & Bottlenecks**:
  - Indexer mis-routing: Dropped key positions receive zero attention gradient, inducing permanently un-trainable blind spots.
  - High entropy tail truncation: On distributed uniform distributions, top-$k$ pruning incurs large bounded total variation error.

---

### 4. Linear Attention & Gated DeltaNet
- **Institutional Genesis**:
  - Idiap Research Institute / EPFL: Katharopoulos et al. (2020 - "Transformers are RNNs", arXiv:2006.16236) formalized kernel linearization $\phi(x) = \mathrm{elu}(x) + 1$.
  - IDSIA: Schlag et al. (2021) demonstrated that linear Transformers act as fast weight programmers.
  - Microsoft Research: Sun et al. (2023 - RetNet, arXiv:2307.08621) introduced multi-scale retention with chunkwise decay.
  - NVIDIA Research: Yang, Kautz, Hatamizadeh (2024 - Gated Delta Networks, arXiv:2412.06464; Yang et al., 2024 - arXiv:2406.06484) integrated per-channel diagonal decay with online least-squares SGD.
- **Mathematical Manifold Mechanics**:
  $$S_t = \alpha_t \odot S_{t-1} + \beta_t \, k_t (v_t - S_{t-1}^\top k_t)^\top, \quad o_t = S_t^\top q_t$$
  The state update is the closed-form online least-squares gradient step on loss $\ell_t(S) = \frac{1}{2}\|S^\top k_t - v_t\|^2$ with proximal decay $\alpha_t \in (0, 1)$.
- **Implementation in `archmath`**:
  - `archmath/impl/linear_attn.py` (`GatedDeltaNetLM`): Implements both sequential autoregressive linear recurrence and exact parallel chunkwise scan ($C \times C$).
  - `archmath/symbolic/linear_attn.py`: Verifies exact numerical equivalence between sequential recurrence and chunkwise scan ($< 2\cdot 10^{-4}$ max absolute error).
- **Empirical Frontiers & Benchmarks**:
  - Parity with Transformer++ on 7B parameter models while running at $O(1)$ constant memory during decode.
  - $2.5\text{--}4.2\times$ decode speedup over FlashAttention-2 at $S > 32\text{k}$.
- **Theoretical Boundaries & Bottlenecks**:
  - Finite associative memory capacity: The state matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ can store at most $d_k$ orthogonal key vectors before memory interference occurs.

---

### 5. State-Space Models & Mamba-2 (State Space Duality)
- **Institutional Genesis**:
  - Stanford University: Gu et al. (2020 - HiPPO, arXiv:2008.07669) introduced continuous polynomial projection memory; Gu, Goel, Re (2021 - S4, arXiv:2111.00396) introduced diagonal state spaces.
  - CMU / Stanford: Gu & Dao (2023 - Mamba, arXiv:2312.00752) introduced input-dependent selective state spaces $(\Delta, B, C)$.
  - Princeton / CMU: Dao & Gu (2024 - Mamba-2 / SSD, arXiv:2405.21060) discovered State Space Duality, establishing 1-semiseparable (1-SS) matrix equivalence to causal attention.
- **Mathematical Manifold Mechanics**:
  $$\dot{h}(t) = A h(t) + B x(t), \quad \bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \Delta B$$
  In Mamba-2 SSD, $A$ is restricted to scalar-identity per head: $A_t = a_t I$, enabling computation via block-matrix chunked semi-separable scan:
  $$Y = \mathrm{tril}\big((C B^\top) \odot M\big) U + (e^G \mathbf{1}^\top) \odot (C h_0)$$
- **Implementation in `archmath`**:
  - `archmath/impl/ssm.py` (`Mamba2LM`): Causal 1D depthwise conv (width 4), input-dependent selective discretization parameters, and chunked SSD scan.
  - `archmath/symbolic/ssm.py`: Verifies HiPPO matrix measure, contractive transition $\bar{A}_t \in (0, 1)$, and equivalence of naive recurrence to chunked scan ($< 2\cdot 10^{-5}$ error).
- **Empirical Frontiers & Benchmarks**:
  - Mamba-2 achieves $8\times$ faster training and prefill throughput than Mamba-1 by mapping directly onto GPU Tensor Cores.
  - Expands recurrent state dimension from $16$ to $128$ or $256$ with zero runtime penalty.
- **Theoretical Boundaries & Bottlenecks**:
  - Lack of content-based random access: Pure SSMs struggle on multi-hop associative recall and needle-in-a-haystack tasks compared to full softmax attention.

---

### 6. Sparse Mixture-of-Experts (Switch Transformer, Mixtral, DeepSeek-V3)
- **Institutional Genesis**:
  - Google Brain: Shazeer et al. (2017 - Sparsely-Gated MoE, arXiv:1701.06538); Lepikhin et al. (2020 - GShard); Fedus, Zoph, Shazeer (2021 - Switch Transformer, arXiv:2101.03961 - Top-1 routing).
  - Mistral AI: Jiang et al. (2024 - Mixtral of Experts, arXiv:2401.04088 - Top-2 routing with 8 experts).
  - DeepSeek-AI: Dai et al. (2024 - DeepSeekMoE, arXiv:2401.06066 - Fine-grained routing + shared experts); Wang et al. (2024 - Auxiliary-Loss-Free Balancing, arXiv:2408.15664); DeepSeek-V3 (2024/2025).
- **Mathematical Manifold Mechanics**:
  $$p = \mathrm{softmax}(W_r x + b), \quad \mathcal{T} = \mathrm{TopK}(p, k), \quad g_i = \frac{p_i}{\sum_{j \in \mathcal{T}} p_j}$$
  $$y = \sum_{i \in \mathcal{T}} g_i E_i(x) + E_{\text{shared}}(x)$$
  DeepSeek-V3 dynamic auxiliary-loss-free bias adjustment:
  $$b_i \leftarrow b_i + \gamma \cdot \mathrm{sign}(\mathrm{load}_i - \overline{\mathrm{load}})$$
- **Implementation in `archmath`**:
  - `archmath/impl/moe.py` (`SparseMoETransformer`): Token-choice routing, fine-grained routed experts ($N=8$, top-$k=2$), dedicated shared expert, auxiliary load balancing loss, and dynamic non-differentiable routing bias buffers.
  - `archmath/symbolic/moe.py`: Proves active parameters: $\text{active} = (k + n_{\text{shared}}) \cdot 3 d \cdot d_{ff}$ vs total parameters $(N + n_{\text{shared}}) \cdot 3 d \cdot d_{ff}$.
- **Empirical Frontiers & Benchmarks**:
  - DeepSeek-V3 (671B total, 37B active) achieves GPT-4o parity at $1/10$ the pre-training compute cost.
  - Mixtral 8x7B (46.7B total, 12.9B active) matches Llama 2 70B while running $3\times$ faster during decode.
- **Theoretical Boundaries & Bottlenecks**:
  - All-to-all communication bound: Token dispatch and combination across nodes saturate InfiniBand / NVLink bandwidth in large clusters.
  - Expert routing collapse: Without auxiliary loss or dynamic bias correction, routing distribution degenerates to a small subset of experts.

---

### 7. Diffusion Transformers & Discrete Flow Matching (DiT, LLaDA)
- **Institutional Genesis**:
  - UC Berkeley / NYU: Peebles & Xie (2023 - Scalable Diffusion Models with Transformers / DiT, arXiv:2212.09748).
  - Meta FAIR: Lipman et al. (2022 - Flow Matching for Generative Modeling, arXiv:2210.02747).
  - Tsinghua / Renmin / Tencent: Nie et al. (2025 - Large Language Diffusion Models / LLaDA, arXiv:2502.09992).
  - Khanna et al. (2025 - Mercury, arXiv:2506.17298).
- **Mathematical Manifold Mechanics**:
  Discrete absorbing forward jump process:
  $$q_t(x_t^i = \mathrm{M} \mid x_0) = 1 - t, \quad q_t(x_t^i = x_0^i \mid x_0) = t$$
  Evidence Lower Bound (ELBO) with importance reweighting:
  $$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], x_0, x_t} \left[ \frac{1}{1-t} \sum_{i: x_t^i = \mathrm{M}} -\log p_\theta(x_0^i \mid x_t) \right]$$
  Confidence-based parallel sampling unmasks tokens with highest predicted probability $\max_v p_\theta(x_i = v \mid x_t)$.
- **Implementation in `archmath`**:
  - `archmath/impl/diffusion.py` (`MaskedDiffusionLM`): Bidirectional Transformer backbone, absorbing mask forward jump process, continuous-time ELBO loss, and iterative confidence-scheduled parallel sampler.
  - `archmath/symbolic/diffusion.py`: Verifies: attention mask is all-ones (bidirectional), $1/(1-t)$ reweighting yields unbiased per-token NLL estimate.
- **Empirical Frontiers & Benchmarks**:
  - LLaDA 8B matches LLaMA-3 8B in in-context learning and reasoning while natively supporting bidirectional conditioning, text infilling, and non-left-to-right generation.
  - Mercury achieves ultra-fast diffusion sampling on text by combining confidence scheduling with speculative unmasking.
- **Theoretical Boundaries & Bottlenecks**:
  - Iterative decode FLOPs: Generating $S$ tokens requires $K$ forward passes through the full bidirectional Transformer, increasing generation FLOPs by $K \times$.

---

### 8. Joint Embedding Predictive Architecture (JEPA & World Models)
- **Institutional Genesis**:
  - Meta FAIR / NYU: LeCun (2022 - "A Path Towards Autonomous Machine Intelligence"); Assran et al. (2023 - I-JEPA, arXiv:2301.08243); Bardes et al. (2024 - V-JEPA, arXiv:2304.07193).
  - Meta FAIR: Bardes et al. (2022 - VICReg, arXiv:2105.04906).
  - Google DeepMind: Hafner et al. (2019–2023 - DreamerV1–V3) world models.
- **Mathematical Manifold Mechanics**:
  Non-generative prediction in abstract latent space:
  $$s_x = f_\theta(x), \quad s_y = f_\xi(y), \quad \xi \leftarrow \tau \xi + (1-\tau)\theta, \quad \hat{s}_y = g_\phi(s_x, a)$$
  $$\mathcal{L} = \|\hat{s}_y - \mathrm{sg}(s_y)\|^2 + \lambda_v \mathcal{V}(s_x) + \lambda_c \mathcal{C}(s_x)$$
  $$\mathcal{V}(Z) = \frac{1}{d}\sum_{j=1}^d \max\big(0, 1 - \sqrt{\mathrm{Var}(Z)_j}\big), \quad \mathcal{C}(Z) = \frac{1}{d}\sum_{i \ne j} \mathrm{Cov}(Z)_{ij}^2$$
- **Implementation in `archmath`**:
  - `archmath/impl/jepa.py` (`JEPATransformer`): Online representation encoder, EMA target encoder, action-conditioned latent transition predictor, and VICReg variance/covariance regularizers.
  - `archmath/symbolic/jepa.py`: Proves mathematical collapse condition: $\mathrm{Var}(f_\theta(x)) \ge 1$ guarantees prevention of dimensional representation collapse.
- **Empirical Frontiers & Benchmarks**:
  - V-JEPA sets SOTA on frozen video representation benchmarks without pixel reconstruction.
  - Achieves $10\times$ compute savings over pixel-space generative diffusion or masked autoencoders (MAE) for world state forecasting.
- **Theoretical Boundaries & Bottlenecks**:
  - Perceptual blindness: By abstracting away token/pixel space, JEPA cannot render or generate visual/textual outputs directly without auxiliary generative heads.

---

### 9. Kolmogorov-Arnold Networks (KAN)
- **Institutional Genesis**:
  - MIT / Harvard: Liu, Wang, Vaidya, Ruehle, Halverson, Soljacic, Hou, Tegmark (2024 - KAN, arXiv:2404.19756).
  - Mathematics: Kolmogorov (1957); Arnold (1957); Cox (1972) & de Boor (1972) B-spline recursions.
- **Mathematical Manifold Mechanics**:
  $$y_j = \sum_{i=1}^n \phi_{ij}(x_i), \quad \phi_{ij}(x) = w_b \, \mathrm{SiLU}(x) + w_s \sum_{k} c_k B_k(x)$$
  Univariate B-splines evaluate recursively via Cox-de Boor algorithm over grid points $t_0, \dots, t_{G+k}$.
- **Implementation in `archmath`**:
  - `archmath/impl/kan.py` (`KANTransformer`): Replaces standard dense MLP layers with learnable spline activations across input-output coordinate projections.
  - `archmath/symbolic/kan.py`: Derives parameter complexity: $P_{\text{KAN}} = n \cdot m \cdot (G + k + 1)$ vs $P_{\text{MLP}} = 4 n m$, yielding crossover condition $G = 3 - k$.
- **Empirical Frontiers & Benchmarks**:
  - KAN achieves superior Pareto frontier on PDE solving, symbolic regression, and knot theory data fitting, recovering closed-form mathematical equations via $L_1$ spline pruning.
- **Theoretical Boundaries & Bottlenecks**:
  - GPU GEMM incompatibility: Spline basis evaluations cannot map cleanly onto Tensor Cores, resulting in slow training throughput compared to linear projections.
  - High parameter-to-FLOP ratio: For $G=5, k=3$, KAN requires $9\times$ more parameters per channel than linear weights.

---

### 10. Titans Memory (Neural Test-Time Memory Architecture)
- **Institutional Genesis**:
  - Google Research: Behrouz, Santacroce, Mirhoseini et al. (2025 - Titans, arXiv:2501.00663).
  - Historical roots: Ba et al. (2016 - Fast Weights); Schlag et al. (2021).
- **Mathematical Manifold Mechanics**:
  Dual-branch architecture combining sliding window attention (short-term) and test-time online gradient descent on neural fast weight matrix $M_t$:
  $$\ell_t(M) = \frac{1}{2}\|M k_t - v_t\|^2, \quad \nabla_M \ell_t(M_{t-1}) = (M_{t-1} k_t - v_t) k_t^\top$$
  $$S_t = \eta_t S_{t-1} - \theta_t \nabla_M \ell_t(M_{t-1}), \quad M_t = (1 - \alpha_t) M_{t-1} + S_t$$
  Gated branch output: $y_t = g_t a_t + (1 - g_t) (M_{t-1} q_t)$.
- **Implementation in `archmath`**:
  - `archmath/impl/titans.py` (`TitansTransformer`): Online test-time gradient updates, surprise metrics $\mathcal{S}_t = \|M_{t-1} k_t - v_t\|^2$, momentum accumulation, and adaptive branch mixing gates.
  - `archmath/symbolic/titans.py`: Proves bounded Frobenius norm $\|M_t\|_F < \infty$ and strictly causal memory updates.
- **Empirical Frontiers & Benchmarks**:
  - Scaling to $>2\text{M}$ tokens with constant inference memory footprint, outperforming pure Transformers and Mamba on needle-in-a-haystack tasks.
- **Theoretical Boundaries & Bottlenecks**:
  - Gate collapse: If surprise gate $g_t \to 1$, the model degenerates to short-term attention, losing long-horizon memories; if decay $\alpha_t$ is too high, test-time weight overwrites past facts.

---

### 11. Byte Latent Transformer (BLT)
- **Institutional Genesis**:
  - Meta FAIR / University of Washington: Pagnoni, Ram, Pasunuru, Toutanova et al. (2024 - BLT, arXiv:2412.09871).
  - Fast BLT: Kallini et al. (2026 - arXiv:2605.08044).
- **Mathematical Manifold Mechanics**:
  Tokenizer-free byte architecture. Local causal convolution predicts next-byte entropy:
  $$H_i = -\sum_{v=1}^{256} p_\phi(x_i = v \mid x_{<i}) \log p_\phi(x_i = v \mid x_{<i})$$
  Dynamic patch boundary placed when $H_i > \theta$ or patch length reaches $P_{\max}$. Global Transformer attends strictly across patches $P \ll N$:
  $$C = O(P^2 d + P d^2) + O(N d_{\text{local}}^2)$$
- **Implementation in `archmath`**:
  - `archmath/impl/blt.py` (`ByteLatentTransformer`): Local byte n-gram encoder, entropy-based patch segmenter, global latent cross-attention Transformer, and lightweight local byte decoder.
  - `archmath/symbolic/blt.py`: Verifies invariant: byte entropy is strictly bounded in $[0, \ln 256]$, patch lengths $\in [1, P_{\max}]$, monotone patch boundaries.
- **Empirical Frontiers & Benchmarks**:
  - Matches Llama 3 tokenized performance at equivalent compute while demonstrating complete immunity to tokenizer artifacts, character typos, and multilingual tokenization skew (yielding $3\text{--}5\times$ byte patch compression).
- **Theoretical Boundaries & Bottlenecks**:
  - Dynamic sequence batching: Dynamic patch lengths cause ragged sequence arrays across batches, requiring specialized kernel padding or nested ragged tensors.

---

### 12. Neuromorphic / Spiking Neural Networks (SNN)
- **Institutional Genesis**:
  - TU Graz: Maass (1997 - "Networks of Spiking Neurons", Neural Networks).
  - UC Irvine: Neftci et al. (2019 - Surrogate Gradients, IEEE SPM).
  - Gerstner & Kistler (2002); Zhou et al. (2022 - Spikformer).
- **Mathematical Manifold Mechanics**:
  First-order discrete Leaky Integrate-and-Fire (LIF) dynamics:
  $$V_t = \beta V_{t-1} + W x_t - V_{th} s_{t-1}, \quad s_t = \Theta(V_t - V_{th})$$
  Surrogate gradient backward pass replaces non-differentiable Dirac delta with fast sigmoid / arctan derivative:
  $$\frac{\partial s}{\partial V}\Big|_{\mathrm{bwd}} = \frac{\alpha}{(1 + \alpha |V - V_{th}|)^2}$$
- **Implementation in `archmath`**:
  - `archmath/impl/neuromorphic.py` (`NeuromorphicTransformer`): LIF membrane dynamics, soft/hard reset, surrogate backpropagation, and synaptic energy proxy $E \approx n_{\text{spikes}} \cdot E_{\text{MAC}}$.
  - `archmath/symbolic/neuromorphic.py`: Tracks participation ratio and validates finite membrane potential boundaries.
- **Empirical Frontiers & Benchmarks**:
  - $10\text{--}50\times$ lower energy consumption on neuromorphic hardware (e.g. Intel Loihi 2, SynSense) compared to standard GPUs during sparse event-driven processing.
- **Theoretical Boundaries & Bottlenecks**:
  - Surrogate gradient bias: The backward surrogate gradient does not equal the true gradient of the discontinuous Heaviside function, causing gradient degradation and dead-neuron syndrome in deep networks ($L > 20$).

---

### 13. Multi-Head Latent Attention (MLA)
- **Institutional Genesis**:
  - DeepSeek-AI: DeepSeek-V2 (2024 - arXiv:2405.04434); DeepSeek-V3 (2024/2025 - arXiv:2412.19437); Yuan et al. (2025 - arXiv:2502.11089).
- **Mathematical Manifold Mechanics**:
  Compresses keys and values into a shared low-rank latent vector $c_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c = 512$):
  $$c_t^{KV} = W_{DKV} x_t, \quad k_t^C = W_{UK} c_t^{KV}, \quad v_t^C = W_{UV} c_t^{KV}$$
  Decoupled Rotary Key $k_t^R = \mathrm{RoPE}(W_{KR} x_t, t)$ carries positional information.
  At inference time, up-projection weights are absorbed into query projections:
  $$W_{\text{absorbed}} = W_{UQ} W_{UK}^\top \in \mathbb{R}^{d_q \times d_c}$$
- **Implementation in `archmath`**:
  - `archmath/impl/mla.py` (`MLATransformer`): Joint low-rank KV compression, decoupled RoPE, and inference matrix absorption.
  - `archmath/symbolic/mla.py`: Proves exact numerical equivalence between absorbed and non-absorbed attention, Eckart-Young SVD truncation optimality, and $93.3\%$ KV cache size reduction.
- **Empirical Frontiers & Benchmarks**:
  - KV cache footprint per token reduced from $2 \cdot n_h \cdot d_h = 128 \times 128 = 16\text{k}$ elements to $d_c + d_R = 512 + 64 = 576$ elements ($93.3\%$ reduction), enabling $5\text{--}8\times$ larger inference generation batch sizes.
- **Theoretical Boundaries & Bottlenecks**:
  - Rank truncation error: By Eckart-Young theorem, compression incurs error $\sigma_{d_c+1}(KV)$. If token embeddings possess high effective rank, compression loss increases.

---

### 14. Multimodal Early Fusion (Chameleon, Flamingo, Gemini)
- **Institutional Genesis**:
  - Meta FAIR: Chameleon Team (2024 - arXiv:2405.09818).
  - Google DeepMind: Alayrac et al. (2022 - Flamingo, arXiv:2206.01714); Gemini Team (2023 - Gemini).
  - An et al. (2026 - Toward Native Multimodal Modeling, arXiv:2605.25343); Shukor et al. (2025 - arXiv:2504.07951).
- **Mathematical Manifold Mechanics**:
  Unified token intertwining across vision patches, audio frames, and text tokens:
  $$\mathbf{z} = [E_v(v_1), \dots, E_a(a_1), \dots, E_t(x_1)] \in \mathbb{R}^{S_{\text{total}} \times d}$$
  Modality-aware attention mask: vision/audio prefix modalities attend bidirectionally within themselves, while text attends to all prefix modalities and autoregressively to prior text.
- **Implementation in `archmath`**:
  - `archmath/impl/multimodal.py` (`NativeMultimodal`): Cross-modal token projection, modality routing masks, manifold alignment loss $\mathcal{D}_{align}$, and balanced multi-task loss weighting.
  - `archmath/symbolic/multimodal.py`: Verifies invariant: no image or audio token attends to text, sequence length equals sum of all modality tokens.
- **Empirical Frontiers & Benchmarks**:
  - Chameleon 34B achieves state-of-the-art mixed-modal reasoning, document understanding, and image generation within a single unified autoregressive decoder.
- **Theoretical Boundaries & Bottlenecks**:
  - Modality dominance & gradient conflict: Image tokens often generate significantly larger gradient norms than text tokens, destabilizing training without per-modality loss balancing.

---

### 15. Classical Recurrence (RNN, LSTM, Constant Error Carousel)
- **Institutional Genesis**:
  - TUM / IDSIA: Hochreiter & Schmidhuber (1997 - Long Short-Term Memory, Neural Computation).
  - IDSIA: Gers, Schmidhuber, Cummins (2000 - "Learning to Forget: Continual Prediction with LSTM").
  - MILA: Pascanu, Mikolov, Bengio (2013 - "On the Difficulty of Training Recurrent Neural Networks").
- **Mathematical Manifold Mechanics**:
  $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \quad h_t = o_t \odot \tanh(c_t)$$
  The Constant Error Carousel (CEC) enforces additive linear memory updates:
  $$\frac{\partial c_t}{\partial c_{t-1}} = \mathrm{diag}(f_t) \implies \frac{\partial c_T}{\partial c_t} = \prod_{k=t+1}^T \mathrm{diag}(f_k) = \exp\left( \sum_{k=t+1}^T \log f_k \right)$$
- **Implementation in `archmath`**:
  - `archmath/impl/rnn.py` (`RNNModel`): Stacked LSTM cells with residual skip connections, forget/input/output gating, and Jacobian spectral radius tracking.
  - `archmath/symbolic/rnn.py`: Verifies: gates bounded in $(0, 1)$, hidden states bounded in $(-1, 1)$, linear cell gradient backpropagation.
- **Empirical Frontiers & Benchmarks**:
  - Historical cornerstone of sequence modeling; superseded in large foundation models due to $O(S)$ sequential training bottleneck on GPUs.
- **Theoretical Boundaries & Bottlenecks**:
  - Sequential execution bottleneck: Non-parallelizable state updates prevent efficient mapping to massively parallel GPU/TPU architectures.

---

### 16. Deep Dense MLP (Residual MLP, Neural Tangent Kernel)
- **Institutional Genesis**:
  - Rumelhart, Hinton, Williams (1986 - Learning representations by back-propagating errors, Nature).
  - EPFL: Jacot, Gabriel, Hongler (2018 - Neural Tangent Kernel, NeurIPS, arXiv:1806.07572).
  - Dong, Cordonnier, Ma (ICML 2021 - arXiv:2103.03404).
- **Mathematical Manifold Mechanics**:
  Strict point-wise coordinate transformation without sequence token mixing:
  $$x_{l+1} = x_l + \mathrm{SwiGLU}(\mathrm{RMSNorm}(x_l))$$
  In the infinite-width limit, dynamics are governed by the deterministic Neural Tangent Kernel:
  $$\Theta(x, x') = \mathbb{E}_{\theta} \left[ \nabla_\theta f(x) \nabla_\theta f(x')^\top \right]$$
- **Implementation in `archmath`**:
  - `archmath/impl/mlp.py` (`MLPModel`): Deep residual SwiGLU MLP stacks, tracking participation ratio and effective rank.
  - `archmath/symbolic/mlp.py`: Proves token independence along sequence axis; preserves and expands representation rank to counteract self-attention collapse.
- **Empirical Frontiers & Benchmarks**:
  - Standard Transformers allocate $\sim 67\%$ of total parameters and FLOPs to feed-forward MLP blocks.
- **Theoretical Boundaries & Bottlenecks**:
  - Inability to perform cross-token reasoning: Point-wise MLPs cannot propagate information across sequence positions without attention or recurrence.

---

### 17. Hybrid Attention-SSM (Jamba, Nemotron-H, Solar Open 2)
- **Institutional Genesis**:
  - AI21 Labs: Lieber et al. (2024 - Jamba, arXiv:2403.19887).
  - Zyphra: Glorioso et al. (2024 - Zamba, arXiv:2405.16712).
  - Microsoft Research: Ren et al. (2024 - Samba, arXiv:2406.07522).
  - NVIDIA: (2025 - Nemotron-H, arXiv:2504.03624).
  - Upstage: Park et al. (2026 - Solar Open 2 Technical Report, arXiv:2607.20062).
  - Ye et al. (2026 - arXiv:2602.01763).
- **Mathematical Manifold Mechanics**:
  Interleaved stack characterized by softmax fraction $f = \frac{N_{\text{softmax}}}{L}$:
  $$M_{\text{hyb}}(S) = f \cdot L \cdot \kappa S + (1-f) L \cdot \rho$$
  KV cache crossover threshold: $S > \frac{\rho}{\kappa}$, where $\rho$ is recurrent state bytes and $\kappa$ is attention KV cache bytes per token.
- **Implementation in `archmath`**:
  - `archmath/impl/hybrid.py` (`HybridStack`): Builds heterogeneous stacks combining softmax attention, Gated DeltaNet, and Mamba-2 layers according to configurable ratio recipes.
  - `archmath/symbolic/hybrid.py`: Formulates Pareto frontier: saving $4(1-f) n_h d_h S^2$ attention FLOPs while guaranteeing full associative recall capacity via designated softmax layers.
- **Empirical Frontiers & Benchmarks**:
  - Jamba 52B (12B active via MoE) achieves $3\times$ throughput on long sequences while matching Llama 2 70B quality.
  - Solar Open 2 establishes $f = 1/4$ as the Pareto-optimal operating point for $128\text{k}$ context.
- **Theoretical Boundaries & Bottlenecks**:
  - Layer layout sensitivity: Grouping softmax layers sub-optimally (e.g. only at early layers) severely damages multi-hop reasoning over long token spans.

## 3. Manifold Geometry & Latent Invariant Telemetry

The `archmath/manifold.py` module integrates differential geometry, spectral graph theory, and information theory into real-time runtime telemetry for arbitrary neural network architectures:

1. **Pullback Metric Tensor & Local Volume Distortion**:
   $$J = \frac{\partial z_L}{\partial x_{\text{embed}}}, \quad G = J^\top J, \quad \mathrm{VolDist} = \sqrt{\det(G + \epsilon I)}$$
   Evaluated via Sylvester's determinant theorem: $\det(I + \epsilon^{-1} J J^\top)$. Detects non-contractive singularities and exploding gradients across layers.

2. **Graph Dirichlet Energy & Over-Smoothing**:
   $$E(Z) = \frac{1}{2} \mathrm{Tr}(Z^\top L Z) = \frac{1}{4} \sum_{i,j} (A_{ij} + A_{ji}) \|z_i - z_j\|^2$$
   Normalized Dirichlet Energy $\frac{E(Z)}{\|Z\|_F^2} \to 0$ flags representation collapse (tokens collapsing into a single uniform representation).

3. **Centered Kernel Alignment (CKA) & Geometric Velocity**:
   $$\mathrm{CKA}(K_l, K_{l+1}) = \frac{\mathrm{HSIC}(K_l, K_{l+1})}{\sqrt{\mathrm{HSIC}(K_l, K_l) \mathrm{HSIC}(K_{l+1}, K_{l+1})}}, \quad v_{\text{geom}} = 1 - \mathrm{CKA}$$
   Measures layer-to-layer representation rotation and drift velocity.

4. **Spectral Intrinsic Dimensionality (Participation Ratio)**:
   $$\mathrm{PR} = \frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2}, \quad H_{\text{spec}} = -\sum_i p_i \ln p_i$$
   Extracts the effective dimensional subspace carrying variance, identifying dimensional collapse.

---

## 4. PRAXIST Research Lineage Classification

Under the PRAXIST Lineage Protocol, the 17 architectural mechanics are systematically classified into durable research lanes:

### 🌟 Frontier Lane (Pareto-Optimal, Production SOTA)
- **Multi-Head Latent Attention (MLA)**: Definitive solution to the KV cache memory wall (93.3% footprint reduction).
- **Sparse MoE with Aux-Loss-Free Balancing**: Unlocks GPT-4 class capabilities at 10x lower training/inference compute.
- **Hybrid Attention-SSM (SSD)**: Emerging industrial standard for sub-quadratic long-context foundation models.
- **Native Sparse Attention (NSA)**: Hardware-aligned selective attention with sub-quadratic scaling.

### 🧪 Incubator Lane (Active Multi-Generational Hypotheses)
- **Titans Test-Time Memory**: Neural memory with surprise-driven SGD; outstanding potential for million-token recall.
- **Byte Latent Transformer (BLT)**: Tokenizer-free architecture eliminating tokenization pathologies.
- **Linear Attention / Gated DeltaNet**: Fast weight recurrence; competitive with SSMs when armed with delta-rule updates.
- **Diffusion Transformers for Discrete Text (LLaDA)**: Iterative non-autoregressive decoding and bidirectional reasoning.

### 💎 Gems Lane (Rare Mathematical & Structural Breakthroughs)
- **State Space Duality (SSD / Mamba-2)**: The exact 1-SS matrix duality connecting continuous SSMs to attention.
- **Kolmogorov-Arnold Networks (KAN)**: Mathematical elegance of edge-based B-splines; supreme for scientific computing.
- **Joint-Embedding Predictive Architecture (JEPA)**: Pure non-generative representation learning without perceptual hallucination.

### 🚫 Negative Evidence Lane (Confirmed Anti-Patterns & Bottlenecks)
- **Pure Self-Attention without MLPs / Residuals**: Confirmed double-exponential rank collapse (Dong et al., 2021).
- **Pure SSMs without Softmax Attention for Open-Domain LLMs**: Persistent failure on complex needle-in-a-haystack associative recall.
- **Unregularized MoE Routing (Aux-Loss Only)**: Prone to routing collapse, expert starvation, or severe all-to-all communication bottlenecks.
- **Deep Neuromorphic SNNs for General Language Modeling**: Surrogate gradient bias and non-GEMM execution cause fatal training degradation.

---

## 5. Bibliography & Academic Citations

1. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Hoffmann, J., et al. (2022). Training compute-optimal large language models. *NeurIPS*. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
3. Dubey, A., et al. (2024). The Llama 3 herd of models. *Meta AI Technical Report*. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
4. Shazeer, N. (2019). Fast transformer decoding: One write-head is all you need. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
5. Shazeer, N. (2020). GLU variants improve transformer. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
6. Su, J., et al. (2021). RoFormer: Enhanced transformer with rotary position embedding. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
7. Dao, T., et al. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
8. Dao, T. (2023). FlashAttention-2: Faster attention with better parallelism. *ICLR*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
9. Gu, A., Goel, K., & Re, C. (2021). Efficiently modeling long sequences with structured state spaces. *ICLR*. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
10. Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
11. Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized models through structured state space duality. *ICML*. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
12. Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR*. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
13. Fedus, W., Zoph, B., & Shazeer, N. (2021). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *JMLR*. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
14. Jiang, A. Q., et al. (2024). Mixtral of experts. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
15. DeepSeek-AI (2024). DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
16. DeepSeek-AI (2024). DeepSeek-V3 technical report. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
17. Wang, Y., et al. (2024). Auxiliary-loss-free load balancing for mixture-of-experts. [arXiv:2408.15664](https://arxiv.org/abs/2408.15664)
18. Yuan, Y., et al. (2025). Native sparse attention: Hardware-aligned and differentiable. [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)
19. Katharopoulos, A., et al. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. *ICML*. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
20. Yang, S., Kautz, J., & Hatamizadeh, A. (2024). Gated delta networks: Improving Mamba2 with delta rule. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
21. Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. *ICCV*. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
22. Nie, S., et al. (2025). Large language diffusion models. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
23. LeCun, Y. (2022). A path towards autonomous machine intelligence. *OpenReview*.
24. Assran, M., et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. *CVPR*. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
25. Bardes, A., et al. (2024). V-JEPA: Latent video prediction. *CVPR*. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
26. Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold networks. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
27. Behrouz, A., et al. (2025). Titans: Learning to memorize at test time. [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
28. Pagnoni, A., et al. (2024). Byte latent transformer: Patches vs. tokens. [arXiv:2412.09871](https://arxiv.org/abs/2412.09871)
29. Chameleon Team (2024). Chameleon: Mixed-modal early-fusion foundation models. [arXiv:2405.09818](https://arxiv.org/abs/2405.09818)
30. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
31. Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. *NeurIPS*. [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)
32. Dong, Y., Cordonnier, J. B., & Ma, A. (2021). Attention is not all you need: Pure attention loses rank doubly exponentially with depth. *ICML*. [arXiv:2103.03404](https://arxiv.org/abs/2103.03404)
33. Lieber, O., et al. (2024). Jamba: A hybrid Transformer-Mamba language model. [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)
34. Park, K., et al. (2026). Solar Open 2 technical report. [arXiv:2607.20062](https://arxiv.org/abs/2607.20062)
