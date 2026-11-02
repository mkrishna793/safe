# ArchMath — Architecture Atlas

A mathematical decoding and real-time hidden-space tracing framework for modern AI architectures.

## Layout

```
archmath/
  core.py          contracts: ArchConfig, MathSpec, ArchModule, registry
  trace.py         real-time hidden-space tracer (forward hooks + grads)
  symbolic/        one MathSpec per architecture (equations, complexity, invariants)
  impl/            executable PyTorch blocks (register + tiny_config)
  docs/            per-architecture mathematical derivations (this directory)
  sage/            SageMath symbolic verification (derive.sage, all checks pass)
  bench.py         cross-architecture training benchmark with telemetry
  tests/           integration suite (all 17 architectures verified)
  cli.py           list / math / complexity / verify / trace / export
```

## The 17 Architectures: Master Comparison

| Architecture | Family | Token Mixing Mechanism | State Capacity | Inference Complexity | Memory Complexity | Key Mathematical Invariant |
|---|---|---|---|---|---|---|
| [dense](dense.md) | `dense` | Causal Softmax Attention ($QK^\top/\sqrt{d_h}$) | $O(S)$ tokens | $O(L(d^2 + d \cdot d_{ff} + S d))$ / tok | $O(L \cdot h_{kv} \cdot d_h \cdot S)$ (KV cache) | $\sum_j A_{ij}=1$, causal mask $M_{ij}=-\infty\ (j>i)$ |
| [mlp](mlp.md) | `mlp` | None (pointwise SwiGLU / GeLU) | $0$ (stateless across sequence) | $O(L \cdot d \cdot d_{ff})$ / tok ($O(1)$ seq steps) | $O(1)$ (zero KV cache, zero state) | $\frac{\partial x_{l,t}}{\partial x_{l-1,s}} = 0\ \forall t \ne s$; NTK linear flow |
| [rnn](rnn.md) | `rnn` | Recurrence via Constant Error Carousel (CEC) | $O(d)$ fixed vector ($h_t, c_t$) | $O(L \cdot d^2)$ / tok ($O(S)$ seq steps) | $O(L \cdot d)$ (constant in $S$, no KV cache) | $\frac{\partial c_T}{\partial c_t} = \prod_{k=t+1}^T \operatorname{diag}(f_k)$; gates $\in (0, 1)$ |
| [moe](moe.md) | `sparse-moe` | Causal Softmax Attention + Top-$k$ Router | $O(S)$ context, $E$ experts | $O(L(S d + k \cdot d \cdot d_{ff}))$ active FLOPs | $O(L \cdot h_{kv} d_h S) + O(E \cdot d \cdot d_{ff})$ params | $\sum_{j \in \mathrm{TopK}} g_j = 1$; router load balance |
| [mla](mla.md) | `latent-attention` | Low-Rank Compressed Attention + Decoupled RoPE | $O(S)$ compressed to rank $d_c \ll h d_h$ | $O(S \cdot d_c)$ via matrix absorption | $O(L(d_c + d_r)S)$ compressed KV cache | Absorbed $W_{UK}^\top W_Q$ strictly equals uncompressed attention |
| [sparse_attn](sparse_attn.md) | `sparse-attention` | Lightning Indexer + Exact Top-$k$ Attention | $k_{sparse} \ll S$ selected historical tokens | $O(S \cdot d_{idx} + k_{sparse} \cdot d_h)$ / tok | $O(L \cdot h_{kv} d_h S)$ cache, $O(k_{sparse})$ active | $\sum_{j \in \mathcal{K}_t} A_{tj} = 1$; sparsity $k_{sparse}/S \to 0$ |
| [linear_attn](linear_attn.md) | `linear_attention` | Gated DeltaNet (online least-squares recurrence) | $O(d_k d_v)$ matrix state $S_t$ | $O(d_k d_v)$ / tok ($O(S d_k d_v)$ sequence) | $O(L \cdot h \cdot d_k d_v)$ (constant in $S$) | Delta rule minimizes $\|S k_t - v_t\|^2$; recurrent $=$ chunked |
| [ssm](ssm.md) | `state_space` | Causal Conv-4 + Diagonal SSM ($h_t = \alpha_t h_{t-1} + B_t x_t$) | $O(d \cdot d_{state})$ diagonal state | $O(L \cdot d \cdot d_{state})$ / tok | $O(L(d \cdot d_{state} + k_{conv} d))$ (constant in $S$) | Recurrence equals 1-semiseparable chunked scan; $\alpha_t \in (0, 1)$ |
| [hybrid](hybrid.md) | `hybrid` | Interleaved Softmax Attention + Recurrent SSM | $O(S)$ for attention + $O(d_{state})$ for SSM | $O(n_{soft} S d + (L-n_{soft}) d \cdot d_{state})$ | $O(n_{soft} h_{kv} d_h S + (L-n_{soft}) d \cdot d_{state})$ | Allocation schedule preserves exact ratio; KV memory drops by $1 - n_{soft}/L$ |
| [looped](looped.md) | `looped` | Recursive Shared Attention Block ($K$ iterations) | $O(S)$ context, iterative reasoning depth | $O(K \cdot S^2 d)$ | $O(h_{kv} d_h S)$ per active iteration | At rank 0, all iterations are identical map $T$; $\|\Delta h_{k+1}\| \le \gamma \|\Delta h_k\|$ |
| [diffusion](diffusion.md) | `diffusion` | Bidirectional Softmax Attention (no causal mask) | Bidirectional sequence field $S$ | $O(T_{diff} \cdot S^2 d)$ iterative denoising | $O(S d)$ per denoising step (no KV cache) | Bidirectional mask (all-ones); ELBO loss weight $\frac{1}{1-t} > 0$ |
| [blt](blt.md) | `blt` | Dynamic Entropy Patching: Local Conv + Patch Attention | $O(P)$ patches ($P \le \lceil S/K_{\max} \rceil$) | $O(P^2 d + S \cdot d_{local}^2)$ | $O(L \cdot P \cdot d)$ patch KV cache | Monotone patch boundaries; patch size $\in [1, K_{\max}]$ |
| [titans](titans.md) | `titans` | Sliding Window Attention + Test-Time Neural Memory | Window $W$ tokens + $M_t \in \mathbb{R}^{d_{mem}^2}$ | $O(W^2 d + d_{mem}^2)$ / tok | $O(W d)$ sliding cache $+ O(d_{mem}^2)$ memory | Surprise gradient update $M_t = (1-\alpha_t)M_{t-1} + \eta_t \nabla \mathcal{L}$; $\|M_t\| < \infty$ |
| [jepa](jepa.md) | `jepa` | Latent Target Prediction via Bidirectional Predictor | Latent abstract representation $z \in \mathbb{R}^d$ | $O(S d^2)$ forward evaluation | $O(S d)$ activations | $\mathrm{std}(z) > 0$ (collapse prevented by VICReg / stop-gradient EMA) |
| [kan](kan.md) | `kolmogorov-arnold` | Learnable Univariate B-Splines on Network Edges | Pointwise feature manifold | $O(S \cdot d^2 \cdot (G + k + 1))$ | $O(S d)$ activations (zero KV cache) | B-spline partition of unity $\sum_i B_{i, k}(x) = 1$ on $[0, G]$ |
| [neuromorphic](neuromorphic.md) | `neuromorphic` | Event-Driven Binary Spikes $s_t \in \{0, 1\}$ (LIF) | Membrane potential $V_t \in \mathbb{R}^d$ | $O(T_{sim} \cdot \alpha_{spike} \cdot d^2)$ (AC-only FLOPs) | $O(d)$ membrane state | $V_t$ bounded; surrogate gradient $\frac{\partial s}{\partial V}$ enables backpropagation |
| [multimodal](multimodal.md) | `native-multimodal` | Unified Modality-Masked Early-Fusion Attention | $O(S_{text} + S_{img} + S_{audio})$ cross-modal | $O((S_{text} + S_{img} + S_{audio})^2 d)$ | $O(L \cdot h_{kv} d_h (S_{text} + S_{img} + S_{audio}))$ | Modality-causal block mask prevents unconditioned cross-leakage |

*Note: Conceptual latent reasoning (e.g. TRM / HRM) is documented in [latent_reason.md](latent_reason.md).*

---

## Universal Block Contract

Every architecture satisfies the strict PyTorch block contract:
```python
block(x, **kw) -> (y, aux)
```

`aux` is a dictionary of diagnostic intermediate tensors used by the real-time hidden-space tracer:
- `attn/`: attention scores, entropy, routing weights
- `gate/`: forget, input, output, or expert routing gates
- `state/`: recurrent, SSM, cell, or memory states
- `mlp/`: feed-forward activations, gates, and projections
- `route/`: top-$k$ expert routing assignments
- `patch/`: dynamic byte-patch boundaries and entropy signals
- `latent/`: latent space projections and predictor representations

---

## Quick Start

```bash
# List all 17 executable architectures and symbolic specifications
./.venv/bin/python -m archmath list

# Print mathematical equations and complexity for an architecture
./.venv/bin/python -m archmath math rnn
./.venv/bin/python -m archmath math mlp

# Evaluate symbolic complexity polynomials
./.venv/bin/python -m archmath complexity mla --d-model 128

# Verify all 17 architectures (forward, backward, loss, and invariants)
./.venv/bin/python -m archmath verify

# Trace hidden space and gate dynamics during training
./.venv/bin/python -m archmath trace rnn --steps 10 --out rnn_run.jsonl --json rnn_report.json
./.venv/bin/python -m archmath trace mlp --steps 10 --out mlp_run.jsonl --json mlp_report.json

# Run cross-architecture benchmark suite
./.venv/bin/python -m archmath.bench --steps 6 --out bench.json

# Run SageMath formal symbolic verification
bash archmath/sage/run.sh
```

---

## Research Brief & Theoretical Landscape

For theoretical comparative analysis, scaling laws, memory bounds, and arXiv citations across all 17 model families, consult `../ai-architecture-landscape-2026.md`.
