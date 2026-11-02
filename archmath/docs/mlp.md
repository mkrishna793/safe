# Deep Dense Network (Residual MLP)

## 1. The object being computed

A deep feed-forward coordinate transformation stack without spatial or sequence token mixing. Given a sequence of token representations $x_1, \dots, x_S \in \mathbb{R}^d$, the model maps each token $x_t$ through $L$ residual multi-layer perceptron blocks:
$$
x_{l+1} = x_l + \mathrm{FFN}(\mathrm{RMSNorm}(x_l))
$$
producing next-token logits $\ell_t \in \mathbb{R}^V$ via a tied unembedding head:
$$
\ell_t = W_{\text{vocab}} \mathrm{RMSNorm}(x_{L, t})
$$

### Feedforward Block Variants

**SwiGLU** (Shazeer 2020; standard modern parametrization):
$$
\mathrm{SwiGLU}(x) = \left(\mathrm{SiLU}(x W_g) \odot x W_u\right) W_d
$$
where $W_g, W_u \in \mathbb{R}^{d \times d_{ff}}$ are gate and up-projection matrices, and $W_d \in \mathbb{R}^{d_{ff} \times d}$ is the down-projection matrix, with typical expansion $d_{ff} = \frac{8}{3}d$.

**GELU / Standard 2-Layer MLP** (Hendrycks & Gimpel 2016):
$$
\mathrm{MLP}_{\text{GELU}}(x) = \mathrm{GELU}(x W_1 + b_1) W_2 + b_2,\quad W_1 \in \mathbb{R}^{d \times 4d},\quad W_2 \in \mathbb{R}^{4d \times d}
$$

The residual MLP acts as the control baseline against all sequence-mixing architectures (Transformers, State Space Models, RNNs): it models token features purely pointwise.

---

## 2. Mathematical Derivation from first principles

### Residual Stream as Discretized Dynamical System (Neural ODE)
Writing the residual update in discrete difference form:
$$
x_{l+1} - x_l = \Delta t \cdot f(x_l; \theta_l)
$$
In the continuous depth limit $L \to \infty$ with step size $\Delta t = \frac{1}{L} \to 0$, the residual stream is the forward Euler discretisation of an autonomous Ordinary Differential Equation:
$$
\frac{d x(t)}{dt} = f\big(x(t);\, \theta(t)\big),\quad x(0) = x_0 \in \mathbb{R}^d
$$
Each MLP layer computes a parameterized velocity vector field $f: \mathbb{R}^d \to \mathbb{R}^d$ that transports token representations along continuous trajectories on a $d$-dimensional feature manifold.

### The Neural Tangent Kernel (NTK) Regime
Consider an overparameterized deep dense network $f_\theta(x)$ trained by continuous gradient flow under mean squared error or cross-entropy:
$$
\frac{d\theta}{dt} = -\nabla_\theta \mathcal{L}(\theta) = -\sum_{i=1}^N \frac{\partial f_\theta(x_i)}{\partial \theta}^\top \nabla_{f}\mathcal{L}(f(x_i), y_i)
$$
By chain rule, the temporal evolution of network output $f_\theta(x)$ is governed by the Neural Tangent Kernel (Jacot, Gabriel, & Hongler 2018):
$$
\frac{d f_\theta(x)}{dt} = \sum_{p=1}^P \frac{\partial f_\theta(x)}{\partial \theta_p} \frac{d\theta_p}{dt} = -\sum_{i=1}^N \Theta_t(x, x_i) \nabla_f \mathcal{L}(f_\theta(x_i), y_i)
$$
where the empirical NTK is defined as:
$$
\Theta_t(x, x') = \left\langle \nabla_\theta f_\theta(x),\, \nabla_\theta f_\theta(x') \right\rangle = \sum_{p=1}^P \frac{\partial f_\theta(x)}{\partial \theta_p} \left(\frac{\partial f_\theta(x')}{\partial \theta_p}\right)^\top
$$
As hidden width $d_{ff} \to \infty$:
1. The kernel converges at initialization to a deterministic limit $\Theta_\infty(x, x')$.
2. During training, the parameters undergo infinitesimal displacement $\|\theta_t - \theta_0\| \sim O(1/\sqrt{d_{ff}})$, while $\sup_{t \ge 0} \|\Theta_t - \Theta_\infty\| \to 0$.
3. The network dynamics become strictly linear in function space:
   $$
   f_t(x) = f_0(x) + \Theta_\infty(x, X) \Theta_\infty(X, X)^{-1} \left(I - e^{-\Theta_\infty(X,X) t}\right) (Y - f_0(X))
   $$

### Coordinate Transformation and Spatial Decoupling
Because the MLP lacks any horizontal or sequence-mixing operator, the Jacobian between different sequence coordinates $t \neq s$ is identically zero:
$$
\frac{\partial x_{l+1, t}}{\partial x_{l, s}} = 0 \quad \forall t \neq s \in \{1, \dots, S\}
$$
The global sequence Jacobian matrix $\mathcal{J}_l \in \mathbb{R}^{Sd \times Sd}$ is strictly block-diagonal:
$$
\mathcal{J}_l = \begin{bmatrix}
J_{l, 1} & 0 & \dots & 0 \\
0 & J_{l, 2} & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & J_{l, S}
\end{bmatrix},\quad J_{l, t} = I_d + \frac{\partial \mathrm{FFN}(\hat{x}_{l, t})}{\partial x_{l, t}} \in \mathbb{R}^{d \times d}
$$
Every token evolves entirely along its own uncoupled 1D manifold stream.

---

## 3. Token-level walkthrough

For token $t$ at sequence position $s$:
1. **Token Embedding**:
   $$
   x_{0, t} = E_{x_t} \in \mathbb{R}^d
   $$
2. **RMS Normalisation**:
   $$
   \hat{x}_{l, t} = \frac{x_{l, t}}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_{l,t,j}^2 + \epsilon}} \odot g_l \in \mathbb{R}^d
   $$
3. **Dual Linear Projections**:
   $$
   g_{l, t} = \hat{x}_{l, t} W_g \in \mathbb{R}^{d_{ff}},\quad u_{l, t} = \hat{x}_{l, t} W_u \in \mathbb{R}^{d_{ff}}
   $$
4. **Swish Elementwise Gating**:
   $$
   a_{l, t} = \mathrm{SiLU}(g_{l, t}) \odot u_{l, t} = \left(g_{l, t} \odot \frac{1}{1 + e^{-g_{l, t}}}\right) \odot u_{l, t} \in \mathbb{R}^{d_{ff}}
   $$
5. **Down-projection**:
   $$
   \Delta x_{l, t} = a_{l, t} W_d \in \mathbb{R}^d
   $$
6. **Residual Stream Accumulation**:
   $$
   x_{l+1, t} = x_{l, t} + \Delta x_{l, t}
   $$
7. **Next-Token Prediction**:
   $$
   \ell_t = \mathrm{RMSNorm}(x_{L, t}) W_{\text{vocab}}^\top \in \mathbb{R}^V
   $$
Crucially, $\ell_t$ is a deterministic function solely of $x_t$. The prediction cannot take prefix tokens $x_{<t}$ into account.

---

## 4. Hidden-space consequences (what the tracer should see)

- **Total Absence of Cross-Token Covariance**:
  Evaluating cross-token correlations $\mathbb{E}_{t \ne s}[\langle x_{l, t}, x_{l, s}\rangle]$ reveals that feature similarities are determined entirely by static embedding similarities, with zero dynamic context-dependent routing.
- **Representation Rank Collapse**:
  Let $X_l \in \mathbb{R}^{S \times d}$ be the activation matrix at layer $l$, with singular value decomposition:
  $$
  X_l = U_l \Sigma_l V_l^\top,\quad \Sigma_l = \operatorname{diag}(\sigma_1, \sigma_2, \dots, \sigma_{\min(S, d)})
  $$
  Without token-mixing repulsion (such as self-attention or inter-token contrastive losses), iterated non-linear point-wise contractions push representation vectors into a narrow cone in $\mathbb{R}^d$.
  The **effective rank** (Roy & Vetterli 2007):
  $$
  \mathrm{erank}(X_l) = \exp\left(-\sum_{k=1}^r p_k \ln p_k\right),\quad p_k = \frac{\sigma_k}{\sum_{j=1}^r \sigma_j}
  $$
  decays monotonically with depth $l$, indicating severe dimensional collapse:
  $$
  \lim_{l \to \infty} \frac{\sigma_k(X_l)}{\sigma_1(X_l)} = 0 \quad \forall k \ge 2
  $$
- **Singular Value Spectrum of Weight Matrices**:
  The tracer inspects `mlp/gate`, `mlp/up`, `mlp/act`, `mlp/out`. The product matrix $W_d \operatorname{diag}(\sigma'(g)) W_u$ governs the local Lipschitz constant $\mathrm{Lip}(f) \le \|W_d\|_2 \|W_u\|_2$. When unconstrained, contraction collapses variance while expansion causes exploding pre-norm activations.

---

## 5. Complexity and failure modes

- **Parameters**:
  $$
  V d + L\Big(3 d \cdot d_{ff} + d\Big) + d = V d + L\Big(8 d^2 + d\Big) + d \quad \text{for } d_{ff} = \frac{8}{3}d
  $$
- **Time per token**:
  $$
  O(L \cdot d \cdot d_{ff}) = O(L \cdot d^2)
  $$
- **Sequence Processing**: $O(S \cdot L \cdot d \cdot d_{ff})$ total FLOPs. **Inference and training are 100% parallel across all sequence tokens $S$ with zero sequential dependencies ($O(1)$ sequential steps)**.
- **Memory at Inference**: $O(1)$ persistent state per token (zero KV cache, zero recurrent state buffer).
- **Failure modes**:
  - **Zero Contextual Capacity (Context Blindness)**: Cannot solve any task requiring information exchange across time steps (e.g. language modeling, associative recall, syntactic parsing, in-context learning). The model is mathematically equivalent to a unigram frequency predictor.
  - **Dimensional Collapse**: In deep unnormalized stacks without residual connections, representations collapse to a 1D line ($\mathrm{erank} \to 1$).
  - **Redundant Capacity**: Increasing sequence length $S$ provides zero additional reasoning context or working memory.

---

## 6. References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors*. Nature, 323(6088), 533-536.
- Jacot, A., Gabriel, F., & Hongler, C. (2018). *Neural tangent kernel: Convergence and generalization in neural networks*. NeurIPS.
- Shazeer, N. (2020). *GLU variants improve Transformer*. arXiv:2002.05202.
- Hendrycks, D., & Gimpel, K. (2016). *Gaussian error linear units (GELUs)*. arXiv:1606.08415.
- Weinan, E. (2017). *A proposal on machine learning via dynamical systems*. Communications in Mathematics and Statistics, 5(1), 1-11.
- Roy, O., & Vetterli, M. (2007). *The effective rank: A measure of effective dimensionality*. EUSIPCO.
- Dong, Y., Cordonnier, J. B., & Ma, A. (2021). *Attention is not all you need: Pure attention loses rank doubly exponentially*. ICML.
