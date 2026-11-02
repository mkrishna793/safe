# Recurrent Neural Networks (RNN, LSTM, GRU)

## 1. The object being computed

A sequence model with an iterative recurrent hidden state that maps an input sequence $x_1, \dots, x_S \in \mathbb{R}^d$ to hidden states $h_1, \dots, h_S \in \mathbb{R}^d$.

**Vanilla RNN** (Elman 1990):
$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

**LSTM with Forget Gates** (Hochreiter & Schmidhuber 1997; Gers et al. 2000):
$$
\begin{aligned}
f_t &= \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f) && \text{(forget gate)} \\
i_t &= \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i) && \text{(input gate)} \\
\tilde{c}_t &= \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c) && \text{(candidate cell)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t && \text{(cell state / CEC)} \\
o_t &= \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o) && \text{(output gate)} \\
h_t &= o_t \odot \tanh(c_t) && \text{(hidden output state)}
\end{aligned}
$$

**GRU** (Cho et al. 2014):
$$
\begin{aligned}
z_t &= \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z) && \text{(update gate)} \\
r_t &= \sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_r) && \text{(reset gate)} \\
\tilde{h}_t &= \tanh(W_{xh} x_t + W_{hh}(r_t \odot h_{t-1}) + b_h) && \text{(candidate state)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t && \text{(interpolated state)}
\end{aligned}
$$

The model is stacked into $L$ residual layers equipped with pre-layer RMSNorm and feed-forward SwiGLU/SiLU MLP blocks:
$$
y_l = x_l + h^{(l)}(x_l),\quad x_{l+1} = y_l + \mathrm{MLP}(\mathrm{RMSNorm}(y_l))
$$

---

## 2. Mathematical Derivation from first principles

### The Vanishing Gradient Problem in Vanilla RNNs
Backpropagation Through Time (BPTT) computes the gradient of loss $\mathcal{L}_T$ at time $T$ with respect to hidden state $h_t$ at time $t < T$:
$$
\frac{\partial \mathcal{L}_T}{\partial h_t} = \frac{\partial \mathcal{L}_T}{\partial h_T} \frac{\partial h_T}{\partial h_t} = \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}}
$$
For the vanilla RNN $h_k = \tanh(a_k)$ with $a_k = W_{xh} x_k + W_{hh} h_{k-1} + b_h$, the single-step Jacobian is:
$$
J_k = \frac{\partial h_k}{\partial h_{k-1}} = \operatorname{diag}(1 - h_k^2) W_{hh}^\top
$$
Taking the operator 2-norm:
$$
\left\|\frac{\partial h_T}{\partial h_t}\right\|_2 \le \prod_{k=t+1}^T \|J_k\|_2 \le \prod_{k=t+1}^T \left\|\operatorname{diag}(1 - h_k^2)\right\|_2 \|W_{hh}\|_2 \le \left(\gamma \|W_{hh}\|_2\right)^{T-t}
$$
Because $\tanh'(z) = 1 - \tanh^2(z) \in (0, 1]$, we have $\gamma \le 1$. If the spectral radius $\rho(W_{hh}) < 1$ or if hidden units saturate ($|h_k| \to 1 \implies 1 - h_k^2 \to 0$), the Jacobian norm decays exponentially:
$$
\left\|\frac{\partial h_T}{\partial h_t}\right\|_2 \sim \exp\left(-(T - t) / \tau\right) \to 0
$$
preventing the model from learning dependencies beyond temporal horizon $\tau = \frac{1}{|\ln \gamma|}$. Conversely, if $\|W_{hh}\|_2 > 1$ and states remain in the linear regime ($h \approx 0$), gradients explode exponentially.

### The Constant Error Carousel (CEC)
Hochreiter & Schmidhuber (1997) enforced constant error flow by designing an additive linear recurrence with unit gain:
$$
c_t = c_{t-1} + i_t \odot \tilde{c}_t \implies \frac{\partial c_t}{\partial c_{t-1}} = I
$$
In the presence of Gers et al.'s dynamic forget gates $f_t \in (0, 1)^d$, the cell state update becomes:
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$
Differentiating $c_T$ with respect to $c_t$ directly along the memory highway (neglecting the indirect loop through $h_{k-1} \to f_k, i_k, \tilde{c}_k$):
$$
\frac{\partial c_T}{\partial c_t} = \prod_{k=t+1}^T \frac{\partial c_k}{\partial c_{k-1}} = \prod_{k=t+1}^T \operatorname{diag}(f_k) = \operatorname{diag}\left(\prod_{k=t+1}^T f_{k, 1},\, \dots,\, \prod_{k=t+1}^T f_{k, d}\right)
$$
When the forget gate bias is initialized to a positive constant $b_f \ge 1.0$ (Jozefowicz et al. 2015), $f_k \approx \mathbf{1}$. Consequently:
$$
\frac{\partial c_T}{\partial c_t} \approx \prod_{k=t+1}^T I = I
$$
The gradient flows across arbitrarily long temporal intervals without vanishing or exploding. The forget gate dynamically modulates the effective memory half-life per channel: channel $j$ is preserved when $f_{k,j} \approx 1$ and erased when $f_{k,j} \approx 0$.

---

## 3. Token-level walkthrough

For token $t$ in sequence $x_{1:S}$:
1. **Pre-normalization**: Normalize input vector $x_t \in \mathbb{R}^d$ via RMSNorm:
   $$
   \hat{x}_t = \frac{x_t}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_{t,j}^2 + \epsilon}} \odot g
   $$
2. **Fused Affine Projection**: Compute pre-activations for all 4 gates in parallel:
   $$
   g_t = W_x \hat{x}_t + W_h h_{t-1} + b \in \mathbb{R}^{4d}
   $$
   where $b = [b_f,\, b_i,\, b_c,\, b_o]^\top$ with $b_f = \mathbf{1}$ and $b_i = b_c = b_o = \mathbf{0}$.
3. **Non-linear Gate Splitting**:
   $$
   f_t = \sigma(g_{t, [0:d]}),\quad i_t = \sigma(g_{t, [d:2d]}),\quad \tilde{c}_t = \tanh(g_{t, [2d:3d]}),\quad o_t = \sigma(g_{t, [3d:4d]})
   $$
4. **State Accumulation**:
   $$
   c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \in \mathbb{R}^d
   $$
5. **Output Gating & Bounded Readout**:
   $$
   h_t = o_t \odot \tanh(c_t) \in (-1, 1)^d
   $$
6. **Residual Stream Injection & Feedforward**:
   $$
   y_t = x_t + h_t,\quad x_{out, t} = y_t + \mathrm{MLP}(\mathrm{RMSNorm}(y_t))
   $$

---

## 4. Hidden-space consequences (what the tracer should see)

- **Jacobian Spectral Radius**: The tracer computes the singular values of recurrent projection matrix $W_h \in \mathbb{R}^{4d \times d}$. The effective recurrent spectral radius:
  $$
  \rho(W_h) = \max_i |\lambda_i(W_h)|
  $$
  is monitored at `state/w_hh_spectral_radius`. Stable propagation requires $\|W_{hf}\|_2, \|W_{hc}\|_2$ to remain well-conditioned.
- **Lyapunov Exponents & Dynamical Regimes**: For the autonomous cell map $c_t = f_t \odot c_{t-1}$, the $j$-th local Lyapunov exponent over sequence length $T$ is:
  $$
  \lambda_j = \frac{1}{T} \sum_{t=1}^T \ln f_{t, j}
  $$
  Since $f_{t, j} \in (0, 1)$, $\ln f_{t, j} \le 0$, ensuring $\lambda_j \le 0$. The system operates in a non-chaotic, contractive dynamical regime. As $f_{t,j} \to 1$, $\lambda_j \to 0^-$, placing the channel precisely at criticality (marginal stability, allowing non-decaying persistent memory).
- **Gate Saturation & Bounded State Norms**:
  - `gate/forget`: values near $1.0$ indicate active memory retention; drops toward $0.0$ denote boundary reset events.
  - `state/hidden`: strictly bounded inside $(-1, 1)^d$ due to outer $\tanh$ and sigmoid gating $o_t \in (0, 1)$.
  - `state/cell`: unbounded in theory, but stabilized in practice around $[-c_{\max}, c_{\max}]$ by the balance between $f_t$ decay and $i_t \odot \tilde{c}_t$ injection.

---

## 5. Complexity and failure modes

- **Parameters**:
  $$
  V d + L\Big(4(d^2 + d) + 4 d^2 + 2 d \cdot d_{ff}\Big) + d = V d + L\Big(8 d^2 + 4 d + 2 d \cdot d_{ff}\Big) + d
  $$
- **Time per token**: $O(L(8 d^2 + 2 d \cdot d_{ff}))$ arithmetic operations.
- **Sequence Processing**: $O(S \cdot L \cdot d^2)$ total FLOPs. **Crucially, inference and training along the sequence dimension are strictly sequential**: step $t+1$ cannot begin until step $t$ completes.
- **Memory at Inference**: $O(L \cdot d)$ total state ($h_t, c_t$). Unlike Transformers, **the inference memory is strictly $O(1)$ with respect to sequence length $S$ (no KV cache growth)**.
- **Failure modes**:
  - **Sequential Wall-Clock Bottleneck**: Cannot parallelize sequence processing across GPU cores during training (unlike FlashAttention or chunked SSMs).
  - **Bounded State Capacity (Memory Bottleneck)**: Storing the entire sequence history into a fixed vector of dimension $d$ fundamentally limits associative recall capacity to $O(d)$ bits, failing on long-range copy or multi-query retrieval tasks.
  - **Catastrophic Forgetting under Drift**: If forget gates drift toward $0$ across many consecutive tokens, early contextual history is permanently destroyed with no mechanism for non-local retrieval.

---

## 6. References

- Elman, J. L. (1990). *Finding structure in time*. Cognitive Science, 14(2), 179-211.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. Neural Computation, 9(8), 1735-1780.
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). *Learning to forget: Continual prediction with LSTM*. Neural Computation, 12(10), 2451-2471.
- Cho, K., et al. (2014). *Learning phrase representations using RNN encoder-decoder for statistical machine translation*. arXiv:1406.1078.
- Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). *An empirical exploration of recurrent network architectures*. ICML.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the difficulty of training recurrent neural networks*. ICML.
