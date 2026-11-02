# Neuromorphic (Spiking)

## 1. The object being computed

A network of leaky integrate-and-fire (LIF) neurons communicating via
discrete spikes, trained with surrogate gradients.  Event-driven and
energy-efficient, but with a capability gap at scale.

## 2. Derivation

**RC circuit**: $\tau\, dV/dt = -V(t)+R\,I(t)$.

**Discrete LIF**: $V_t = \beta V_{t-1} + W x_t - V_{\mathrm{th}} s_{t-1}$,
$s_t = \Theta(V_t - V_{\mathrm{th}})$ (Heaviside).

**Surrogate gradient**: forward uses the Heaviside; backward uses a smooth
surrogate (fast sigmoid or arctan) with slope $\alpha$.  This introduces bias:
the backward pass is not the true gradient, causing dead neurons and
gradient bias.

**Energy model**: $E \approx n_{\mathrm{spikes}}\cdot E_{\mathrm{MAC}}$ plus
leak.  Advantage requires sparse spike trains; dense firing removes it.

## 3. Token-level walkthrough

At each timestep, membrane potentials integrate input, neurons fire when
threshold is crossed, and the spike train carries the signal.  The surrogate
gradient lets the whole stack train end to end.

## 4. Hidden-space consequences

- `spike/rate` (firing rate per neuron) should be sparse (5-20%).
- `state/membrane` tracks sub-threshold dynamics.
- `spike/sparsity` is the energy proxy.

## 5. Complexity and honest limits

Energy advantage is real for sparse spikes on neuromorphic hardware, but there
is no credible LLM-scale spiking result as of 2026.  The capability gap
(accuracy vs. dense Transformers) remains large.

## 6. References

- Maass 1997 -- networks of spiking neurons
- Neftci et al. 2019 -- surrogate gradient learning
- "Nonlinear dynamical systems as physical neural networks" (arXiv:2610.40471)
EOF
echo done