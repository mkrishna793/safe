# Native Multimodal Early Fusion

## 1. The object being computed

A single Transformer over a unified token stream of text, image patches and
audio frames, all projected into the same $d$-dimensional space with
modality-type and position embeddings, with a modality-aware attention mask.

## 2. Derivation

**Unified stream**: $x = [\text{text tokens}; \text{image patches}; \text{audio frames}]$,
each projected by a modality-specific embedder into $\mathbb{R}^d$, plus a
modality-type embedding and per-modality position embedding.

**Modality-aware mask**: blocks that should not attend to each other (e.g.
audio and image are independent until text arrives) are masked.  The mask is
$M_{ij}=0$ if modalities $i,j$ are allowed to attend.

**Balanced loss**: per-token loss weighted by $1/(\text{modality token count})$
so no modality dominates the gradient.

**Late fusion baseline**: encode each modality separately, then project to a
shared space.  The projector is a rank-limited bottleneck: its information
capacity is bounded by its rank, which is the core argument for early fusion.

**Scaling result** (arXiv:2504.07951): early fusion has no disadvantage and
is better at low parameter counts; MoE lets modality-specific weights emerge.

## 3. Token-level walkthrough

Modality-specific embedders project heterogeneous inputs into one stream; a
single attention stack mixes them according to the mask; the balanced loss
keeps gradients fair across modalities.

## 4. Hidden-space consequences

- Per-modality loss diverges if unbalanced.
- Attention block structure reveals the mask.
- In MoE layers, modality-specific experts activate.

## 5. Complexity and failure modes

Params: embedders + shared Transformer.  The shared stack is the bottleneck
for modality-specific processing.

## 6. References

- Native multimodal roadmap (arXiv:2605.25343)
- Scaling laws for native multimodal (arXiv:2504.07951)
- DeepSeek-V4.1-Flash multimodal MoE (arXiv:2609.19969)
EOF
echo done