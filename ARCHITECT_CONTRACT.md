"""Contract for architecture contributors.

Every architecture module must implement exactly three files and register itself.
Read `impl/dense.py` first -- it is the reference implementation.

--------------------------------------------------------------------------
FILE 1:  archmath/impl/<name>.py
--------------------------------------------------------------------------

    from ..core import ArchConfig, ArchModule, MathSpec, register, register_spec

    @register("<name>")
    class <Name>Model(ArchModule):
        AUX_KEYS = (...)

        def __init__(self, cfg: ArchConfig): ...

        def forward(self, tokens, **kw) -> tuple[Tensor, dict[str, Tensor]]:
            ...

Requirements
  * Accept `tokens` of shape [B, S] (long) for full models, OR a hidden-state
    block that accepts [B, S, d_model]. Document which in the docstring.
  * Return `(output, aux)`. `aux` MUST include at least one tensor per
    mathematical term in your spec, named with the prefixes below.
  * All returned tensors must be finite for finite inputs.
  * `cfg` keys you may rely on (all present in tiny_config):
        d_model, n_heads, n_kv_heads, d_head, n_layers, d_ff, vocab, seq
    additional keys must use `cfg.get(key, default)`.
  * Also export `def tiny_config(**over) -> ArchConfig`.
  * Must run forward AND backward on CPU in under ~2 seconds at d_model=64.

Aux key prefixes (the tracer groups by these)
    attn/*    attention probabilities, scores, masks
    route/*   router logits, expert ids, load
    state/*   recurrent state, memory slots, loop hidden states
    gate/*    scalar gates, decays, halting probabilities
    latent/*  compressed / latent representations
    patch/*   byte or patch boundaries
    world/*   JEPA predictions and targets
    mlp/*     feed-forward intermediates

--------------------------------------------------------------------------
FILE 2:  archmath/symbolic/<name>.py
--------------------------------------------------------------------------

    @register_spec("<name>")
    def <name>_spec() -> MathSpec:
        return MathSpec(
            name="<name>", family="<family>", summary="one sentence",
            equations=[("label", r"...latex..."), ...],
            assumptions=[...],
            complexity={"params": "<sympy expr>", "time": "<sympy expr>", ...},
            invariants=[...],
            references=["Author Year (arXiv:XXXX.XXXXX)", ...],
        )

`complexity` expressions are evaluated by sympy against the config.  Use ONLY
config key names and these functions: log, exp, sqrt, abs, max, min, ceil, floor.

--------------------------------------------------------------------------
FILE 3:  archmath/docs/<name>.md
--------------------------------------------------------------------------

A derivation, not a summary.  Structure:

    # <Name>
    ## 1. The object being computed        (definition)
    ## 2. Derivation                        (from first principles, show steps)
    ## 3. What each symbol does one at a time  (token-level walkthrough)
    ## 4. Hidden-space consequences         (what the tracer should see, and why)
    ## 5. Complexity and failure modes
    ## 6. References

--------------------------------------------------------------------------
Self-check before you finish
--------------------------------------------------------------------------

    cd /Users/mohana/Documents/Default\ Project
    ./.venv/bin/python -c "
    from archmath.impl.<name> import tiny_config
    from archmath.core import get_arch
    m = get_arch('<name>')(tiny_config())
    import torch
    x = torch.randint(0, 256, (2, 16))
    y, aux = m(x)
    y.sum().backward()
    print(y.shape, {k: tuple(v.shape) for k, v in aux.items()})
    "

If that prints finite shapes, you are done.  Do not modify shared files
(core.py, trace.py, impl/__init__.py, symbolic/__init__.py, cli.py).
"""
