"""Integration tests across every registered architecture.

Run:  ./.venv/bin/python -m archmath.tests.test_archs
      ./.venv/bin/python -m archmath.tests.test_archs moe        # one arch

Each architecture is subjected to the same battery:

  1. registration + tiny config build
  2. forward produces finite logits/hidden states
  3. backward produces finite, non-zero gradients
  4. aux exposes at least one tensor and all aux tensors are finite
  5. attention probabilities (if exposed) are row-stochastic
  6. the tracer survives two training steps and records telemetry
  7. symbolic complexity evaluates to positive numbers
"""

from __future__ import annotations

import math
import sys
import traceback

import torch

import archmath.impl  # noqa: F401
import archmath.symbolic  # noqa: F401
from archmath.core import get_arch, get_spec, list_archs, list_specs
from archmath.symbolic.base import complexity_report
from archmath.trace import TraceSession


def _cfg(name: str):
    mod = sys.modules.get(f"archmath.impl.{name}")
    if mod is not None and hasattr(mod, "tiny_config"):
        return mod.tiny_config()
    raise AssertionError(f"archmath.impl.{name} does not export tiny_config()")


def _logits(out: torch.Tensor, vocab: int) -> torch.Tensor:
    if out.dim() == 3 and out.shape[-1] == vocab:
        return out
    if out.dim() == 3:
        return torch.nn.Linear(out.shape[-1], vocab)(out)
    raise AssertionError(f"unexpected output shape {tuple(out.shape)}")


def check(name: str) -> list[str]:
    fails: list[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    try:
        cfg = _cfg(name)
    except Exception as e:
        return [f"tiny_config: {type(e).__name__}: {e}"]

    vocab = int(cfg.get("vocab", 256))
    seq = int(cfg.get("seq", 16))

    try:
        model = get_arch(name)(cfg)
    except Exception as e:
        return [f"build: {type(e).__name__}: {e}"]

    model.train()
    torch.manual_seed(0)
    _gv = 256 if name == "blt" else vocab
    tokens = torch.randint(0, _gv, (2, seq))

    # -- forward ---------------------------------------------------------- #
    try:
        out, aux = model(tokens)
        logits = _logits(out, vocab)
    except Exception as e:
        return [f"forward: {type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"]

    expect(torch.isfinite(logits).all().item(), "non-finite logits")
    expect(len(aux) > 0, "aux is empty")
    bad = [k for k, v in aux.items() if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()]
    expect(not bad, f"non-finite aux: {bad[:4]}")

    # -- attention invariants --------------------------------------------- #
    for k, v in aux.items():
        if isinstance(v, torch.Tensor) and k.endswith("attn/probs") and v.dim() >= 2:
            s = v.float().sum(-1)
            expect(torch.allclose(s, torch.ones_like(s), atol=2e-3), f"{k} rows do not sum to 1")
            expect((v >= -1e-6).all().item(), f"{k} has negative probabilities")

    # -- backward --------------------------------------------------------- #
    try:
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab), tokens[:, 1:].reshape(-1)
        )
        loss.backward()
    except Exception as e:
        return fails + [f"backward: {type(e).__name__}: {e}"]

    expect(torch.isfinite(loss).item(), "non-finite loss")
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    expect(len(grads) > 0, "no gradients")
    expect(all(torch.isfinite(g).all() for g in grads), "non-finite gradients")
    expect(any(g.abs().sum() > 0 for g in grads), "all gradients are zero")

    # -- tracer ------------------------------------------------------------ #
    try:
        model.zero_grad(set_to_none=True)
        with TraceSession(model, tag=name, every=1) as tr:
            for step in range(2):
                tr.begin_step(step)
                o, _ = model(tokens)
                lg = _logits(o, vocab)
                l = torch.nn.functional.cross_entropy(
                    lg[:, :-1].reshape(-1, vocab), tokens[:, 1:].reshape(-1)
                )
                model.zero_grad(set_to_none=True)
                l.backward()
                tr.end_step(loss=l)
        rep = tr.report()
        expect(len(rep.records) == 2, "tracer recorded wrong number of steps")
        expect("modules" in rep.records[0] and len(rep.records[0]["modules"]) > 0, "tracer recorded no modules")
        nonfinite = [a for a in rep.anomalies if a["kind"] == "non_finite"]
        expect(not nonfinite, f"tracer found non-finite tensors: {nonfinite[:3]}")
    except Exception as e:
        fails.append(f"tracer: {type(e).__name__}: {e}")

    # -- symbolic spec ----------------------------------------------------- #
    if name in list_specs():
        try:
            spec = get_spec(name)
            expect(len(spec.equations) >= 3, "spec has fewer than 3 equations")
            expect(len(spec.assumptions) >= 1, "spec has no assumptions")
            comp = complexity_report(spec, cfg)
            expect(len(comp) >= 1, "spec has no complexity entries")
            expect(any(v > 0 for v in comp.values()), "all complexity entries are non-positive")
            for k, v in comp.items():
                expect(v >= 0, f"complexity {k}={v} is negative")
        except Exception as e:
            fails.append(f"spec: {type(e).__name__}: {e}")
    else:
        fails.append("no symbolic spec registered")

    return fails


def main(argv: list[str]) -> int:
    targets = argv[1:] or list_archs()
    print(f"ArchMath integration suite -- {len(targets)} architectures\n")
    total_fail = 0
    for name in targets:
        fails = check(name)
        if fails:
            total_fail += 1
            print(f"  [FAIL] {name}")
            for f in fails:
                print(f"         - {f}")
        else:
            print(f"  [PASS] {name}")
    print(f"\n{len(targets) - total_fail}/{len(targets)} architectures passed")
    if archmath.impl.ERRORS:
        print("\nimport errors:")
        for k, v in archmath.impl.ERRORS.items():
            print(f"  {k}: {v}")
    if archmath.symbolic.ERRORS:
        print("\nsymbolic import errors:")
        for k, v in archmath.symbolic.ERRORS.items():
            print(f"  {k}: {v}")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
