"""Command line: list, math, complexity, trace, verify, export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import archmath.impl  # noqa: F401  (triggers registration)
import archmath.symbolic  # noqa: F401
from archmath.core import ArchConfig, get_arch, get_spec, list_archs, list_specs
from archmath.symbolic.base import complexity_report, contractivity_report
from archmath.trace import TraceSession


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _print_spec(name: str) -> None:
    s = get_spec(name)
    print(f"\n{'='*78}\n  {s.name}  [{s.family}]\n{'='*78}")
    print(f"\n{s.summary}\n")
    print("-- equations " + "-" * 64)
    for label, eq in s.equations:
        print(f"  [{label}]")
        print(f"    {eq}")
    print("\n-- assumptions " + "-" * 62)
    for a in s.assumptions:
        print(f"  * {a}")
    print("\n-- complexity " + "-" * 63)
    for k, v in s.complexity.items():
        print(f"  {k:<22} {v}")
    print("\n-- invariants " + "-" * 63)
    for i in s.invariants:
        print(f"  ! {i}")
    print("\n-- references " + "-" * 63)
    for r in s.references:
        print(f"  - {r}")
    print()


def _tiny_cfg(name: str) -> ArchConfig:
    mod = sys.modules.get(f"archmath.impl.{_module_for(name)}")
    if mod is not None and hasattr(mod, "tiny_config"):
        return mod.tiny_config()
    # fall back to a generic config
    return ArchConfig(name, "generic", dict(d_model=64, n_heads=4, n_kv_heads=2, d_head=16,
                                            n_layers=2, d_ff=128, vocab=256, seq=32))


def _module_for(name: str) -> str:
    """Map registry name -> module filename when they differ."""
    return {
        "sparse_attn": "sparse_attn",
        "linear_attn": "linear_attn",
        "latent_reason": "latent_reason",
    }.get(name, name)


def _build(name: str, cfg: ArchConfig):
    cls = get_arch(name)
    try:
        return cls(cfg)
    except TypeError:
        return cls(**cfg.params)


def _as_logits(out: torch.Tensor, d_model: int, vocab: int) -> torch.Tensor:
    if out.dim() == 3 and out.shape[-1] == vocab:
        return out
    if out.dim() == 3:
        head = torch.nn.Linear(out.shape[-1], vocab).to(out.device)
        return head(out)
    raise ValueError(f"cannot interpret output shape {tuple(out.shape)}")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_list(args: argparse.Namespace) -> None:
    archs = list_archs()
    specs = set(list_specs())
    print(f"{len(archs)} executable architectures, {len(specs)} symbolic specs\n")
    for a in archs:
        s = get_spec(a) if a in specs else None
        fam = s.family if s else "?"
        desc = s.summary if s else "(no spec)"
        mark = "OK " if s else "!! "
        print(f"  {mark} {a:<16} [{fam:<12}] {desc[:70]}")
    missing = specs - set(archs)
    if missing:
        print(f"\n  specs without implementation: {sorted(missing)}")
    if archmath.impl.ERRORS:
        print("\n  import errors:")
        for k, v in archmath.impl.ERRORS.items():
            print(f"    {k}: {v}")


def cmd_math(args: argparse.Namespace) -> None:
    _print_spec(args.arch)


def cmd_complexity(args: argparse.Namespace) -> None:
    cfg = _tiny_cfg(args.arch)
    if args.d_model:
        cfg = cfg.merged(d_model=args.d_model)
    rep = complexity_report(get_spec(args.arch), cfg)
    print(json.dumps({"config": cfg.params, "complexity": rep}, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    ok, fail = 0, 0
    torch.manual_seed(0)
    for name in list_archs():
        if args.arch and name != args.arch:
            continue
        try:
            cfg = _tiny_cfg(name)
            model = _build(name, cfg)
            model.train()
            _gv = 256 if name == "blt" else int(cfg.get("vocab", 256))
            tokens = torch.randint(0, _gv, (2, cfg.get("seq", 16)))
            out, aux = model(tokens)
            logits = _as_logits(out, cfg["d_model"], cfg.get("vocab", 256))
            loss = logits.float().logsumexp(-1).mean()
            loss.backward()
            finite = torch.isfinite(logits).all().item() and torch.isfinite(loss).item()
            grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            # attention rows sum to 1 invariant, where applicable
            attn_ok = True
            for k, v in aux.items():
                if k.endswith("attn/probs") and v.dim() >= 2:
                    s = v.float().sum(-1)
                    attn_ok &= bool(torch.allclose(s, torch.ones_like(s), atol=1e-3))
            status = "PASS" if (finite and grad_ok and attn_ok) else "FAIL"
            if status == "PASS":
                ok += 1
            else:
                fail += 1
            print(f"  [{status}] {name:<16} out={tuple(logits.shape)} aux={len(aux)} loss={float(loss):.4f} attn_rows={attn_ok}")
        except Exception as e:
            fail += 1
            print(f"  [ERROR] {name:<15} {type(e).__name__}: {e}")
    print(f"\n{ok} passed, {fail} failed")
    if fail:
        raise SystemExit(1)


def cmd_trace(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    cfg = _tiny_cfg(args.arch)
    cfg = cfg.merged(d_model=args.d_model, seq=args.seq, vocab=args.vocab)
    model = _build(args.arch, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_path = Path(args.out) if args.out else None
    with TraceSession(model, out=out_path, tag=args.arch, every=1) as tr:
        for step in range(args.steps):
            tokens = torch.randint(0, args.vocab, (args.batch, args.seq))
            tr.begin_step(step, inputs=None)
            out, aux = model(tokens)
            d = cfg["d_model"]
            if out.shape[-1] != args.vocab and out.dim() == 3:
                head = torch.nn.Linear(out.shape[-1], args.vocab)
                out = head(out)
            logits = out.float()
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, args.vocab), tokens[:, 1:].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            rec = tr.end_step(loss=loss)
            if args.verbose:
                print(f"step {step:>3}  loss {float(loss):.4f}")
    rep = tr.report()
    print("\n" + rep.summary())
    if args.out:
        print(f"\nrecords -> {args.out}")
    if args.json:
        rep.save(args.json)
        print(f"report  -> {args.json}")


def cmd_export(args: argparse.Namespace) -> None:
    data = []
    for name in list_specs():
        s = get_spec(name)
        data.append({
            "name": s.name, "family": s.family, "summary": s.summary,
            "equations": s.equations, "assumptions": s.assumptions,
            "complexity": s.complexity, "invariants": s.invariants,
            "references": s.references,
        })
    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"exported {len(data)} specs -> {args.out}")


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("archmath", description="Decode, rebuild and trace AI architectures.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list architectures and specs")

    m = sub.add_parser("math", help="print the symbolic spec")
    m.add_argument("arch")

    c = sub.add_parser("complexity", help="evaluate symbolic complexity")
    c.add_argument("arch")
    c.add_argument("--d-model", type=int, default=0)

    v = sub.add_parser("verify", help="forward/backward all architectures")
    v.add_argument("--arch", default="")

    t = sub.add_parser("trace", help="train briefly and report hidden-space telemetry")
    t.add_argument("arch")
    t.add_argument("--steps", type=int, default=5)
    t.add_argument("--seq", type=int, default=16)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--d-model", type=int, default=64)
    t.add_argument("--vocab", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="")
    t.add_argument("--json", default="")
    t.add_argument("--verbose", action="store_true")

    e = sub.add_parser("export", help="export all specs to JSON")
    e.add_argument("--out", default="archmath/specs.json")

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    {
        "list": cmd_list,
        "math": cmd_math,
        "complexity": cmd_complexity,
        "verify": cmd_verify,
        "trace": cmd_trace,
        "export": cmd_export,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
