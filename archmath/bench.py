"""Cross-architecture training benchmark with real-time hidden-space telemetry.

    ./.venv/bin/python -m archmath.bench --steps 6 --archs all

For every architecture it trains a tiny model on random tokens, records the full
hidden-space trace, and reports:

    params          -- actual parameter count
    loss 0 -> n     -- first and last training loss
    dL              -- loss change (negative = learning)
    norm[min,max]   -- residual-stream norm range across layers
    rank            -- mean effective rank of the residual stream
    attn H          -- mean attention entropy (nats), where exposed
    anomalies       -- tracer smoke-alarm count
    aux             -- number of traced intermediates

The point is not that a 64-dimensional model learns language.  The point is that
the same instrumentation runs unchanged on every architecture, so the *shape* of
the hidden-space dynamics can be compared.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import archmath.impl  # noqa: F401
import archmath.symbolic  # noqa: F401
from archmath.core import get_arch, list_archs
from archmath.trace import TraceSession


def _cfg(name: str):
    mod = sys.modules.get(f"archmath.impl.{name}")
    if mod is None or not hasattr(mod, "tiny_config"):
        raise RuntimeError(f"{name}: no tiny_config()")
    return mod.tiny_config()


def _logits(out: torch.Tensor, vocab: int) -> torch.Tensor:
    if out.dim() == 3 and out.shape[-1] == vocab:
        return out
    if out.dim() == 3:
        return torch.nn.Linear(out.shape[-1], vocab).to(out.device)(out)
    raise RuntimeError(f"unexpected output shape {tuple(out.shape)}")


def run_one(name: str, steps: int, seq: int, lr: float, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    cfg = _cfg(name)
    vocab = int(cfg.get("vocab", 256))
    model = get_arch(name)(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []

    t0 = time.time()
    with TraceSession(model, tag=name, every=1) as tr:
        for step in range(steps):
            gen_vocab = 256 if name == "blt" else vocab
            tokens = torch.randint(0, gen_vocab, (2, seq))
            tr.begin_step(step)
            out, _ = model(tokens)
            logits = _logits(out, vocab).float()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), tokens[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr.end_step(loss=loss)
            losses.append(float(loss.detach()))
    rep = tr.report()

    prof_norm = rep.layer_profile("norm")
    prof_rank = rep.layer_profile("effective_rank")
    last = rep.records[-1]["modules"] if rep.records else {}

    attn_h = [
        v["mean"] for k, v in last.items()
        if k.endswith("/entropy") and "attn" in k and isinstance(v, dict) and "mean" in v
    ]
    ranks = [v for v in prof_rank.values() if math.isfinite(v)]

    return {
        "arch": name,
        "params": n_params,
        "loss_first": losses[0] if losses else float("nan"),
        "loss_last": losses[-1] if losses else float("nan"),
        "delta_loss": (losses[-1] - losses[0]) if losses else float("nan"),
        "norm_min": min(prof_norm.values()) if prof_norm else float("nan"),
        "norm_max": max(prof_norm.values()) if prof_norm else float("nan"),
        "rank_mean": sum(ranks) / len(ranks) if ranks else float("nan"),
        "attn_entropy": sum(attn_h) / len(attn_h) if attn_h else float("nan"),
        "anomalies": len(rep.anomalies),
        "aux_tensors": len(last),
        "seconds": time.time() - t0,
        "report": rep,
    }


def fmt(v: float, nd: int = 3) -> str:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "  --"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("archmath.bench")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--archs", default="all", help="comma separated, or 'all'")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    names = list_archs() if a.archs == "all" else [s.strip() for s in a.archs.split(",") if s.strip()]
    rows: list[dict] = []

    print(f"training {len(names)} architectures  ({a.steps} steps, seq={a.seq}, lr={a.lr})\n")
    print(f"{'arch':<15}{'params':>10}{'loss0':>8}{'lossN':>8}{'dL':>8}{'norm_min':>9}{'norm_max':>9}{'rank':>7}{'attnH':>7}{'anom':>6}{'aux':>5}{'s':>6}")
    print("-" * 108)

    for name in names:
        try:
            r = run_one(name, a.steps, a.seq, a.lr)
        except Exception as e:
            print(f"{name:<15}  ERROR  {type(e).__name__}: {e}")
            continue
        rows.append(r)
        print(
            f"{r['arch']:<15}{r['params']:>10,}{fmt(r['loss_first']):>8}{fmt(r['loss_last']):>8}"
            f"{fmt(r['delta_loss'], 3):>8}{fmt(r['norm_min'], 1):>9}{fmt(r['norm_max'], 1):>9}"
            f"{fmt(r['rank_mean'], 1):>7}{fmt(r['attn_entropy'], 2):>7}{r['anomalies']:>6}"
            f"{r['aux_tensors']:>5}{r['seconds']:>6.1f}"
        )

    if a.out:
        p = Path(a.out)
        p.write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "report"} | {"summary": r["report"].summary()}
             for r in rows], indent=2, allow_nan=True
        ))
        print(f"\n-> {p}")

    print("\nlegend: dL = final - initial loss (negative = learning); rank = mean effective rank;")
    print("        attnH = mean attention entropy in nats (low = sharp); anom = tracer alarms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
