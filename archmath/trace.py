"""Real-time hidden-space tracing.

This is the *counter-frame* to the symbolic layer: the symbolic layer says what
should happen, the tracer measures what actually happens, per step, per layer,
per token, inside the residual stream.

What it answers:

  * Where does the signal live?          -> per-module norm / std / effective rank
  * Is a layer dead or saturated?        -> sparsity, kurtosis, rank collapse
  * Is attention sharp or diffuse?       -> entropy, max weight, sink mass
  * Are experts balanced?                -> router entropy, load histogram
  * Is the loop converging?              -> state delta across iterations
  * Which input tokens matter?           -> gradient x activation attribution
  * Is training healthy right now?       -> anomaly flags, drift, grad norms

Design constraints:

  * Zero changes to the model: everything is forward hooks + parameter grads.
  * Works on any nn.Module, not just ours.
  * Cheap enough to leave on: rank estimation is subsampled, stats are floats.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from .core import unwrap


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class TensorStats:
    """A fixed-width summary of one tensor at one step.

    Fixed width matters: the report is a time series, and a time series with
    varying keys is not a time series.
    """

    name: str
    step: int
    shape: tuple[int, ...]
    numel: int
    mean: float
    std: float
    norm: float
    max_abs: float
    kurtosis: float
    sparsity: float
    effective_rank: float
    entropy: float          # of the normalised magnitude spectrum, in nats
    dtype: str

    @classmethod
    def from_tensor(
        cls,
        t: torch.Tensor,
        name: str,
        step: int,
        *,
        max_rows: int = 256,
        want_rank: bool = True,
    ) -> "TensorStats":
        x = t.detach()
        if not x.is_floating_point():
            x = x.float()
        flat = x.reshape(-1, x.shape[-1]) if x.dim() >= 2 else x.reshape(1, -1)
        flat = flat.float()

        # subsample rows so rank estimation stays O(256 * d) not O(N * d)
        if flat.shape[0] > max_rows:
            idx = torch.linspace(0, flat.shape[0] - 1, max_rows, device=flat.device).long()
            sample = flat[idx]
        else:
            sample = flat

        finite = torch.isfinite(x)
        n_finite = int(finite.sum())
        if n_finite == 0:
            return cls(name, step, tuple(x.shape), x.numel(), *([float("nan")] * 10), str(x.dtype))

        xf = x[finite].float()
        mu = xf.mean()
        sigma = xf.std(unbiased=False)
        centered = xf - mu
        kurt = (centered.pow(4).mean() / (sigma.pow(4) + 1e-12) - 3.0) if sigma > 0 else 0.0

        er, spec_ent = 0.0, 0.0
        if want_rank and sample.numel() > 0:
            try:
                s = torch.linalg.svdvals(sample)
                s = s[s > 0]
                if s.numel():
                    er = float((s.sum() ** 2) / (s.pow(2).sum() + 1e-12))
                    p = s / (s.sum() + 1e-12)
                    spec_ent = float(-(p * p.log()).sum())
            except Exception:
                er, spec_ent = float("nan"), float("nan")

        return cls(
            name=name,
            step=step,
            shape=tuple(x.shape),
            numel=x.numel(),
            mean=float(mu),
            std=float(sigma),
            norm=float(xf.norm()),
            max_abs=float(xf.abs().max()),
            kurtosis=float(kurt),
            sparsity=float((xf.abs() < 1e-6).float().mean()),
            effective_rank=er,
            entropy=spec_ent,
            dtype=str(t.dtype),
        )

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["shape"] = list(self.shape)
        return d


def _attention_stats(p: torch.Tensor, name: str, step: int) -> dict[str, TensorStats]:
    """Entropy / sharpness / sink statistics for an attention probability tensor.

    Accepts [B, H, S, S] or [B, H, S, K].
    """
    out: dict[str, TensorStats] = {}
    with torch.no_grad():
        p = p.float()
        # normalize defensively (some blocks expose scores, not probs)
        if p.min() < 0 or not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-3):
            p = torch.softmax(p, dim=-1)
        ent = -(p.clamp_min(1e-9).log() * p).sum(-1)          # [B,H,S]
        maxw = p.max(-1).values
        last = p[..., -1, :] if p.dim() == 4 else p           # query = last token
        sink = last[..., 0] if last.shape[-1] > 1 else torch.zeros_like(last[..., 0])
        out[f"{name}/entropy"] = _scalar_stats(ent, f"{name}/entropy", step)
        out[f"{name}/max_weight"] = _scalar_stats(maxw, f"{name}/max_weight", step)
        out[f"{name}/sink_mass"] = _scalar_stats(sink, f"{name}/sink_mass", step)
    return out


def _scalar_stats(t: torch.Tensor, name: str, step: int) -> TensorStats:
    x = t.detach().float().reshape(-1)
    if x.numel() == 0:
        x = torch.zeros(1)
    return TensorStats(
        name=name, step=step, shape=tuple(t.shape), numel=t.numel(),
        mean=float(x.mean()), std=float(x.std(unbiased=False)),
        norm=float(x.norm()), max_abs=float(x.abs().max()),
        kurtosis=0.0, sparsity=float((x.abs() < 1e-6).float().mean()),
        effective_rank=1.0, entropy=0.0, dtype="float32",
    )


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #

class TraceSession:
    """Attach to a model, run steps, get a forensic record of every layer.

    Usage::

        with TraceSession(model, out="run.jsonl") as tr:
            for step, (x, y) in enumerate(loader):
                tr.begin_step(step, inputs=x)
                logits, aux = model(x)
                loss = criterion(logits, y)
                loss.backward()
                tr.end_step(loss=loss)

        print(tr.report().summary())
    """

    def __init__(
        self,
        model: nn.Module,
        out: str | Path | None = None,
        *,
        every: int = 1,
        max_rows: int = 256,
        record_rank: bool = True,
        record_grad: bool = True,
        record_attention: bool = True,
        record_input_attribution: bool = True,
        record_activation_grad: bool = False,
        include: str = "",
        exclude: str = "",
        anomaly_z: float = 4.0,
        tag: str = "",
    ) -> None:
        self.model = model
        self.out = Path(out) if out else None
        self.every = max(1, every)
        self.max_rows = max_rows
        self.record_rank = record_rank
        self.record_grad = record_grad
        self.record_attention = record_attention
        self.record_input_attribution = record_input_attribution
        self.record_activation_grad = record_activation_grad
        self._include = re.compile(include) if include else None
        self._exclude = re.compile(exclude) if exclude else None
        self.anomaly_z = anomaly_z
        self.tag = tag

        self._handles: list[Any] = []
        self._fwd: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}
        self._prev: dict[str, torch.Tensor] = {}
        self._records: list[dict[str, Any]] = []
        self._step = -1
        self._inputs: torch.Tensor | None = None
        self._loss: float | None = None
        self._t0 = time.time()
        self._open_file: Any = None
        self._anomalies: list[dict[str, Any]] = []

    # -- lifecycle -------------------------------------------------------- #

    def __enter__(self) -> "TraceSession":
        if self.out:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self._open_file = self.out.open("a", encoding="utf-8")
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if self._include and not self._include.search(name):
                continue
            if self._exclude and self._exclude.search(name):
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._open_file:
            self._open_file.close()
            self._open_file = None

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            if self._step % self.every != 0:
                return
            y, aux = unwrap(output)
            if isinstance(y, torch.Tensor):
                if self.record_activation_grad and y.requires_grad:
                    y.retain_grad()
                self._fwd[name] = (y, aux)
        return hook

    # -- step API --------------------------------------------------------- #

    def begin_step(self, step: int, inputs: torch.Tensor | None = None) -> None:
        self._step = step
        self._fwd.clear()
        self._loss = None
        self._inputs = None
        if inputs is not None:
            self.mark_input(inputs)

    def mark_input(self, x: torch.Tensor) -> None:
        """Retain gradient on the input so token attribution can be computed."""
        if x.is_floating_point() and x.requires_grad:
            x.retain_grad()
        self._inputs = x

    def end_step(self, loss: torch.Tensor | float | None = None) -> dict[str, Any]:
        if loss is not None:
            self._loss = float(loss) if not isinstance(loss, torch.Tensor) else float(loss.detach())
        if self._step % self.every != 0:
            return {}

        modules: dict[str, dict[str, Any]] = {}
        for name, (y, aux) in self._fwd.items():
            st = TensorStats.from_tensor(y, name, self._step, max_rows=self.max_rows, want_rank=self.record_rank)
            modules[f"{name}/out"] = st.as_dict()

            # hidden-space drift: how much did this layer move since last step?
            prev = self._prev.get(name)
            if prev is not None and prev.shape == y.shape:
                with torch.no_grad():
                    a = y.detach().float().reshape(-1, y.shape[-1])
                    b = prev.float().reshape(-1, prev.shape[-1])
                    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
                    modules[f"{name}/drift"] = _scalar_stats(1.0 - cos, f"{name}/drift", self._step).as_dict()
            self._prev[name] = y.detach()

            if self.record_activation_grad and y.grad is not None:
                modules[f"{name}/grad_out"] = TensorStats.from_tensor(
                    y.grad, f"{name}/grad_out", self._step, max_rows=self.max_rows, want_rank=False
                ).as_dict()

            if self.record_attention:
                for key, val in aux.items():
                    if not isinstance(val, torch.Tensor):
                        continue
                    if key.startswith("attn/") and val.dim() >= 3:
                        modules.update({k: v.as_dict() for k, v in _attention_stats(val, f"{name}/{key}", self._step).items()})
                        # token-level entropy: which positions are sharp, which are diffuse
                        with torch.no_grad():
                            pv = val.float()
                            if pv.min() < 0 or not torch.allclose(pv.sum(-1), torch.ones_like(pv.sum(-1)), atol=1e-3):
                                pv = torch.softmax(pv, dim=-1)
                            ent = -(pv.clamp_min(1e-9).log() * pv).sum(-1)      # [B,H,S]
                            per_tok = ent.mean(1).reshape(-1)                   # [B*S]
                            modules[f"{name}/{key}/entropy_tokens"] = {
                                "name": f"{name}/{key}/entropy_tokens",
                                "step": self._step,
                                "values": [float(v) for v in per_tok[:512]],
                            }
                    elif val.dim() == 0 or val.numel() < 4096:
                        modules[f"{name}/{key}"] = TensorStats.from_tensor(
                            val, f"{name}/{key}", self._step, max_rows=self.max_rows, want_rank=self.record_rank
                        ).as_dict()

        if self.record_grad:
            modules.update(self._grad_stats())

        if self.record_input_attribution and self._inputs is not None and self._inputs.grad is not None:
            with torch.no_grad():
                g = (self._inputs.grad.float() * self._inputs.detach().float()).abs().sum(-1)  # [B,S]
                modules["input/attribution"] = _scalar_stats(g, "input/attribution", self._step).as_dict()
                modules["input/attribution_per_token"] = {
                    "name": "input/attribution_per_token",
                    "step": self._step,
                    "values": [float(v) for v in g.reshape(-1)[:512]],
                }

        rec = {
            "tag": self.tag,
            "step": self._step,
            "loss": self._loss,
            "wall_ms": (time.time() - self._t0) * 1000.0,
            "modules": modules,
        }
        self._records.append(rec)
        self._anomalies.extend(self._detect(rec))
        if self._open_file:
            self._open_file.write(json.dumps(rec, allow_nan=True) + "\n")
        return rec

    def _grad_stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        groups: dict[str, list[float]] = defaultdict(list)
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            top = name.split(".")[0]
            groups[top].append(float(g.norm()))
        for k, v in groups.items():
            t = torch.tensor(v)
            out[f"grad/{k}"] = TensorStats(
                name=f"grad/{k}", step=self._step, shape=(len(v),), numel=len(v),
                mean=float(t.mean()), std=float(t.std(unbiased=False)), norm=float(t.norm()),
                max_abs=float(t.abs().max()), kurtosis=0.0, sparsity=0.0,
                effective_rank=1.0, entropy=0.0, dtype="float32",
            ).as_dict()
        return out

    # -- analysis --------------------------------------------------------- #

    def _detect(self, rec: dict[str, Any]) -> list[dict[str, Any]]:
        """Cheap, honest anomaly rules.  Not a monitor -- a smoke alarm."""
        found: list[dict[str, Any]] = []
        norms = [(k, v["norm"]) for k, v in rec["modules"].items() if k.endswith("/out") and "norm" in v]
        if len(norms) >= 3:
            vals = torch.tensor([n for _, n in norms if math.isfinite(n)])
            if vals.numel():
                med = vals.median()
                for k, n in norms:
                    if math.isfinite(n) and med > 0 and n > self.anomaly_z * med:
                        found.append({"step": rec["step"], "kind": "norm_outlier", "where": k, "value": n, "median": float(med)})
        for k, v in rec["modules"].items():
            if not isinstance(v, dict):
                continue
            if "std" in v and v["std"] == 0.0 and v.get("numel", 0) > 1:
                found.append({"step": rec["step"], "kind": "dead_tensor", "where": k})
            if k.endswith("/out") and v.get("effective_rank", 99) < 1.5 and v.get("numel", 0) > 16:
                found.append({"step": rec["step"], "kind": "rank_collapse", "where": k, "rank": v["effective_rank"]})
            if "entropy" in k and v.get("mean", 1) < 0.05:
                found.append({"step": rec["step"], "kind": "attention_collapse", "where": k})
            if v.get("norm") is not None and not math.isfinite(v["norm"]):
                found.append({"step": rec["step"], "kind": "non_finite", "where": k})
        return found

    def report(self) -> "HiddenSpaceReport":
        return HiddenSpaceReport(self._records, self._anomalies, self.tag)

    def last(self) -> dict[str, Any]:
        return self._records[-1] if self._records else {}


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #


def _f(v: Any, nd: int = 3) -> str:
    if v is None:
        return "--"
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "--"
    except Exception:
        pass
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}"
    return str(v)


def _spark(vals: list[float], width: int = 60) -> str:
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi != lo else 1.0
    bars = " ▂▃▄▅▆▇█"
    out = []
    for v in vals[:width]:
        idx = int((v - lo) / rng * (len(bars) - 1))
        idx = max(0, min(len(bars) - 1, idx))
        out.append(bars[idx])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

class HiddenSpaceReport:
    """Aggregated view over recorded steps."""

    def __init__(self, records: list[dict[str, Any]], anomalies: list[dict[str, Any]], tag: str = "") -> None:
        self.records = records
        self.anomalies = anomalies
        self.tag = tag

    # -- aggregate -------------------------------------------------------- #

    def series(self, key: str) -> list[tuple[int, float]]:
        out = []
        for r in self.records:
            v = r["modules"].get(key)
            if isinstance(v, dict) and "mean" in v:
                out.append((r["step"], v["mean"]))
        return out

    def token_attribution(self) -> list[float]:
        """Gradient x activation magnitude for every input token, last step."""
        if not self.records:
            return []
        v = self.records[-1]["modules"].get("input/attribution_per_token")
        return v["values"] if v else []

    def activation_entropy_profile(self, step: int | None = None) -> dict[str, list[float]]:
        """Per-layer, per-token attention entropy for the selected step."""
        rec = self.records[-1] if step is None else next((r for r in self.records if r["step"] == step), None)
        if rec is None:
            return {}
        return {
            k[: -len("/entropy_tokens")]: v["values"]
            for k, v in rec["modules"].items()
            if k.endswith("/entropy_tokens")
        }

    def layer_profile(self, metric: str = "norm", step: int | None = None) -> dict[str, float]:
        """metric vs layer for one step (default: last)."""
        rec = self.records[-1] if step is None else next((r for r in self.records if r["step"] == step), None)
        if rec is None:
            return {}
        return {
            k[: -len("/out")]: v[metric]
            for k, v in rec["modules"].items()
            if k.endswith("/out") and metric in v
        }

    def summary(self) -> str:
        if not self.records:
            return "no records"
        last = self.records[-1]
        lines = [
            f"ArchMath hidden-space report  [{self.tag}]",
            f"  steps recorded : {len(self.records)}",
            f"  final loss     : {last['loss']}",
            f"  modules traced : {len(last['modules'])}",
            f"  anomalies      : {len(self.anomalies)}",
        ]
        prof = self.layer_profile("norm")
        if prof:
            items = sorted(prof.items(), key=lambda kv: kv[1])
            lo, hi = items[0], items[-1]
            lines.append(f"  norm range     : {lo[0]}={lo[1]:.3g} .. {hi[0]}={hi[1]:.3g}")
        ranks = self.layer_profile("effective_rank")
        if ranks:
            vals = [v for v in ranks.values() if math.isfinite(v)]
            if vals:
                lines.append(f"  effective rank : min={min(vals):.1f} max={max(vals):.1f} (of {max(vals):.0f}+)")
        if self.anomalies:
            kinds = defaultdict(int)
            for a in self.anomalies:
                kinds[a["kind"]] += 1
            lines.append("  anomaly kinds  : " + ", ".join(f"{k}x{v}" for k, v in sorted(kinds.items())))
        return "\n".join(lines)

    def markdown(self, step: int | None = None, max_rows: int = 48) -> str:
        """A readable layer-by-layer table, for reports and for eyeballing."""
        rec = self.records[-1] if step is None else next((r for r in self.records if r["step"] == step), None)
        if rec is None:
            return "_no records_"

        rows = []
        for k, v in rec["modules"].items():
            if not k.endswith("/out") or not isinstance(v, dict):
                continue
            base = k[: -len("/out")]
            drift = rec["modules"].get(f"{base}/drift", {})
            grad = rec["modules"].get(f"{base}/grad_out", {})
            rows.append((
                base, v.get("norm"), v.get("std"), v.get("effective_rank"),
                v.get("kurtosis"), drift.get("mean"), grad.get("norm"),
            ))
        rows = rows[:max_rows]

        out = [
            f"### hidden-space profile — step {rec['step']}",
            "",
            "| module | ‖x‖ | σ | eff. rank | kurtosis | drift | ‖∇x‖ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for base, norm, std, rank, kurt, drift, gnorm in rows:
            out.append(
                f"| `{base}` | {_f(norm)} | {_f(std)} | {_f(rank)} | {_f(kurt, 2)} | "
                f"{_f(drift, 3)} | {_f(gnorm)} |"
            )

        attribution = self.token_attribution()
        if attribution:
            out += ["", "### input-token attribution (|∇x ⊙ x|, last step)", "", "```"]
            out.append(_spark(attribution))
            out.append("```")

        return "\n".join(out)

    def to_dict(self) -> dict[str, Any]:
        return {"tag": self.tag, "records": self.records, "anomalies": self.anomalies}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=True))
        return p
