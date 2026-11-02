"""Symbolic specifications, one module per architecture family."""

from __future__ import annotations

import importlib
import pkgutil

ERRORS: dict[str, str] = {}

for _m in pkgutil.iter_modules(__path__):
    if _m.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_m.name}")
    except Exception as _e:  # pragma: no cover
        ERRORS[_m.name] = f"{type(_e).__name__}: {_e}"
