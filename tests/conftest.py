"""Bind subprocess-based tests to the exact checkout under pytest."""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOTS = (
    _ROOT / "src",
    _ROOT / "integrations/alfworld/src",
    _ROOT / "integrations/spreadsheetbench/src",
    _ROOT / "experimental/integrations/skillrl/src",
    _ROOT / "experimental/integrations/mce/src",
    _ROOT / "experimental/integrations/webshop/src",
)
_EXISTING = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = os.pathsep.join(
    (
        *(str(path) for path in _SOURCE_ROOTS),
        *((_EXISTING,) if _EXISTING else ()),
    )
)
